"""
LangGraph Multi-Agent System
=============================
Graph nodes:
  1. classify_query      — Supervisor: routes to specialist agent
  2. retrieve            — HybridRetriever (BM25 + Dense + RRF + Rerank)
  3. grade_documents     — C-RAG: score each chunk, decide if relevant
  4. rewrite_query       — C-RAG: rewrite query if retrieval failed
  5. retrieval_agent     — Fact extraction + citation
  6. synthesis_agent     — Cross-table comparison/aggregation
  7. math_agent          — Number extraction + formula setup
  8. calculate           — Safe Python math evaluator
  9. reflect             — Self-RAG: grounding check on the answer
 10. format_final_answer — Clean, structured output

Conditional edges:
  grade_documents → retrieval_agent | synthesis_agent | math_agent | rewrite_query
  reflect         → format_final_answer | appropriate_agent (regenerate once)

State: AgentState (TypedDict)
"""

import os
import json
import math
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict, Literal, Annotated
from datetime import datetime

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import operator

from retriever import get_retriever
from prompts import (
    SUPERVISOR_SYSTEM, SUPERVISOR_USER,
    CRAG_GRADER_SYSTEM, CRAG_GRADER_USER,
    QUERY_REWRITER_SYSTEM, QUERY_REWRITER_USER,
    RETRIEVAL_AGENT_SYSTEM, RETRIEVAL_AGENT_USER,
    SYNTHESIS_AGENT_SYSTEM, SYNTHESIS_AGENT_USER,
    MATH_AGENT_SYSTEM, MATH_AGENT_USER,
    SELF_RAG_REFLECTOR_SYSTEM, SELF_RAG_REFLECTOR_USER,
    FINAL_FORMATTER_SYSTEM, FINAL_FORMATTER_USER,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ── Groq LLM clients ─────────────────────────────────────────────
MAIN_MODEL = os.getenv("GROQ_MAIN_MODEL", "llama-3.3-70b-versatile")
FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")
CRAG_THRESHOLD  = float(os.getenv("CRAG_RELEVANCE_THRESHOLD", 0.5))
SELFRAG_THRESHOLD = float(os.getenv("SELF_RAG_GROUNDING_THRESHOLD", 0.6))

def get_main_llm() -> ChatGroq:
    return ChatGroq(
        model=MAIN_MODEL,
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        max_retries=3,
    )

def get_fast_llm() -> ChatGroq:
    return ChatGroq(
        model=FAST_MODEL,
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        max_retries=3,
    )


# ── Helper: parse LLM JSON response ─────────────────────────────

def parse_json_response(text: str) -> Dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Strip markdown code fences if present
    if "```" in text:
        text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {}


def chunks_to_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a readable context string."""
    if not chunks:
        return "No chunks retrieved."
    parts = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        page = meta.get("page_number", "?")
        etype = meta.get("element_type", "text")
        heading = meta.get("section_heading", "")
        parts.append(
            f"--- Chunk {i} | Page {page} | Type: {etype} | Section: {heading} ---\n"
            f"{c['text']}\n"
        )
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# Agent State
# ═══════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    # Input
    query               : str
    # Routing
    query_type          : str    # verification | synthesis | forecasting | general
    routing_reason      : str
    # Retrieval
    retrieved_chunks    : List[Dict]
    retrieval_attempts  : int
    rewritten_query     : Optional[str]
    # CRAG grading
    relevant_chunks     : List[Dict]
    irrelevant_count    : int
    crag_passed         : bool
    # Agent outputs
    raw_answer          : Optional[str]
    calc_expression     : Optional[str]
    calc_result         : Optional[str]
    math_data_sources   : List[str]
    # Self-RAG
    grounding_score     : float
    grounding_passed    : bool
    ungrounded_claims   : List[str]
    regeneration_count  : int
    # Final
    final_answer        : Optional[str]
    # Trace
    steps               : List[Dict]
    start_time          : str
    error               : Optional[str]


def _log_step(state: AgentState, node: str, detail: str) -> List[Dict]:
    step = {
        "node"     : node,
        "detail"   : detail,
        "timestamp": datetime.utcnow().isoformat(),
    }
    logger.info(f"[{node}] {detail}")
    return state.get("steps", []) + [step]


# ═══════════════════════════════════════════════════════════════════
# Node 1: Classify Query (Supervisor)
# ═══════════════════════════════════════════════════════════════════

def classify_query(state: AgentState) -> AgentState:
    llm   = get_fast_llm()
    query = state["query"]

    response = llm.invoke([
        SystemMessage(content=SUPERVISOR_SYSTEM),
        HumanMessage(content=SUPERVISOR_USER.format(query=query)),
    ])

    parsed = parse_json_response(response.content)
    qtype  = parsed.get("query_type", "general")
    reason = parsed.get("reasoning", "")

    steps = _log_step(state, "classify_query",
                      f"type={qtype} | reason={reason}")

    return {
        **state,
        "query_type"    : qtype,
        "routing_reason": reason,
        "steps"         : steps,
        "retrieval_attempts": 0,
        "regeneration_count": 0,
        "retrieved_chunks"  : [],
        "relevant_chunks"   : [],
        "math_data_sources" : [],
        "ungrounded_claims" : [],
        "grounding_score"   : 0.0,
    }


# ═══════════════════════════════════════════════════════════════════
# Node 2: Retrieve (Hybrid RAG)
# ═══════════════════════════════════════════════════════════════════

def retrieve(state: AgentState) -> AgentState:
    retriever = get_retriever()
    # Use rewritten query if available
    query = state.get("rewritten_query") or state["query"]

    chunks = retriever.retrieve(query)

    steps = _log_step(state, "retrieve",
                      f"query='{query[:60]}…' → {len(chunks)} chunks retrieved")

    return {
        **state,
        "retrieved_chunks" : chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "steps"            : steps,
    }


# ═══════════════════════════════════════════════════════════════════
# Node 3: Grade Documents (C-RAG)
# ═══════════════════════════════════════════════════════════════════

def grade_documents(state: AgentState) -> AgentState:
    """
    C-RAG: evaluate each retrieved chunk for relevance to the query.
    Uses fast 8b model — one call per chunk.
    """
    llm    = get_fast_llm()
    query  = state["query"]
    chunks = state["retrieved_chunks"]

    relevant   = []
    irrelevant = 0

    for chunk in chunks:
        response = llm.invoke([
            SystemMessage(content=CRAG_GRADER_SYSTEM),
            HumanMessage(content=CRAG_GRADER_USER.format(
                query=query,
                chunk=chunk["text"][:600],  # truncate for speed
            )),
        ])
        parsed = parse_json_response(response.content)
        is_rel   = parsed.get("relevant", False)
        conf     = float(parsed.get("confidence", 0.5))

        if is_rel and conf >= CRAG_THRESHOLD:
            chunk["crag_confidence"] = conf
            relevant.append(chunk)
        else:
            irrelevant += 1

    crag_passed = len(relevant) >= 2

    steps = _log_step(state, "grade_documents",
                      f"relevant={len(relevant)} irrelevant={irrelevant} "
                      f"passed={crag_passed}")

    return {
        **state,
        "relevant_chunks": relevant,
        "irrelevant_count": irrelevant,
        "crag_passed"    : crag_passed,
        "steps"          : steps,
    }


# ═══════════════════════════════════════════════════════════════════
# Node 4: Rewrite Query (C-RAG fallback)
# ═══════════════════════════════════════════════════════════════════

def rewrite_query(state: AgentState) -> AgentState:
    llm = get_fast_llm()

    reason = (f"Only {len(state.get('relevant_chunks', []))} relevant chunks found "
              f"out of {len(state.get('retrieved_chunks', []))} retrieved.")

    response = llm.invoke([
        SystemMessage(content=QUERY_REWRITER_SYSTEM),
        HumanMessage(content=QUERY_REWRITER_USER.format(
            query=state["query"],
            reason=reason,
        )),
    ])
    parsed   = parse_json_response(response.content)
    new_query = parsed.get("rewritten_query", state["query"])
    strategy  = parsed.get("strategy", "")

    steps = _log_step(state, "rewrite_query",
                      f"new='{new_query[:60]}…' strategy={strategy}")

    return {**state, "rewritten_query": new_query, "steps": steps}


# ═══════════════════════════════════════════════════════════════════
# Node 5: Retrieval Agent (verification / general)
# ═══════════════════════════════════════════════════════════════════

def retrieval_agent(state: AgentState) -> AgentState:
    llm    = get_main_llm()
    chunks = state.get("relevant_chunks") or state.get("retrieved_chunks", [])

    # For verification queries, also pull page content for exact citations
    if state.get("query_type") == "verification" and chunks:
        retriever = get_retriever()
        pages = {c["metadata"].get("page_number") for c in chunks}
        extra = []
        for pg in list(pages)[:3]:  # max 3 pages
            if pg:
                extra.extend(retriever.get_page_chunks(pg))
        # Merge without duplicates
        seen = {c["text"][:80] for c in chunks}
        for e in extra:
            if e["text"][:80] not in seen:
                chunks.append(e)
                seen.add(e["text"][:80])

    context = chunks_to_context(chunks)

    response = get_main_llm().invoke([
        SystemMessage(content=RETRIEVAL_AGENT_SYSTEM),
        HumanMessage(content=RETRIEVAL_AGENT_USER.format(
            query=state["query"],
            chunks=context,
        )),
    ])

    steps = _log_step(state, "retrieval_agent",
                      f"answer preview: {response.content[:120]}…")

    return {**state, "raw_answer": response.content, "steps": steps}


# ═══════════════════════════════════════════════════════════════════
# Node 6: Synthesis Agent (cross-table comparison)
# ═══════════════════════════════════════════════════════════════════

def synthesis_agent(state: AgentState) -> AgentState:
    chunks = state.get("relevant_chunks") or state.get("retrieved_chunks", [])
    context = chunks_to_context(chunks)

    response = get_main_llm().invoke([
        SystemMessage(content=SYNTHESIS_AGENT_SYSTEM),
        HumanMessage(content=SYNTHESIS_AGENT_USER.format(
            query=state["query"],
            chunks=context,
        )),
    ])

    steps = _log_step(state, "synthesis_agent",
                      f"answer preview: {response.content[:120]}…")

    return {**state, "raw_answer": response.content, "steps": steps}


# ═══════════════════════════════════════════════════════════════════
# Node 7: Math Agent (number extraction + formula)
# ═══════════════════════════════════════════════════════════════════

def math_agent(state: AgentState) -> AgentState:
    chunks  = state.get("relevant_chunks") or state.get("retrieved_chunks", [])
    context = chunks_to_context(chunks)

    response = get_main_llm().invoke([
        SystemMessage(content=MATH_AGENT_SYSTEM),
        HumanMessage(content=MATH_AGENT_USER.format(
            query=state["query"],
            chunks=context,
        )),
    ])

    parsed = parse_json_response(response.content)
    expression   = parsed.get("expression", "")
    data_sources = parsed.get("data_sources", [])

    # Build a descriptive raw answer with the extracted values
    raw = (
        f"Extracted values:\n"
        f"  Start value : {parsed.get('start_value', 'N/A')} ({parsed.get('start_year', 'N/A')})\n"
        f"  End value   : {parsed.get('end_value', 'N/A')} ({parsed.get('end_year', 'N/A')})\n"
        f"  Years       : {parsed.get('n_years', 'N/A')}\n"
        f"  Formula     : {parsed.get('formula', 'N/A')}\n"
        f"  Expression  : {expression}\n"
        f"Data sources  : {'; '.join(data_sources)}"
    )

    steps = _log_step(state, "math_agent",
                      f"expression={expression} sources={data_sources}")

    return {
        **state,
        "raw_answer"       : raw,
        "calc_expression"  : expression,
        "math_data_sources": data_sources,
        "steps"            : steps,
    }


# ═══════════════════════════════════════════════════════════════════
# Node 8: Calculator (safe math eval)
# ═══════════════════════════════════════════════════════════════════

def calculate(state: AgentState) -> AgentState:
    expression = state.get("calc_expression", "")

    if not expression:
        steps = _log_step(state, "calculate", "No expression to evaluate — skipping")
        return {**state, "calc_result": None, "steps": steps}

    safe_globals = {
        "__builtins__": {},
        "math": math,
        "abs" : abs,
        "round": round,
        "pow"  : pow,
    }
    try:
        result = eval(expression, safe_globals, {})
        if isinstance(result, float) and 0 < result < 1:
            formatted = (
                f"Result     : {result:.8f}\n"
                f"As percentage : {result * 100:.4f}%\n"
                f"Expression : {expression}"
            )
        else:
            formatted = (
                f"Result     : {result}\n"
                f"Expression : {expression}"
            )
        steps = _log_step(state, "calculate", f"result={result}")
        return {**state, "calc_result": formatted, "steps": steps}
    except Exception as e:
        err = f"Calculation error: {e} | Expression: {expression}"
        steps = _log_step(state, "calculate", err)
        return {**state, "calc_result": err, "steps": steps}


# ═══════════════════════════════════════════════════════════════════
# Node 9: Self-RAG Reflection
# ═══════════════════════════════════════════════════════════════════

def reflect(state: AgentState) -> AgentState:
    """
    Self-RAG: verify that the generated answer is grounded in source chunks.
    Uses fast model for efficiency.
    """
    llm         = get_fast_llm()
    raw_answer  = state.get("raw_answer", "")
    chunks      = state.get("relevant_chunks") or state.get("retrieved_chunks", [])
    context     = chunks_to_context(chunks[:4])   # top 4 for grounding check

    response = llm.invoke([
        SystemMessage(content=SELF_RAG_REFLECTOR_SYSTEM),
        HumanMessage(content=SELF_RAG_REFLECTOR_USER.format(
            answer=raw_answer,
            chunks=context,
        )),
    ])

    parsed = parse_json_response(response.content)
    grounding_score  = float(parsed.get("grounding_score", 0.5))
    grounding_passed = grounding_score >= SELFRAG_THRESHOLD
    ungrounded       = parsed.get("ungrounded_claims", [])
    verdict          = parsed.get("verdict", "")

    steps = _log_step(state, "reflect",
                      f"score={grounding_score:.2f} passed={grounding_passed} "
                      f"verdict={verdict}")

    return {
        **state,
        "grounding_score"  : grounding_score,
        "grounding_passed" : grounding_passed,
        "ungrounded_claims": ungrounded,
        "steps"            : steps,
    }


# ═══════════════════════════════════════════════════════════════════
# Node 10: Format Final Answer
# ═══════════════════════════════════════════════════════════════════

def format_final_answer(state: AgentState) -> AgentState:
    response = get_main_llm().invoke([
        SystemMessage(content=FINAL_FORMATTER_SYSTEM),
        HumanMessage(content=FINAL_FORMATTER_USER.format(
            query=state["query"],
            raw_answer=state.get("raw_answer", ""),
            grounding_score=state.get("grounding_score", 0),
            calc_result=state.get("calc_result") or "N/A",
        )),
    ])

    steps = _log_step(state, "format_final_answer", "Final answer formatted")

    return {**state, "final_answer": response.content, "steps": steps}


# ═══════════════════════════════════════════════════════════════════
# Conditional Edge Functions
# ═══════════════════════════════════════════════════════════════════

def route_after_grading(state: AgentState) -> str:
    """After C-RAG grading, route to specialist agent or rewrite query."""
    # If retrieval failed and we haven't retried yet → rewrite query
    if not state.get("crag_passed") and state.get("retrieval_attempts", 0) < 2:
        return "rewrite_query"

    # Route by query type
    qtype = state.get("query_type", "general")
    if qtype == "verification":
        return "retrieval_agent"
    elif qtype == "synthesis":
        return "synthesis_agent"
    elif qtype == "forecasting":
        return "math_agent"
    else:
        return "retrieval_agent"


def route_after_math(state: AgentState) -> str:
    """After math agent, go to calculator if expression was generated."""
    if state.get("calc_expression"):
        return "calculate"
    return "reflect"


def route_after_reflect(state: AgentState) -> str:
    """
    Self-RAG decision:
    - Passed → format final answer
    - Failed but haven't regenerated → re-run specialist agent once
    - Failed and already regenerated → format anyway (with caveat)
    """
    if state.get("grounding_passed"):
        return "format_final_answer"
    if state.get("regeneration_count", 0) < 1:
        # Increment regeneration count and re-run agent
        qtype = state.get("query_type", "general")
        if qtype == "forecasting":
            return "math_agent"
        elif qtype == "synthesis":
            return "synthesis_agent"
        else:
            return "retrieval_agent"
    # Already regenerated once → accept and format
    return "format_final_answer"


def increment_regen(state: AgentState) -> AgentState:
    """Helper: increment regeneration count (called on re-run path)."""
    return {**state, "regeneration_count": state.get("regeneration_count", 0) + 1}


# ═══════════════════════════════════════════════════════════════════
# Build the LangGraph
# ═══════════════════════════════════════════════════════════════════

def build_graph() -> Any:
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classify_query",       classify_query)
    workflow.add_node("retrieve",             retrieve)
    workflow.add_node("grade_documents",      grade_documents)
    workflow.add_node("rewrite_query",        rewrite_query)
    workflow.add_node("retrieval_agent",      retrieval_agent)
    workflow.add_node("synthesis_agent",      synthesis_agent)
    workflow.add_node("math_agent",           math_agent)
    workflow.add_node("calculate",            calculate)
    workflow.add_node("reflect",              reflect)
    workflow.add_node("format_final_answer",  format_final_answer)

    # Entry point
    workflow.set_entry_point("classify_query")

    # Linear edges
    workflow.add_edge("classify_query",      "retrieve")
    workflow.add_edge("retrieve",            "grade_documents")
    workflow.add_edge("rewrite_query",       "retrieve")         # CRAG retry loop

    # Conditional: after grading → route to specialist or rewrite
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "rewrite_query"  : "rewrite_query",
            "retrieval_agent": "retrieval_agent",
            "synthesis_agent": "synthesis_agent",
            "math_agent"     : "math_agent",
        }
    )

    # Specialist → reflect (with calc step for math)
    workflow.add_conditional_edges(
        "math_agent",
        route_after_math,
        {
            "calculate": "calculate",
            "reflect"  : "reflect",
        }
    )
    workflow.add_edge("calculate",       "reflect")
    workflow.add_edge("retrieval_agent", "reflect")
    workflow.add_edge("synthesis_agent", "reflect")

    # Self-RAG conditional
    workflow.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "format_final_answer": "format_final_answer",
            "retrieval_agent"    : "retrieval_agent",
            "synthesis_agent"    : "synthesis_agent",
            "math_agent"         : "math_agent",
        }
    )

    workflow.add_edge("format_final_answer", END)

    return workflow.compile()


# ═══════════════════════════════════════════════════════════════════
# Public Query Runner
# ═══════════════════════════════════════════════════════════════════

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(query: str) -> Dict:
    """
    Run a query through the full multi-agent pipeline.
    Returns structured response with answer, trace, and metadata.
    """
    graph = get_graph()

    initial_state: AgentState = {
        "query"              : query,
        "query_type"         : "",
        "routing_reason"     : "",
        "retrieved_chunks"   : [],
        "retrieval_attempts" : 0,
        "rewritten_query"    : None,
        "relevant_chunks"    : [],
        "irrelevant_count"   : 0,
        "crag_passed"        : False,
        "raw_answer"         : None,
        "calc_expression"    : None,
        "calc_result"        : None,
        "math_data_sources"  : [],
        "grounding_score"    : 0.0,
        "grounding_passed"   : False,
        "ungrounded_claims"  : [],
        "regeneration_count" : 0,
        "final_answer"       : None,
        "steps"              : [],
        "start_time"         : datetime.utcnow().isoformat(),
        "error"              : None,
    }

    try:
        final_state = graph.invoke(initial_state)
        return {
            "query"          : query,
            "answer"         : final_state.get("final_answer") or final_state.get("raw_answer"),
            "query_type"     : final_state.get("query_type"),
            "grounding_score": final_state.get("grounding_score"),
            "calc_result"    : final_state.get("calc_result"),
            "crag_passed"    : final_state.get("crag_passed"),
            "retrieval_attempts": final_state.get("retrieval_attempts"),
            "steps"          : final_state.get("steps", []),
            "status"         : "success",
            "start_time"     : initial_state["start_time"],
            "end_time"       : datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Graph execution error: {e}", exc_info=True)
        return {
            "query" : query,
            "answer": None,
            "error" : str(e),
            "status": "error",
            "steps" : initial_state["steps"],
            "start_time": initial_state["start_time"],
            "end_time"  : datetime.utcnow().isoformat(),
        }
