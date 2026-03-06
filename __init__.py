"""
Agent Nodes Package
====================
Each file = one specialist agent or logical group of nodes.

  state.py          — Shared AgentState TypedDict
  base.py           — LLM clients, helpers, logger (shared utilities)
  supervisor.py     — Node 1:  classify_query
  retriever_node.py — Node 2:  retrieve
  crag.py           — Nodes 3-4: grade_documents, rewrite_query
  retrieval_agent.py— Node 5:  retrieval_agent (verification/general)
  synthesis_agent.py— Node 6:  synthesis_agent (comparison/tables)
  math_agent.py     — Nodes 7-8: math_agent + calculate
  self_rag.py       — Nodes 9-10: reflect + format_final_answer
"""

from .state import AgentState
from .supervisor import classify_query
from .retriever_node import retrieve
from .crag import grade_documents, rewrite_query
from .retrieval_agent import retrieval_agent
from .synthesis_agent import synthesis_agent
from .math_agent import math_agent, calculate
from .self_rag import reflect, format_final_answer

__all__ = [
    "AgentState",
    "classify_query",
    "retrieve",
    "grade_documents",
    "rewrite_query",
    "retrieval_agent",
    "synthesis_agent",
    "math_agent",
    "calculate",
    "reflect",
    "format_final_answer",
]
