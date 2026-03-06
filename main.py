"""
FastAPI Backend
================
Endpoints:
  POST /query          — main query endpoint (full agent pipeline)
  POST /ingest         — trigger ETL re-ingestion
  GET  /health         — system status
  GET  /traces         — list saved traces
  GET  /traces/{id}    — retrieve specific trace
  GET  /graph/diagram  — Mermaid diagram of the LangGraph
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from graph import run_query

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

LOG_DIR = Path("./logs/traces")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Cyber Ireland 2022 — Multi-Agent RAG System",
    description=(
        "LangGraph multi-agent system with Hybrid RAG (BM25 + Dense + RRF), "
        "Cross-Encoder Reranking, C-RAG, and Self-RAG for the "
        "Cyber Ireland 2022 cybersecurity sector report.\n\n"
        "LLMs: Groq Llama-3.3-70b (main) + Llama-3.1-8b (grader/reflector)\n"
        "Embeddings: BAAI/bge-large-en-v1.5 (local)"
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        example="What is the total number of jobs reported and where is this stated?",
    )
    save_trace: bool = Field(default=True)


class QueryResponse(BaseModel):
    query           : str
    answer          : Optional[str]
    status          : str
    query_type      : Optional[str]
    grounding_score : Optional[float]
    calc_result     : Optional[str]
    crag_passed     : Optional[bool]
    retrieval_attempts: Optional[int]
    duration_seconds: float
    trace_id        : Optional[str]
    steps           : list


class IngestRequest(BaseModel):
    pdf_path: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────
@app.get("/", tags=["UI"], response_class=HTMLResponse)
@app.get("/ui", tags=["UI"], response_class=HTMLResponse)
def serve_ui():
    """
    Built-in testing dashboard.
    Open http://localhost:8000/ui in your browser — no extra setup needed.
    The UI is served directly from FastAPI so CORS is never an issue.
    """
    ui_path = Path(__file__).parent / "ui" / "index.html"
    if not ui_path.exists():
        return HTMLResponse(
            "<h2 style='font-family:monospace;padding:40px'>UI not found."
            "<br>Place cyber_rag_v2_ui.html at <code>ui/index.html</code></h2>",
            status_code=404
        )
    return HTMLResponse(ui_path.read_text())


@app.get("/health", tags=["System"])
def health():
    return {
        "status"   : "online",
        "service"  : "Cyber Ireland 2022 Multi-Agent RAG",
        "version"  : "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "models"   : {
            "main_llm"  : os.getenv("GROQ_MAIN_MODEL"),
            "fast_llm"  : os.getenv("GROQ_FAST_MODEL"),
            "embeddings": os.getenv("EMBEDDING_MODEL"),
        }
    }


@app.post("/query", response_model=QueryResponse, tags=["Agent"])
def query_endpoint(request: QueryRequest):
    """
    Submit a query to the multi-agent pipeline.

    The system will:
    1. **Classify** the query (Supervisor)
    2. **Retrieve** via Hybrid RAG (BM25 + Dense + RRF + Cross-Encoder)
    3. **Grade** retrieved chunks (C-RAG)
    4. **Rewrite** query if grading fails (C-RAG retry)
    5. Route to **specialist agent** (Retrieval / Synthesis / Math)
    6. **Calculate** if math expression was generated
    7. **Reflect** on grounding (Self-RAG)
    8. **Format** and return final answer
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info(f"📥 Query: {request.query!r}")
    start = time.time()

    result = run_query(request.query)
    duration = round(time.time() - start, 2)

    if result["status"] == "error":
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {result.get('error', 'Unknown')}"
        )

    # Save trace
    trace_id = None
    if request.save_trace:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c if c.isalnum() else "_" for c in request.query[:40])
        trace_file = LOG_DIR / f"{ts}_{safe}.json"
        trace_data = {**result, "duration_seconds": duration}
        with open(trace_file, "w") as f:
            json.dump(trace_data, f, indent=2, default=str)
        trace_id = trace_file.name
        logger.info(f"💾 Trace saved: {trace_id}")

    logger.info(f"✅ Done in {duration}s")

    return QueryResponse(
        query             = result["query"],
        answer            = result.get("answer"),
        status            = result["status"],
        query_type        = result.get("query_type"),
        grounding_score   = result.get("grounding_score"),
        calc_result       = result.get("calc_result"),
        crag_passed       = result.get("crag_passed"),
        retrieval_attempts= result.get("retrieval_attempts"),
        duration_seconds  = duration,
        trace_id          = trace_id,
        steps             = result.get("steps", []),
    )


@app.post("/ingest", tags=["System"])
def ingest_endpoint(request: IngestRequest, background_tasks: BackgroundTasks):
    """Trigger ETL pipeline to re-ingest the PDF."""
    import os
    if request.pdf_path:
        os.environ["PDF_PATH"] = request.pdf_path

    def _run():
        from etl.ingest import run_ingestion
        run_ingestion()

    background_tasks.add_task(_run)
    return {"status": "ingestion_started", "pdf_path": os.getenv("PDF_PATH")}


@app.get("/traces", tags=["Observability"])
def list_traces():
    files = sorted(LOG_DIR.glob("*.json"), reverse=True)
    return {"traces": [f.name for f in files], "count": len(files)}


@app.get("/traces/{filename}", tags=["Observability"])
def get_trace(filename: str):
    tf = LOG_DIR / filename
    if not tf.exists():
        raise HTTPException(status_code=404, detail="Trace not found")
    with open(tf) as f:
        return json.load(f)


@app.get("/graph/diagram", tags=["System"])
def graph_diagram():
    """Return a Mermaid diagram of the LangGraph agent flow."""
    diagram = """
graph TD
    START --> classify_query
    classify_query --> retrieve
    retrieve --> grade_documents
    grade_documents -->|CRAG fail + attempts<2| rewrite_query
    rewrite_query --> retrieve
    grade_documents -->|verification| retrieval_agent
    grade_documents -->|synthesis| synthesis_agent
    grade_documents -->|forecasting| math_agent
    math_agent -->|has expression| calculate
    math_agent -->|no expression| reflect
    calculate --> reflect
    retrieval_agent --> reflect
    synthesis_agent --> reflect
    reflect -->|grounded| format_final_answer
    reflect -->|not grounded regen=0| retrieval_agent
    reflect -->|not grounded regen=0| synthesis_agent
    reflect -->|not grounded regen=0| math_agent
    reflect -->|regen>=1| format_final_answer
    format_final_answer --> END
"""
    return {"mermaid": diagram.strip()}
