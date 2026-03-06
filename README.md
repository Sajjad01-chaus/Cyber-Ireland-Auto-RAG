# Cyber Ireland 2022 — Multi-Agent Autonomous RAG System

> **Stack:** LangGraph · Groq Llama-3.3-70b · BAAI/bge-large · ChromaDB · BM25 · Cross-Encoder · C-RAG · Self-RAG

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                              │
│          POST /query  |  GET /health  |  GET /traces               │
└─────────────────────────────┬──────────────────────────────────────┘
                               │
┌─────────────────────────────▼──────────────────────────────────────┐
│                  LangGraph Multi-Agent Graph                        │
│                                                                     │
│  classify_query ──► retrieve ──► grade_documents                   │
│  (Supervisor)      (Hybrid RAG)     (C-RAG)                        │
│                         ▲               │                           │
│                         │         ┌─────┴──────┐                   │
│                    rewrite_query  │ relevant?  │                   │
│                    (C-RAG retry)  └─────┬──────┘                   │
│                                         │                           │
│                          ┌──────────────┼──────────────┐           │
│                          ▼              ▼              ▼           │
│                   retrieval_agent synthesis_agent  math_agent       │
│                   (verification)  (comparison)  (forecasting)      │
│                          │              │        ▼    │             │
│                          │              │   calculate │             │
│                          └──────────────┴────────┬───┘             │
│                                                   ▼                 │
│                                               reflect               │
│                                             (Self-RAG)             │
│                                                   │                 │
│                                        ┌──────────┴──────────┐     │
│                                        │ grounded?           │     │
│                                        └──────────┬──────────┘     │
│                                                   ▼                 │
│                                         format_final_answer         │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                    Hybrid Retrieval Engine                          │
│                                                                     │
│   Query ──► BM25 (exact keyword)  ──┐                              │
│         ──► Dense (bge-large)     ──┼──► RRF Fusion ──► Reranker   │
│                                     │   (Reciprocal  (ms-marco     │
│                                     │    Rank Fusion) cross-enc.)  │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                     ETL Pipeline                                    │
│                                                                     │
│   PDF ──► pymupdf (text + headings) ──┐                             │
│       ──► pdfplumber (tables → MD)  ──┼──► Semantic Chunker        │
│                                       │    (cosine similarity       │
│                                       │     split boundaries)       │
│                                       ▼                             │
│                            Parent-Child Chunks                      │
│                            ├── ChromaDB (dense index)               │
│                            └── BM25 (sparse index)                 │
└────────────────────────────────────────────────────────────────────┘
```

---

## Setup & Execution

### 1. Prerequisites

- Python 3.10+
- A free Groq API key from [console.groq.com](https://console.groq.com)

### 2. Install

```bash
git clone <repo-url>
cd cyber_rag_v2
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — add your GROQ_API_KEY
```

### 4. Download the PDF

```bash
# Download Cyber Ireland 2022 report and place in project root
# URL: https://cyberireland.ie/wp-content/uploads/2022/05/State-of-the-Cyber-Security-Sector...
# Set PDF_PATH=./cyber_ireland_2022.pdf in .env
```

### 5. Run ETL Pipeline (once)

```bash
python -m etl.ingest
```

Output:
- `./chroma_db/`          — ChromaDB dense vector index
- `./bm25_index.pkl`      — BM25 sparse index
- `./logs/etl_parents.json`  — debug: all parent chunks
- `./logs/etl_children.json` — debug: all indexed child chunks

### 6. Run the 3 Test Queries

```bash
python run_tests.py
```

Output in `./logs/`:
- `*_test_1_verification.json`
- `*_test_2_synthesis.json`
- `*_test_3_forecasting.json`
- `*_all_test_results.json`

### 7. Start the API

```bash
uvicorn main:app --reload --port 8000
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 8. Example API Calls

```bash
# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the total number of cybersecurity jobs?"}'

# Health check
curl http://localhost:8000/health

# View graph diagram (Mermaid)
curl http://localhost:8000/graph/diagram

# List saved traces
curl http://localhost:8000/traces
```

---

## Architecture Justification

### ETL Pipeline

| Decision | Choice | Reason |
|---|---|---|
| PDF text parser | **pymupdf (fitz)** | Reads font metadata (size, bold) → detects section headings automatically. Preserves reading order better than text-only parsers. |
| PDF table parser | **pdfplumber** | Best-in-class bounding-box table extraction. Handles multi-column tables that confuse other parsers. |
| Table format | **Markdown** | LLMs reason significantly better over Markdown tables than CSV, JSON, or linearized text. Preserves relational structure visually. |
| Chunking | **Semantic chunking** (cosine similarity) | Splits at meaning boundaries, not character counts. Consecutive sentences with low semantic similarity → new chunk. Prevents facts from being cut mid-context. |
| Chunk architecture | **Parent-Child** | Small child chunks → precise embedding retrieval. Each child references its parent (larger window). Avoids the precision-vs-context tradeoff. |
| Indexing | **Dual: ChromaDB + BM25** | ChromaDB for semantic/paraphrased queries. BM25 for exact keyword matches (critical for numbers like "7,000 jobs", names like "South-West"). |

### Agent Framework

| Decision | Choice | Reason |
|---|---|---|
| Orchestration | **LangGraph** | Stateful graph with conditional edges. Supports retry loops (C-RAG) and self-correction (Self-RAG) without spaghetti logic. Full state visibility at every node. |
| Main LLM | **Groq Llama-3.3-70b-versatile** | Best open reasoning model on Groq. Temperature=0 for deterministic factual outputs. Groq LPU delivers 10-20x faster inference than cloud APIs. |
| Grader/Reflector | **Groq Llama-3.1-8b-instant** | Fast, high rate-limit model for lightweight grading tasks (C-RAG, Self-RAG). Saves 70b tokens for reasoning-heavy steps. |
| Embeddings | **BAAI/bge-large-en-v1.5** | Top-ranked MTEB embedding model. Free, local, no API dependency. 1024 dimensions outperform ada-002 on retrieval benchmarks. |
| Query routing | **Supervisor classifier** | Routes to specialist agents (verification/synthesis/math) before retrieval. Avoids generic agent trying to do everything with wrong tool strategy. |

### Retrieval Pipeline

| Component | Role |
|---|---|
| **BM25** | Exact keyword/number matching. Critical for Test 1 (exact job count) and Test 2 (region names). |
| **Dense search** | Semantic similarity. Handles paraphrased queries and conceptual questions. |
| **RRF Fusion** | Mathematically principled merge of two ranked lists. No score normalization needed. Consistent benchmark improvements over score-based fusion. |
| **Cross-Encoder (ms-marco-MiniLM)** | Joint query-document encoding → more accurate relevance than bi-encoder. Applied after RRF to re-rank top-8 → top-5 final. |

### Self-Correction Loops

| Mechanism | Trigger | Action |
|---|---|---|
| **C-RAG** (Corrective RAG) | Retrieved chunks grade as irrelevant | Query Rewriter node reformulates query with domain-specific terms, retries retrieval (max 2 attempts) |
| **Self-RAG** | Generated answer has grounding score < 0.6 | Routes back to specialist agent for regeneration (max 1 retry), then accepts with caveat |

### Math Reliability

LLMs cannot reliably compute `(17000/7000)^(1/8) - 1`. This system:
1. Has the Math Agent **extract numbers and write the expression** (LLM's strength)
2. Routes the expression to a **safe Python `eval`** via the `calculate` node (computer's strength)
3. Returns the verified numeric result for inclusion in the final answer

---

## Limitations & Production Scaling

### Current Limitations

| Area | Limitation |
|---|---|
| **PDF tables** | Complex multi-spanning cells may misalign. pdfplumber's line-based detection can miss borderless tables. |
| **Groq rate limits** | Free tier: ~30 req/min for 70b, ~100/min for 8b. CRAG grading runs N LLM calls per chunk — can hit limits on large retrievals. |
| **Single document** | ChromaDB collection is one-document scoped. Cross-document queries require collection routing layer. |
| **No streaming** | Responses are synchronous. Long multi-step queries block until completion. |
| **Local embeddings** | bge-large-en loads ~1.3GB into RAM. First query has cold-start delay. |

### Production Scaling Roadmap

| Layer | Upgrade |
|---|---|
| **PDF Parsing** | `unstructured.io` partition API with `hi_res` strategy + OCR for scanned docs |
| **Embeddings** | Move to hosted inference (Hugging Face Inference Endpoints) to avoid RAM overhead |
| **Vector DB** | Pinecone/Weaviate with metadata filtering and namespace-per-document isolation |
| **LLM** | OpenAI GPT-4o as fallback chain; structured outputs with `response_format=json_object` |
| **Groq Rate Limits** | Batch CRAG grading into single prompt; add Redis-backed rate limiter |
| **Streaming** | FastAPI `StreamingResponse` + LangGraph streaming callbacks |
| **Caching** | Semantic query cache (GPTCache/Redis) — similar queries skip full pipeline |
| **Observability** | LangSmith tracing or Arize Phoenix for production drift detection |
| **Deployment** | Docker Compose (API + Qdrant) → Kubernetes on GKE/EKS |

---

## Project Structure

```
cyber_rag_v2/
├── etl/
│   ├── __init__.py
│   └── ingest.py           # PDF parsing, semantic chunking, ChromaDB + BM25 loading
├── agent/
│   ├── __init__.py
│   ├── retriever.py        # HybridRetriever: BM25 + Dense + RRF + Rerank
│   ├── prompts.py          # All LLM prompts (centralised)
│   └── graph.py            # LangGraph multi-agent graph + all node functions
├── logs/
│   ├── etl_parents.json    # ETL debug output
│   ├── etl_children.json   # ETL debug output
│   └── traces/             # Per-query agent traces (auto-generated)
├── main.py                 # FastAPI app
├── run_tests.py            # Runs all 3 evaluation scenarios
├── requirements.txt
├── .env.example
└── README.md
```
