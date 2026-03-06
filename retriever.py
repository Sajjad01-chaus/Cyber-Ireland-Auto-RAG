"""
Hybrid Retriever
=================
Combines BM25 sparse search + ChromaDB dense search via
Reciprocal Rank Fusion (RRF), then re-ranks with a cross-encoder.

Pipeline:
  Query
    ├─► BM25 (exact keyword match, great for numbers/names)      top-K
    ├─► ChromaDB (semantic similarity, great for paraphrased Q)  top-K
    └─► RRF merge → cross-encoder re-rank → final top-N chunks
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────
CHROMA_PATH     = os.getenv("CHROMA_DB_PATH", "./chroma_db")
BM25_PATH       = os.getenv("BM25_INDEX_PATH", "./bm25_index.pkl")
EMBED_MODEL     = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
COLLECTION_NAME = "cyber_ireland_2022"

TOP_K_DENSE       = int(os.getenv("TOP_K_DENSE", 10))
TOP_K_BM25        = int(os.getenv("TOP_K_BM25", 10))
TOP_K_AFTER_FUSION = int(os.getenv("TOP_K_AFTER_FUSION", 8))
TOP_K_AFTER_RERANK = int(os.getenv("TOP_K_AFTER_RERANK", 5))

RRF_K = 60   # standard RRF constant

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class HybridRetriever:
    """
    Stateful retriever — load once, query many times.
    """

    def __init__(self):
        self._embed_model   : Optional[SentenceTransformer] = None
        self._cross_encoder : Optional[CrossEncoder]        = None
        self._collection    = None
        self._bm25_data     : Optional[Dict]                = None
        self._initialized   = False

    def initialize(self):
        if self._initialized:
            return

        logger.info("Initializing HybridRetriever …")

        # Embedding model (shared with ETL)
        logger.info(f"  Loading embedding model: {EMBED_MODEL}")
        self._embed_model = SentenceTransformer(EMBED_MODEL)

        # Cross-encoder for re-ranking
        logger.info(f"  Loading cross-encoder: {CROSS_ENCODER_MODEL}")
        self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

        # ChromaDB
        logger.info(f"  Connecting to ChromaDB at {CHROMA_PATH}")

        class _EF(embedding_functions.EmbeddingFunction):
            def __init__(self, model):
                self._model = model
            def __call__(self, input):
                return self._model.encode(input, normalize_embeddings=True).tolist()

        ef = _EF(self._embed_model)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        self._collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )
        logger.info(f"  ChromaDB: {self._collection.count()} chunks")

        # BM25
        if not Path(BM25_PATH).exists():
            raise FileNotFoundError(
                f"BM25 index not found at {BM25_PATH}. Run ETL pipeline first."
            )
        logger.info(f"  Loading BM25 index from {BM25_PATH}")
        with open(BM25_PATH, "rb") as f:
            self._bm25_data = pickle.load(f)

        self._initialized = True
        logger.info("HybridRetriever ready ✅")

    # ── Dense Search ─────────────────────────────────────────────

    def _dense_search(self, query: str, top_k: int) -> List[Dict]:
        """ChromaDB semantic similarity search."""
        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text"    : doc,
                "metadata": meta,
                "score"   : 1 - dist,   # cosine similarity
                "source"  : "dense",
            })
        return hits

    # ── Sparse BM25 Search ───────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """BM25 keyword / exact-match search."""
        bm25         = self._bm25_data["bm25"]
        corpus_texts = self._bm25_data["corpus_texts"]
        corpus_ids   = self._bm25_data["corpus_ids"]
        corpus_meta  = self._bm25_data["corpus_meta"]

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        hits = []
        for idx in top_indices:
            if scores[idx] > 0:
                hits.append({
                    "text"    : corpus_texts[idx],
                    "metadata": corpus_meta[idx],
                    "score"   : float(scores[idx]),
                    "source"  : "bm25",
                })
        return hits

    # ── Reciprocal Rank Fusion ───────────────────────────────────

    def _rrf_fusion(
        self,
        dense_hits : List[Dict],
        bm25_hits  : List[Dict],
        top_k      : int,
    ) -> List[Dict]:
        """
        Merge dense and BM25 results via RRF.
        RRF score = 1/(rank + K) summed over both result lists.
        """
        scores: Dict[str, float] = {}
        docs  : Dict[str, Dict]  = {}

        for rank, hit in enumerate(dense_hits):
            key = hit["text"][:100]   # deduplicate by text prefix
            scores[key] = scores.get(key, 0) + 1 / (rank + 1 + RRF_K)
            docs[key]   = hit

        for rank, hit in enumerate(bm25_hits):
            key = hit["text"][:100]
            scores[key] = scores.get(key, 0) + 1 / (rank + 1 + RRF_K)
            if key not in docs:
                docs[key] = hit

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]

        fused = []
        for key in sorted_keys:
            hit = docs[key].copy()
            hit["rrf_score"] = scores[key]
            fused.append(hit)

        return fused

    # ── Cross-Encoder Re-ranking ─────────────────────────────────

    def _rerank(self, query: str, hits: List[Dict], top_k: int) -> List[Dict]:
        """
        Re-rank fused candidates using a cross-encoder.
        Cross-encoder jointly encodes (query, passage) for accurate relevance.
        """
        if not hits:
            return hits

        pairs  = [(query, h["text"]) for h in hits]
        scores = self._cross_encoder.predict(pairs)

        for hit, score in zip(hits, scores):
            hit["rerank_score"] = float(score)

        reranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_k]
        return reranked

    # ── Public API ───────────────────────────────────────────────

    def retrieve(
        self,
        query          : str,
        top_k_dense    : int = TOP_K_DENSE,
        top_k_bm25     : int = TOP_K_BM25,
        top_k_fusion   : int = TOP_K_AFTER_FUSION,
        top_k_final    : int = TOP_K_AFTER_RERANK,
    ) -> List[Dict]:
        """
        Full hybrid retrieval pipeline:
        BM25 + Dense → RRF → Cross-Encoder Rerank → top_k_final chunks
        """
        if not self._initialized:
            self.initialize()

        logger.info(f"HybridRetriever.retrieve: '{query[:60]}…'")

        dense_hits = self._dense_search(query, top_k_dense)
        bm25_hits  = self._bm25_search(query, top_k_bm25)

        logger.info(
            f"  Dense: {len(dense_hits)} hits | BM25: {len(bm25_hits)} hits"
        )

        fused   = self._rrf_fusion(dense_hits, bm25_hits, top_k_fusion)
        reranked = self._rerank(query, fused, top_k_final)

        logger.info(f"  After RRF+rerank: {len(reranked)} final chunks")
        for i, r in enumerate(reranked):
            logger.info(
                f"    [{i+1}] page={r['metadata'].get('page_number','?')} "
                f"type={r['metadata'].get('element_type','?')} "
                f"rerank={r.get('rerank_score', 0):.3f}"
            )

        return reranked

    def get_page_chunks(self, page_number: int) -> List[Dict]:
        """Fetch ALL chunks for a specific page (for citation verification)."""
        if not self._initialized:
            self.initialize()

        results = self._collection.get(
            where={"page_number": page_number},
            include=["documents", "metadatas"],
        )
        hits = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            hits.append({"text": doc, "metadata": meta})
        return hits


# ── Singleton ─────────────────────────────────────────────────────
_retriever: Optional[HybridRetriever] = None

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
        _retriever.initialize()
    return _retriever
