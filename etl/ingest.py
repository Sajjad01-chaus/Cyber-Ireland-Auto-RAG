"""
ETL Pipeline — PDF Ingestion 

Architecture:
  PASS 1 (pymupdf/fitz):
    - Extract text blocks with font metadata → detect headings
    - Detect two-column pages and split accordingly
    - Track table title candidates (bold ALL-CAPS text near table positions)

  PASS 2 (pdfplumber — 3-strategy cascade):
    Strategy A: explicit lines (borders drawn)
    Strategy B: lines + text (mix of borders and text alignment)
    Strategy C: text-only (color-background tables, no borders)
    → Whichever finds more tables wins per page

  Chunking:
    - Tables: atomic (never split) + table title prepended
    - Text: semantic chunking (cosine similarity split) + 1-sentence overlap
    - Parent-child: small child for retrieval, large parent for context

  Storage:
    - ChromaDB (dense, cosine) + BM25 (sparse keyword)
    - Both indexed on same child chunks for hybrid fusion at query time
"""

import os
import re
import json
import pickle
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import fitz           # pymupdf
import pdfplumber
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────
PDF_PATH        = os.getenv("PDF_PATH", "State-of-the-Cyber-Security-Sector-in-Ireland-2022-Report.pdf")
CHROMA_PATH     = os.getenv("CHROMA_DB_PATH", "./chroma_db")
BM25_PATH       = os.getenv("BM25_INDEX_PATH", "./bm25_index.pkl")
EMBED_MODEL     = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
COLLECTION_NAME = "cyber_ireland_2022"

# Semantic chunking thresholds
SEMANTIC_SPLIT_THRESHOLD = 0.45   # cosine sim below this → new chunk
MAX_CHUNK_CHARS          = 800
MIN_CHUNK_CHARS          = 80
CHUNK_OVERLAP_SENTENCES  = 1      

# Two-column detection: if text blocks are mostly in left half AND right half
TWO_COL_THRESHOLD = 0.3   # if >30% blocks in each half → two-column page


# ═══════════════════════════════════════════════════════════════════
# FIX 3: Two-Column Page Detector
# ═══════════════════════════════════════════════════════════════════

def is_two_column_page(page) -> bool:
    """
    Detect if a page uses a two-column layout.
    """
    page_width = page.rect.width
    mid        = page_width / 2

    words = page.get_text("words")
    if not words:
        return False

    left_count  = sum(1 for w in words if w[0] < mid - 20)
    right_count = sum(1 for w in words if w[0] > mid + 20)
    total       = len(words)

    if total == 0:
        return False

    left_ratio  = left_count / total
    right_ratio = right_count / total

    return left_ratio > TWO_COL_THRESHOLD and right_ratio > TWO_COL_THRESHOLD


def extract_column(page, which: str) -> str:
    """Extract text from left or right half of a two-column page."""
    w = page.rect.width
    h = page.rect.height
    mid = w / 2

    if which == "left":
        clip = fitz.Rect(0, 0, mid, h)
    else:
        clip = fitz.Rect(mid, 0, w, h)

    return page.get_text("text", clip=clip).strip()


# ═══════════════════════════════════════════════════════════════════
# PASS 1 — pymupdf: text + heading detection + table title tracking
# ═══════════════════════════════════════════════════════════════════

def extract_text_with_structure(pdf_path: str) -> Tuple[List[Dict], Dict[int, List[str]]]:
    """
    Use pymupdf to extract text blocks with font metadata.

    Returns:
      elements       — list of text/heading elements per page
      table_titles   — dict of {page_num: [title strings]}
                       FIX 2: captures "TABLE X.X TITLE" strings for table heading injection
    """
    doc = fitz.open(pdf_path)
    elements    : List[Dict]         = []
    table_titles: Dict[int, List[str]] = {}
    current_heading = "Introduction"

    # Regex to detect table title lines like "TABLE 4.3 TAXONOMY SUMMARY"
    TABLE_TITLE_RE = re.compile(
        r'^(TABLE|FIGURE|CHART)\s+\d+[\.\d]*\s+.{3,}',
        re.IGNORECASE
    )

    for page_num in range(len(doc)):
        page          = doc[page_num]
        page_num_1    = page_num + 1
        table_titles[page_num_1] = []

        # FIX 3: handle two-column pages
        if is_two_column_page(page):
            logger.debug(f"  Page {page_num_1}: two-column layout detected")
            for col in ["left", "right"]:
                col_text = extract_column(page, col)
                if col_text and len(col_text) > 20:
                    elements.append({
                        "page_number"    : page_num_1,
                        "text"           : col_text,
                        "is_heading"     : False,
                        "section_heading": current_heading,
                        "font_size"      : 10.0,
                        "element_type"   : "text",
                        "column"         : col,
                    })
            # Still scan for headings and table titles in full page
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        else:
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        for block in blocks:
            if block.get("type") != 0:
                continue

            block_text    = ""
            is_heading    = False
            max_font_size = 0

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text       = span.get("text", "").strip()
                    font_size  = span.get("size", 0)
                    font_flags = span.get("flags", 0)
                    is_bold    = bool(font_flags & 2**4)

                    if text:
                        block_text   += text + " "
                        max_font_size = max(max_font_size, font_size)

                        if is_bold and font_size >= 11 and len(text) < 120:
                            is_heading = True

            block_text = block_text.strip()
            if not block_text or len(block_text) < 10:
                continue

            # FIX 2: detect table titles
            if TABLE_TITLE_RE.match(block_text):
                table_titles[page_num_1].append(block_text)
                logger.debug(f"  Table title found p{page_num_1}: {block_text[:60]}")
                is_heading = True   

            if is_heading:
                current_heading = block_text

            # Skip duplicate text blocks on two-column pages
            if is_two_column_page(page):
                continue

            elements.append({
                "page_number"    : page_num_1,
                "text"           : block_text,
                "is_heading"     : is_heading,
                "section_heading": current_heading,
                "font_size"      : round(max_font_size, 1),
                "element_type"   : "heading" if is_heading else "text",
            })

    doc.close()
    logger.info(
        f"[pymupdf] {len(elements)} text blocks | "
        f"{sum(len(v) for v in table_titles.values())} table titles found"
    )
    return elements, table_titles




def extract_figure_data(pdf_path: str) -> List[Dict]:
   
    FIGURE_RE   = re.compile(r'^FIGURE\s+\d+[\.\d]*', re.IGNORECASE)
    # Matches short numeric/percentage strings likely to be chart labels
    LABEL_RE    = re.compile(r'^[\d,\.]+(%|k|m|bn)?$|^\d+%$', re.IGNORECASE)

    doc             = fitz.open(pdf_path)
    figure_elements = []

    for page_num in range(len(doc)):
        page       = doc[page_num]
        page_num_1 = page_num + 1
        blocks     = page.get_text("dict")["blocks"]

        # Find all FIGURE title blocks on this page
        figure_titles = []
        data_labels   = []   # short numeric/text label spans
        axis_labels   = []   # category labels (US, IRELAND, LARGE, MEDIUM...)

        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    # Collect figure titles
                    if FIGURE_RE.match(text) and len(text) > 10:
                        figure_titles.append(text)

                    # Collect short numeric labels (chart bar values)
                    elif LABEL_RE.match(text) and len(text) <= 12:
                        data_labels.append(text)

                    # Collect short all-caps axis labels (country names, categories)
                    elif text.isupper() and 2 <= len(text) <= 20:
                        axis_labels.append(text)

        if not figure_titles:
            continue

        # Build a synthetic figure summary chunk per figure title
        for fig_title in figure_titles:
            # Pair data labels with axis labels (zip in order — best effort)
            pairs = []
            for i, label in enumerate(axis_labels):
                val = data_labels[i] if i < len(data_labels) else "?"
                pairs.append(f"{label}={val}")

            if pairs:
                data_str = " | ".join(pairs)
                chunk_text = (
                    f"{fig_title}\n"
                    f"[FIGURE DATA — extracted from chart labels]\n"
                    f"Data points: {data_str}"
                )
            else:
                # with no label pairs, store figure title + any numbers found
                nums = " | ".join(data_labels[:10]) if data_labels else "see document"
                chunk_text = (
                    f"{fig_title}\n"
                    f"[FIGURE DATA — chart labels on this page]\n"
                    f"Values: {nums}"
                )

            if len(chunk_text.strip()) < 30:
                continue

            figure_elements.append({
                "page_number"    : page_num_1,
                "text"           : chunk_text,
                "is_heading"     : False,
                "section_heading": fig_title,
                "font_size"      : 0,
                "element_type"   : "figure",
            })

    doc.close()
    logger.info(f"[figure harvester] {len(figure_elements)} figure summary chunks created")
    return figure_elements


# ═══════════════════════════════════════════════════════════════════
# PASS 2 — pdfplumber: 3-strategy cascade table extraction
# ═══════════════════════════════════════════════════════════════════

def table_to_markdown(table: List[List[Any]], title: str = "") -> str:
    """
    Convert pdfplumber table → Markdown with optional title prepended.
    """
    if not table:
        return ""

    rows = []
    if title:
        rows.append(f"**{title}**\n")

    for i, row in enumerate(table):
        cells = [
            str(c).strip().replace("\n", " ") if c is not None else ""
            for c in row
        ]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            rows.append("|" + "|".join(["---"] * len(cells)) + "|")

    return "\n".join(rows)


def extract_tables_from_page(page, page_num: int) -> List[List[List[Any]]]:
    """
    3-strategy cascade for table detection.

    Strategy A: lines_strict — for tables with explicit drawn borders
    Strategy B: lines (relaxed) — partial borders
    Strategy C: text — for color-background tables with no visible borders
                (e.g. TABLE 4.3, 4.4, 4.5 in the Cyber Ireland report)

    Returns whichever strategy finds the most tables.
    """
    results = []

    # Strategy A — strict lines (bordered tables)
    try:
        tbls = page.extract_tables({
            "vertical_strategy"  : "lines_strict",
            "horizontal_strategy": "lines_strict",
            "snap_tolerance"     : 3,
        })
        if tbls:
            results.append(("lines_strict", tbls))
    except Exception:
        pass

    # Strategy B — relaxed lines
    try:
        tbls = page.extract_tables({
            "vertical_strategy"  : "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance"     : 5,
            "join_tolerance"     : 3,
        })
        if tbls:
            results.append(("lines", tbls))
    except Exception:
        pass

    # Strategy C — text alignment (catches colored-background tables)
    try:
        tbls = page.extract_tables({
            "vertical_strategy"  : "text",
            "horizontal_strategy": "text",
            "snap_tolerance"     : 5,
            "min_words_vertical" : 2,
            "min_words_horizontal": 1,
        })
        if tbls:
            results.append(("text", tbls))
    except Exception:
        pass

    if not results:
        return []

    # Pick strategy that found the most non-trivial tables
    best_strategy, best_tables = max(
        results,
        key=lambda x: sum(1 for t in x[1] if t and len(t) >= 2)
    )
    valid = [t for t in best_tables if t and len(t) >= 2]

    if valid:
        logger.debug(f"  Page {page_num}: {len(valid)} tables via '{best_strategy}'")

    return valid


def extract_tables(pdf_path: str, table_titles: Dict[int, List[str]]) -> List[Dict]:
    """
    Extract all tables from the PDF using the 3-strategy cascade.
    """
    table_elements = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # FIX 3: For two-column pages, try each column separately
            page_width = page.width
            page_height = page.height

            # Detect if two-column (using word positions)
            words = page.extract_words()
            mid   = page_width / 2
            left_words  = [w for w in words if float(w["x0"]) < mid - 20]
            right_words = [w for w in words if float(w["x0"]) > mid + 20]
            total = len(words)
            is_two_col = (
                total > 0
                and len(left_words)/total > TWO_COL_THRESHOLD
                and len(right_words)/total > TWO_COL_THRESHOLD
            )

            if is_two_col:
                # Extract tables from left and right columns separately
                left_crop  = page.crop((0, 0, mid, page_height))
                right_crop = page.crop((mid, 0, page_width, page_height))
                column_pages = [("left", left_crop), ("right", right_crop)]
            else:
                column_pages = [("full", page)]

            page_titles = table_titles.get(page_num, [])
            title_idx   = 0

            for col_name, col_page in column_pages:
                tables = extract_tables_from_page(col_page, page_num)

                for t_idx, table in enumerate(tables):
                    md_title = ""
                    if title_idx < len(page_titles):
                        md_title  = page_titles[title_idx]
                        title_idx += 1

                    md = table_to_markdown(table, title=md_title)
                    if not md.strip():
                        continue

                    section_heading = md_title if md_title else f"Table on Page {page_num}"

                    table_elements.append({
                        "page_number"    : page_num,
                        "text"           : f"[TABLE on Page {page_num}]\n{md}",
                        "is_heading"     : False,
                        "section_heading": section_heading,
                        "font_size"      : 0,
                        "element_type"   : "table",
                        "column"         : col_name,
                    })

    logger.info(f"[pdfplumber] Extracted {len(table_elements)} tables")
    return table_elements



class SemanticChunker:
    """
    Splits text at semantic boundaries (cosine similarity valleys).

    Tables and headings are always kept atomic (never split).
    """

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def _split_sentences(self, text: str) -> List[str]:
        raw = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in raw if len(s.strip()) > 20]

    def chunk(self, text: str, metadata: Dict) -> List[Dict]:
        # Tables + headings: always atomic
        if metadata.get("element_type") in ("table", "heading"):
            return [{"text": text, "metadata": metadata}]

        if len(text) <= MAX_CHUNK_CHARS:
            return [{"text": text, "metadata": metadata}]

        sentences = self._split_sentences(text)
        if len(sentences) <= 2:
            return self._char_split(text, metadata)

        # Embed sentences and compute consecutive cosine similarities
        embeddings = self.model.encode(sentences, normalize_embeddings=True)
        sims = [
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]

        # Split at semantic valleys
        groups: List[List[str]] = []
        current_group = [sentences[0]]

        for i, sim in enumerate(sims):
            if (sim < SEMANTIC_SPLIT_THRESHOLD
                    or len(" ".join(current_group)) > MAX_CHUNK_CHARS):
                if len(" ".join(current_group).strip()) >= MIN_CHUNK_CHARS:
                    groups.append(current_group)
                current_group = [sentences[i + 1]]
            else:
                current_group.append(sentences[i + 1])

        if current_group:
            tail = " ".join(current_group).strip()
            if len(tail) >= MIN_CHUNK_CHARS:
                groups.append(current_group)
            elif groups:
                groups[-1].extend(current_group)

        if not groups:
            return [{"text": text, "metadata": metadata}]

        # Add 1-sentence overlap between consecutive chunks
        overlapped: List[str] = []
        for idx, group in enumerate(groups):
            chunk_sentences = list(group)

            # Prepend last sentence(s) of previous chunk
            if idx > 0 and CHUNK_OVERLAP_SENTENCES > 0:
                prev = groups[idx - 1]
                prefix = prev[-CHUNK_OVERLAP_SENTENCES:]
                chunk_sentences = prefix + chunk_sentences

            # Append first sentence(s) of next chunk
            if idx < len(groups) - 1 and CHUNK_OVERLAP_SENTENCES > 0:
                nxt = groups[idx + 1]
                suffix = nxt[:CHUNK_OVERLAP_SENTENCES]
                chunk_sentences = chunk_sentences + suffix

            overlapped.append(" ".join(chunk_sentences).strip())

        result = []
        for idx, ct in enumerate(overlapped):
            if len(ct) >= MIN_CHUNK_CHARS:
                meta = {
                    **metadata,
                    "chunk_index"  : idx,
                    "total_chunks" : len(overlapped),
                    "has_overlap"  : True,
                }
                result.append({"text": ct, "metadata": meta})

        return result if result else [{"text": text, "metadata": metadata}]

    def _char_split(self, text: str, metadata: Dict) -> List[Dict]:
        """Fallback: character split with overlap."""
        overlap = 120   # increased from 80 → 120 chars
        chunks, start, idx = [], 0, 0
        while start < len(text):
            end        = start + MAX_CHUNK_CHARS
            chunk_text = text[start:end].strip()
            if len(chunk_text) >= MIN_CHUNK_CHARS:
                meta = {**metadata, "chunk_index": idx, "has_overlap": True}
                chunks.append({"text": chunk_text, "metadata": meta})
                idx += 1
            start += MAX_CHUNK_CHARS - overlap
        return chunks


# ═══════════════════════════════════════════════════════════════════
# Parent-Child Chunk Builder 
# ═══════════════════════════════════════════════════════════════════

def build_parent_child_chunks(
    elements: List[Dict],
    chunker : SemanticChunker,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Build parent chunks (large context) and child chunks (precise retrieval).
    Tables always become their own atomic parent + child pair.
    Text elements are grouped into parents per page, then semantically chunked.

    """
    parents : List[Dict] = []
    children: List[Dict] = []

    # Global counters 
    parent_counter = [0]
    child_counter  = [0]

    def make_pid() -> str:
        parent_counter[0] += 1
        return f"p{parent_counter[0]:06d}"

    def make_cid() -> str:
        child_counter[0] += 1
        return f"c{child_counter[0]:06d}"

    pages: Dict[int, List[Dict]] = {}
    for el in elements:
        pages.setdefault(el["page_number"], []).append(el)

    for page_num, page_elements in sorted(pages.items()):
        parent_buffer   = []
        parent_text     = ""
        current_heading = page_elements[0].get("section_heading", "")

        def flush_parent():
            nonlocal parent_text, parent_buffer, current_heading
            if not parent_text.strip():
                return
            pid = make_pid()
            parents.append({
                "id"      : pid,
                "text"    : parent_text.strip(),
                "page_num": page_num,
                "heading" : current_heading,
            })
            for el in parent_buffer:
                for child in chunker.chunk(el["text"], {
                    "page_number"    : page_num,
                    "section_heading": el.get("section_heading", current_heading),
                    "element_type"   : el.get("element_type", "text"),
                    "parent_id"      : pid,
                }):
                    cid = make_cid()
                    children.append({
                        "id"      : cid,
                        "text"    : child["text"],
                        "metadata": {**child["metadata"], "parent_id": pid, "chunk_id": cid},
                    })
            parent_text   = ""
            parent_buffer = []

        for el in page_elements:
            if el["element_type"] in ("table", "figure"):
                flush_parent()
                tid = make_pid()
                parents.append({
                    "id": tid, "text": el["text"],
                    "page_num": page_num, "heading": el.get("section_heading", ""),
                })
                cid = make_cid()
                children.append({
                    "id"  : cid,
                    "text": el["text"],
                    "metadata": {
                        "page_number"    : page_num,
                        "section_heading": el.get("section_heading", ""),
                        "element_type"   : el.get("element_type", "table"),
                        "parent_id"      : tid,
                        "chunk_id"       : cid,
                        "has_overlap"    : False,
                    }
                })
            else:
                parent_text += " " + el["text"]
                parent_buffer.append(el)
                current_heading = el.get("section_heading", current_heading)
                if len(parent_text) > 1200:
                    flush_parent()

        flush_parent()

    logger.info(f"Built {len(parents)} parent chunks, {len(children)} child chunks")
    return parents, children


# ═══════════════════════════════════════════════════════════════════
# ChromaDB + BM25 Loaders 
# ═══════════════════════════════════════════════════════════════════

class LocalEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model: SentenceTransformer):
        self._model = model
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._model.encode(input, normalize_embeddings=True).tolist()


def load_to_chroma(children: List[Dict], embed_model: SentenceTransformer):
    # ── Dedup safety: guarantee unique IDs before touching ChromaDB ──
    seen_ids : set = set()
    deduped  : List[Dict] = []
    for c in children:
        if c["id"] not in seen_ids:
            seen_ids.add(c["id"])
            deduped.append(c)
        else:
            logger.warning(f"  Duplicate child ID skipped: {c['id']} — text[:40]: {c['text'][:40]}")
    if len(deduped) < len(children):
        logger.warning(f"  Removed {len(children)-len(deduped)} duplicate chunks before ChromaDB upsert")
    children = deduped

    ef     = LocalEmbeddingFunction(embed_model)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Dropped existing ChromaDB collection")
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    BATCH = 64
    for i in range(0, len(children), BATCH):
        batch = children[i:i + BATCH]
        collection.upsert(
            ids       = [c["id"] for c in batch],
            documents = [c["text"] for c in batch],
            metadatas = [c["metadata"] for c in batch],
        )
        logger.info(f"  ChromaDB batch {i//BATCH+1}: {len(batch)} chunks")
    logger.info(f"✅ ChromaDB loaded: {collection.count()} chunks")
    return collection


def build_bm25_index(children: List[Dict]) -> Dict:
    corpus_texts = [c["text"] for c in children]
    corpus_ids   = [c["id"]   for c in children]
    corpus_meta  = [c["metadata"] for c in children]
    tokenized    = [t.lower().split() for t in corpus_texts]
    bm25         = BM25Okapi(tokenized)
    index_data   = {
        "bm25": bm25, "corpus_texts": corpus_texts,
        "corpus_ids": corpus_ids, "corpus_meta": corpus_meta,
    }
    Path(BM25_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(index_data, f)
    logger.info(f"✅ BM25 index: {len(corpus_texts)} docs → {BM25_PATH}")
    return index_data


def save_debug(parents: List[Dict], children: List[Dict]):
    Path("./logs").mkdir(exist_ok=True)
    with open("./logs/etl_parents.json", "w") as f:
        json.dump(parents, f, indent=2)
    with open("./logs/etl_children.json", "w") as f:
        json.dump(children, f, indent=2)
    logger.info("ETL debug → logs/etl_parents.json + etl_children.json")


# ═══════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════

def run_ingestion():
    logger.info("╔═══════════════════════════════════╗")
    logger.info("║  Cyber Ireland 2022 — ETL Pipeline║")
    logger.info("╠═══════════════════════════════════╣")
    logger.info("║  3-strategy table cascade         ║")
    logger.info("║  Table title injection            ║")
    logger.info("║  Two-column page detection        ║")
    logger.info("║  Semantic chunk overlap           ║")
    logger.info("╚═══════════════════════════════════╝")

    if not Path(PDF_PATH).exists():
        raise FileNotFoundError(
            f"PDF not found at '{PDF_PATH}'.\n"
            f"Run: curl -L 'https://cyberireland.ie/wp-content/uploads/2022/05/"
            f"State-of-the-Cyber-Security-Sector-in-Ireland-2022-Report.pdf' "
            f"-o cyber_ireland_2022.pdf"
        )

    logger.info(f"Loading embedding model: {EMBED_MODEL} …")
    embed_model = SentenceTransformer(EMBED_MODEL)

    logger.info("── PASS 1: pymupdf text + heading + table-title extraction …")
    text_elements, table_titles = extract_text_with_structure(PDF_PATH)

    logger.info("── PASS 2: pdfplumber 3-strategy table extraction …")
    table_elements = extract_tables(PDF_PATH, table_titles)

    logger.info("── PASS 3: figure/chart data harvester …")
    figure_elements = extract_figure_data(PDF_PATH)

    all_elements = sorted(
        text_elements + table_elements + figure_elements,
        key=lambda x: x["page_number"]
    )
    logger.info(f"Total elements: {len(all_elements)} "
                f"({len(text_elements)} text, {len(table_elements)} tables, "
                f"{len(figure_elements)} figures)")

    logger.info("── Semantic chunking + parent-child build …")
    chunker = SemanticChunker(embed_model)
    parents, children = build_parent_child_chunks(all_elements, chunker)

    save_debug(parents, children)

    logger.info("── Loading → ChromaDB …")
    load_to_chroma(children, embed_model)

    logger.info("── Building BM25 sparse index …")
    build_bm25_index(children)

    logger.info("╔══════════════════════════════════════════╗ ")
    logger.info("║   ETL Pipeline Complete ✅               ║")
    logger.info(f"║   {len(parents):>5} parent chunks                   ║")
    logger.info(f"║   {len(children):>5} child chunks (indexed)         ║")
    logger.info(f"║   {len(table_elements):>5} tables extracted         ║")
    logger.info(f"║   {len(figure_elements):>5} figure summaries created║")
    logger.info("╚══════════════════════════════════════════╝")
    return parents, children


if __name__ == "__main__":
    run_ingestion()