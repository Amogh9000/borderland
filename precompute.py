"""
precompute.py — Phase 1: Offline FAISS Index Builder
Reads candidates.jsonl, generates 384-dim embeddings via all-MiniLM-L6-v2,
normalizes for cosine similarity, builds a flat IP FAISS index, and serializes
both the index and the id-map to the artifacts/ directory.
"""

import os
import sys
import json
import pickle
import logging
import argparse
import time
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 512          # number of records to embed per batch
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


# ---------------------------------------------------------------------------
# Text construction helpers
# ---------------------------------------------------------------------------

def _safe_str(value) -> str:
    """Coerce any value to a stripped string; return '' on None/NaN."""
    if value is None:
        return ""
    s = str(value).strip()
    return s if s.lower() not in ("none", "nan", "null") else ""


def build_text_block(record: dict) -> str:
    """
    Concatenate all semantically relevant fields from a candidate record
    into a single text block for embedding.

    Fields used:
      profile.headline
      profile.summary
      career_history[*].title
      career_history[*].description
    """
    parts: list[str] = []

    profile = record.get("profile", {}) or {}

    headline = _safe_str(profile.get("headline", ""))
    if headline:
        parts.append(headline)

    summary = _safe_str(profile.get("summary", ""))
    if summary:
        parts.append(summary)

    career = record.get("career_history", []) or []
    for job in career:
        if not isinstance(job, dict):
            continue
        title = _safe_str(job.get("title", ""))
        company = _safe_str(job.get("company", ""))
        desc = _safe_str(job.get("description", ""))
        # Build "title at company: description" for richer embedding
        job_parts = []
        if title:
            job_parts.append(title)
        if company:
            job_parts.append(f"at {company}")
        if desc:
            job_parts.append(f": {desc}" if job_parts else desc)
        if job_parts:
            parts.append(" ".join(job_parts))

    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Streaming JSONL reader (memory-efficient for 100 K records)
# ---------------------------------------------------------------------------

def stream_jsonl(filepath: Path):
    """
    Yields (line_number, record_dict) for every valid JSON line.
    Silently skips malformed lines after logging a warning.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield line_no, json.loads(raw)
            except json.JSONDecodeError as exc:
                log.warning("Skipping line %d — JSON parse error: %s", line_no, exc)


# ---------------------------------------------------------------------------
# Embedding + normalisation
# ---------------------------------------------------------------------------

def embed_and_normalise(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Encode a list of texts and L2-normalise the resulting vectors so that
    inner-product search is equivalent to cosine similarity.
    Returns float32 array of shape (len(texts), EMBED_DIM).
    """
    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # SentenceTransformers does L2-norm in-place
    )
    # Defensive re-normalisation in case the version doesn't honour the flag
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vecs = (vecs / norms).astype(np.float32)
    return vecs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_index(jsonl_path: Path, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    index_path = artifacts_dir / "faiss_index.bin"
    ids_path = artifacts_dir / "candidate_ids.pkl"

    log.info("Loading sentence-transformer model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    log.info("Scanning JSONL file: %s", jsonl_path)
    t0 = time.perf_counter()

    # We'll accumulate candidate_ids and text_blocks then embed in large batches
    candidate_ids: list[str] = []
    text_blocks: list[str] = []
    batch_vectors: list[np.ndarray] = []

    pending_ids: list[str] = []
    pending_texts: list[str] = []

    def flush_batch():
        nonlocal pending_ids, pending_texts
        if not pending_texts:
            return
        vecs = embed_and_normalise(model, pending_texts)
        batch_vectors.append(vecs)
        candidate_ids.extend(pending_ids)
        pending_ids = []
        pending_texts = []

    total_records = 0
    skipped = 0

    for line_no, record in stream_jsonl(jsonl_path):
        cid = record.get("candidate_id") or record.get("id")
        if not cid:
            log.warning("Line %d has no candidate_id — skipping.", line_no)
            skipped += 1
            continue

        text = build_text_block(record)
        if not text:
            log.warning("Line %d (id=%s) produced empty text block — skipping.", line_no, cid)
            skipped += 1
            continue

        pending_ids.append(str(cid))
        pending_texts.append(text)
        total_records += 1

        if len(pending_texts) >= BATCH_SIZE:
            flush_batch()
            log.info("  Embedded %d records so far ...", total_records)

    # flush remainder
    flush_batch()

    elapsed = time.perf_counter() - t0
    log.info(
        "Finished reading JSONL: %d records embedded, %d skipped. (%.1fs)",
        total_records, skipped, elapsed,
    )

    if total_records == 0:
        log.error("No valid records found — aborting index build.")
        sys.exit(1)

    # Stack all vectors into a single contiguous float32 matrix
    log.info("Stacking embedding matrix ...")
    all_vectors = np.vstack(batch_vectors).astype(np.float32)
    assert all_vectors.shape == (total_records, EMBED_DIM), (
        f"Shape mismatch: expected ({total_records}, {EMBED_DIM}), got {all_vectors.shape}"
    )

    # Build FAISS index — IndexFlatIP for exact cosine search (vectors are L2-normalised)
    log.info("Building FAISS IndexFlatIP (dim=%d, vectors=%d) ...", EMBED_DIM, total_records)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(all_vectors)
    assert index.ntotal == total_records, (
        f"FAISS index size {index.ntotal} != expected {total_records}"
    )

    # Serialise index
    log.info("Writing FAISS index → %s", index_path)
    faiss.write_index(index, str(index_path))

    # Serialise candidate_ids list (positional mapping: row i ↔ candidate_ids[i])
    log.info("Writing candidate ID map → %s", ids_path)
    with open(ids_path, "wb") as fh:
        pickle.dump(candidate_ids, fh, protocol=pickle.HIGHEST_PROTOCOL)

    total_elapsed = time.perf_counter() - t0
    log.info(
        "Index build complete. %d vectors indexed in %.1fs. Artifacts saved to %s.",
        total_records, total_elapsed, artifacts_dir,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 — Build FAISS index from candidates.jsonl"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "candidates.jsonl",
        help="Path to the candidates JSONL file (default: ./candidates.jsonl)",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directory to write faiss_index.bin and candidate_ids.pkl (default: ./artifacts/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.input.exists():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    build_index(args.input, args.artifacts)
