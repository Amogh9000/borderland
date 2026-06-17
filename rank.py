"""
rank.py — Phase 2 Main Entry Point
Orchestrates the full online ranking pipeline.
"""

import os
import sys
import json
import pickle
import logging
import argparse
import time
import csv
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from feature_extractor import score_skills, score_experience, score_location
from career_scorer import score_career, is_honeypot
from behavioral import behavioral_multiplier
from reasoning import generate_reasoning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Constants
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
FAISS_TOP_K = 500
OUTPUT_TOP_N = 100

JOB_DESCRIPTION = (
    "Senior AI Engineer - Founding Team. "
    "Modern ML systems, embeddings, retrieval, ranking, LLMs, fine-tuning. "
    "Sentence-transformers, FAISS, Qdrant, Pinecone. Pune, Noida."
)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_CSV = RESULTS_DIR / "top100_candidates.csv"

def sanitize_profile(profile: dict) -> dict:
    """
    Safely capture missing blocks, empty signals, or null values,
    populating safe fallbacks to eliminate any chance of a runtime KeyError.
    """
    if not isinstance(profile, dict):
        profile = {}

    sanitized = {}
    sanitized["candidate_id"] = str(profile.get("candidate_id") or profile.get("id") or "Unknown")

    p = profile.get("profile")
    if not isinstance(p, dict):
        p = {}

    sanitized_p = {}
    sanitized_p["headline"] = str(p.get("headline") or profile.get("headline") or "")
    sanitized_p["summary"] = str(p.get("summary") or profile.get("summary") or "")

    yoe = p.get("years_of_experience") or profile.get("years_of_experience")
    try:
        sanitized_p["years_of_experience"] = float(yoe) if yoe is not None else 0.0
    except (ValueError, TypeError):
        sanitized_p["years_of_experience"] = 0.0

    sanitized_p["location"] = str(p.get("location") or profile.get("location") or "")
    sanitized_p["current_title"] = str(p.get("current_title") or profile.get("current_title") or "")
    sanitized_p["current_company"] = str(p.get("current_company") or profile.get("current_company") or "")

    sanitized["profile"] = sanitized_p

    career = profile.get("career_history")
    if not isinstance(career, list):
        career = []
    sanitized_career = []
    for job in career:
        if isinstance(job, dict):
            sanitized_job = {
                "company": str(job.get("company") or job.get("employer") or ""),
                "title": str(job.get("title") or ""),
                "description": str(job.get("description") or ""),
                "duration_months": job.get("duration_months")
            }
            try:
                sanitized_job["duration_months"] = float(sanitized_job["duration_months"]) if sanitized_job["duration_months"] is not None else 0.0
            except (ValueError, TypeError):
                sanitized_job["duration_months"] = 0.0
            sanitized_career.append(sanitized_job)
    sanitized["career_history"] = sanitized_career

    skills = profile.get("skills")
    sanitized_skills = []
    if isinstance(skills, list):
        for s in skills:
            if isinstance(s, dict):
                sanitized_s = {
                    "name": str(s.get("name") or ""),
                    "proficiency": str(s.get("proficiency") or "intermediate"),
                    "duration_months": s.get("duration_months")
                }
                try:
                    sanitized_s["duration_months"] = float(sanitized_s["duration_months"]) if sanitized_s["duration_months"] is not None else 0.1
                except (ValueError, TypeError):
                    sanitized_s["duration_months"] = 0.1
                sanitized_skills.append(sanitized_s)
    elif isinstance(skills, dict):
        for k, v in skills.items():
            if isinstance(v, dict):
                sanitized_s = {
                    "name": k,
                    "proficiency": str(v.get("proficiency") or "intermediate"),
                    "duration_months": v.get("duration_months")
                }
                try:
                    sanitized_s["duration_months"] = float(sanitized_s["duration_months"]) if sanitized_s["duration_months"] is not None else 0.1
                except (ValueError, TypeError):
                    sanitized_s["duration_months"] = 0.1
                sanitized_skills.append(sanitized_s)
            else:
                sanitized_skills.append({
                    "name": k,
                    "proficiency": str(v),
                    "duration_months": 0.1
                })
    sanitized["skills"] = sanitized_skills

    signals = profile.get("redrob_signals")
    if not isinstance(signals, dict):
        signals = {}
    sanitized_signals = {
        "last_active_date": signals.get("last_active_date"),
        "recruiter_response_rate": signals.get("recruiter_response_rate"),
        "open_to_work_flag": signals.get("open_to_work_flag"),
        "notice_period_days": signals.get("notice_period_days"),
        "github_activity_score": signals.get("github_activity_score"),
        "interview_completion_rate": signals.get("interview_completion_rate")
    }
    sanitized["redrob_signals"] = sanitized_signals

    # Top level fallbacks for compatibility
    sanitized["years_of_experience"] = sanitized_p["years_of_experience"]
    sanitized["location"] = sanitized_p["location"]
    sanitized["current_title"] = sanitized_p["current_title"]
    sanitized["headline"] = sanitized_p["headline"]
    sanitized["summary"] = sanitized_p["summary"]

    return sanitized

def expand_query_jd(jd_text: str) -> str:
    """
    Use basic keyword/regex scanning to pull core requirements
    and append them to the raw JD string to optimize initial semantic recall precision.
    """
    expanded = [jd_text]
    jd_lower = jd_text.lower()

    keywords_mapping = {
        "senior": "Senior level ML practitioner, architectural experience.",
        "ai": "Artificial Intelligence, Deep Learning, ML systems.",
        "ml": "Machine Learning algorithms, model fine-tuning, training.",
        "embedding": "vector similarity search, dense embeddings, representation learning.",
        "faiss": "FAISS IndexFlatIP, approximate nearest neighbor search.",
        "sentence": "sentence-transformers, SentenceTransformer, all-MiniLM-L6-v2.",
        "retrieval": "dense retrieval, semantic index, candidate retrieval.",
        "ranking": "two-phase ranking, scoring formula, candidates ranking."
    }

    for kw, append_val in keywords_mapping.items():
        if kw in jd_lower:
            expanded.append(append_val)

    return " ".join(expanded)

def load_artifacts(artifacts_dir: Path) -> Tuple[faiss.Index, List[str]]:
    index_path = artifacts_dir / "faiss_index.bin"
    ids_path = artifacts_dir / "candidate_ids.pkl"

    if not index_path.exists():
        log.error("FAISS index not found: %s — run precompute.py first.", index_path)
        sys.exit(1)
    if not ids_path.exists():
        log.error("Candidate ID map not found: %s — run precompute.py first.", ids_path)
        sys.exit(1)

    log.info("Loading FAISS index from %s ...", index_path)
    index = faiss.read_index(str(index_path))
    log.info("  Index loaded: %d vectors, dim=%d", index.ntotal, index.d)

    log.info("Loading candidate ID map from %s ...", ids_path)
    with open(ids_path, "rb") as fh:
        candidate_ids = pickle.load(fh)
    log.info("  ID map loaded: %d entries", len(candidate_ids))

    assert index.ntotal == len(candidate_ids), (
        f"Mismatch: FAISS has {index.ntotal} vectors but ID map has {len(candidate_ids)} entries."
    )

    return index, [str(cid) for cid in candidate_ids]

def encode_query(model: SentenceTransformer, text: str) -> np.ndarray:
    vec = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return vec / norm

def faiss_search(index: faiss.Index, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    k = min(top_k, index.ntotal)
    distances, indices = index.search(query_vec, k)
    return distances[0], indices[0]

def load_profiles_from_jsonl(jsonl_path: Path, target_ids: set) -> dict:
    collected: dict = {}
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue

            cid = record.get("candidate_id") or record.get("id")
            if cid is None:
                continue
            cid_str = str(cid)
            if cid_str in target_ids:
                collected[cid_str] = record
                if len(collected) == len(target_ids):
                    break
    return collected

def score_candidate(profile: dict, semantic_score: float) -> Optional[dict]:
    cid = profile.get("candidate_id", "unknown")

    # 1. Honeypot check
    try:
        if is_honeypot(profile):
            log.warning("Honeypot detected — disqualifying candidate %s", cid)
            return {
                "candidate_id": cid,
                "final_score": 0.0,
                "semantic_score": semantic_score,
                "skills_score": 0.0,
                "career_score": 0.0,
                "experience_score": 0.0,
                "location_score": 0.0,
                "behavioral_multiplier": 0.50,
                "reasoning": f"Disqualified: honeypot profile signature detected."
            }
    except Exception as exc:
        log.warning("Honeypot check error for %s: %s — treating as clean.", cid, exc)

    # 2. Component scores
    try:
        skills_s = score_skills(profile)
    except Exception as exc:
        log.warning("score_skills error for %s: %s", cid, exc)
        skills_s = 0.0

    try:
        exp_years = profile.get("years_of_experience")
        experience_s = score_experience(exp_years)
    except Exception as exc:
        log.warning("score_experience error for %s: %s", cid, exc)
        experience_s = 0.0

    try:
        location_s = score_location(profile.get("location", ""))
    except Exception as exc:
        log.warning("score_location error for %s: %s", cid, exc)
        location_s = 0.65

    try:
        career_s = score_career(profile)
    except Exception as exc:
        log.warning("score_career error for %s: %s", cid, exc)
        career_s = 0.0

    try:
        beh_mult = behavioral_multiplier(profile.get("redrob_signals", {}) or {}, profile)
    except Exception as exc:
        log.warning("behavioral_multiplier error for %s: %s", cid, exc)
        beh_mult = 0.80

    # Bounded assertions checks
    assert 0.0 <= semantic_score <= 1.0, f"Semantic score out of bounds: {semantic_score}"
    assert 0.0 <= skills_s <= 1.0, f"Skills score out of bounds: {skills_s}"
    assert 0.0 <= career_s <= 1.0, f"Career score out of bounds: {career_s}"
    assert 0.0 <= experience_s <= 1.0, f"Experience score out of bounds: {experience_s}"
    assert 0.0 <= location_s <= 1.0, f"Location score out of bounds: {location_s}"

    # 3. Rubric calculation
    base_score = (0.35 * semantic_score) + (0.10 * skills_s) + (0.30 * career_s) + (0.15 * experience_s) + (0.10 * location_s)
    final_score = base_score * beh_mult
    final_score = max(0.0, min(1.0, final_score))

    # 4. Explanation reasoning
    try:
        metrics = {
            "candidate_id": cid,
            "profile": profile.get("profile", {}),
            "career_history": profile.get("career_history", []),
            "skills": profile.get("skills", []),
            "redrob_signals": profile.get("redrob_signals", {}),
            "semantic_score": semantic_score,
            "skills_score": skills_s,
            "career_score": career_s,
            "experience_score": experience_s,
            "location_score": location_s,
            "behavioral_multiplier": beh_mult,
            "final_score": final_score
        }
        reason = generate_reasoning(metrics, final_score)
    except Exception as exc:
        log.warning("generate_reasoning error for %s: %s", cid, exc)
        reason = f"Candidate score: {final_score:.4f}. Detailed reasoning unavailable."

    return {
        "candidate_id": cid,
        "final_score": final_score,
        "semantic_score": semantic_score,
        "skills_score": skills_s,
        "career_score": career_s,
        "experience_score": experience_s,
        "location_score": location_s,
        "behavioral_multiplier": beh_mult,
        "reasoning": reason,
    }

def _sort_key(result: dict) -> Tuple[float, str]:
    """
    Sort key descending by final_score.
    Resolve matching scores deterministically by sorting candidate_id alphabetically in ascending order.
    """
    return (-result["final_score"], str(result["candidate_id"]))

def write_csv(results: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        # Export matching columns: rank, candidate_id, score, reasoning
        writer = csv.DictWriter(fh, fieldnames=["rank", "candidate_id", "score", "reasoning"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "rank": row["rank"],
                "candidate_id": row["candidate_id"],
                "score": f"{row['final_score']:.6f}",
                "reasoning": row["reasoning"],
            })
    log.info("CSV written: %s (%d rows)", output_path, len(results))

def run_pipeline(jsonl_path: Path, artifacts_dir: Path, output_path: Path) -> None:
    t_start = time.perf_counter()

    # Step 1 & 2: Initialize SentenceTransformer and load artifacts
    log.info("Loading sentence-transformer model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    index, candidate_ids = load_artifacts(artifacts_dir)

    # Step 3: Run query expansion on JD, encode, and perform FAISS search
    log.info("Expanding Job Description...")
    expanded_jd = expand_query_jd(JOB_DESCRIPTION)
    log.info("Encoding Job Description query...")
    query_vec = encode_query(model, expanded_jd)

    log.info("Searching FAISS index for top %d semantic matches...", FAISS_TOP_K)
    distances, nn_indices = faiss_search(index, query_vec, FAISS_TOP_K)

    # Convert rows back to string IDs using the pickle map
    semantic_map: dict = {}
    for dist, idx in zip(distances, nn_indices):
        if idx < 0 or idx >= len(candidate_ids):
            continue
        cid = str(candidate_ids[idx])
        cos_sim = float(max(0.0, min(1.0, dist)))
        semantic_map[cid] = cos_sim

    # Step 5: Parse candidates.jsonl efficiently to load candidate dicts for the target IDs
    log.info("Loading profiles from %s...", jsonl_path)
    target_ids = set(semantic_map.keys())
    profile_map = load_profiles_from_jsonl(jsonl_path, target_ids)
    log.info("Loaded %d profiles from JSONL.", len(profile_map))

    # Step 6 & 7: Sanitize, score and honeypot filter
    log.info("Scoring candidates...")
    valid_results: list = []
    honeypot_count = 0

    for cid, semantic_score in semantic_map.items():
        record = profile_map.get(cid)
        if record is None:
            continue

        try:
            profile = sanitize_profile(record)
        except Exception as exc:
            log.warning("Sanitization failed for %s: %s", cid, exc)
            continue

        result = score_candidate(profile, semantic_score)
        if result is None:
            continue

        if result["final_score"] == 0.0 and "Disqualified" in result["reasoning"]:
            honeypot_count += 1

        valid_results.append(result)

    # Step 9: Sort deterministically
    log.info("Sorting candidates deterministically...")
    valid_results.sort(key=_sort_key)

    # Assign ranks and take top 100
    top_results = valid_results[:OUTPUT_TOP_N]
    for rank_pos, res in enumerate(top_results, start=1):
        res["rank"] = rank_pos

    # Step 10: Export to CSV
    log.info("Writing output CSV...")
    write_csv(top_results, output_path)

    elapsed = time.perf_counter() - t_start
    log.info("Pipeline complete in %.2f seconds.", elapsed)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 — Rank top-100 candidates against the AI Engineer JD"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "candidates.jsonl",
        help="Path to candidates.jsonl (default: ./candidates.jsonl)",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directory containing faiss_index.bin and candidate_ids.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Output CSV path (default: ./results/top100_candidates.csv)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not args.input.exists():
        log.error("Input JSONL not found: %s", args.input)
        sys.exit(1)

    run_pipeline(args.input, args.artifacts, args.output)
