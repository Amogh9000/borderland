"""
reasoning.py — Phase 2 Component (Amended per §6)
Provides:
  generate_reasoning(metrics_or_profile, final_score) -> str

References career description evidence first — NOT the skills list.
"""

import re
import logging
from typing import Optional, Dict, Any, List

# Configure logging
log = logging.getLogger(__name__)

def normalize_string(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())

def generate_reasoning(profile_or_metrics: Dict[str, Any], final_score: Optional[float] = None) -> str:
    """
    Generate a data-driven reasoning string referencing actual career evidence.
    Target: <=200 characters. Must NOT reference skills list as primary evidence.

    Example:
      "ML Eng @ Razorpay (6yr); built recommendation system; open to work; response rate >82%; high JD-profile semantic match"
    """
    if not profile_or_metrics or not isinstance(profile_or_metrics, dict):
        return "Composite ranking based on semantic match and candidate profile details."

    parts: list = []

    # 1. Career anchor — most recent relevant role
    career = profile_or_metrics.get("career_history", []) or []
    profile = profile_or_metrics.get("profile", {}) or profile_or_metrics

    ex_company = ""
    latest_title = ""
    duration_yr = 0
    if career and isinstance(career, list):
        first_job = career[0]
        if isinstance(first_job, dict):
            ex_company = str(first_job.get("company", "")).strip()
            latest_title = str(first_job.get("title", "")).strip()
            try:
                duration_yr = int(float(first_job.get("duration_months", 0) or 0) // 12)
            except (ValueError, TypeError):
                duration_yr = 0

    if not latest_title:
        latest_title = str(profile.get("current_title", "")).strip()
    if not ex_company:
        ex_company = str(profile.get("current_company", "")).strip()

    if ex_company and latest_title:
        yr_str = f" ({duration_yr}yr)" if duration_yr > 0 else ""
        parts.append(f"{latest_title} @ {ex_company}{yr_str}")
    elif latest_title:
        parts.append(latest_title)

    # 2. Career description evidence — what they actually built
    career_text = " ".join(
        j.get("description", "") for j in career if isinstance(j, dict)
    ).lower()

    EVIDENCE_PHRASES = [
        ("faiss",          "shipped FAISS to production"),
        ("recommendation", "built recommendation system"),
        ("retrieval",      "built retrieval pipeline"),
        ("fine-tun",       "fine-tuned LLM"),
        ("deployed",       "deployed ML model"),
        ("embedding",      "worked on embeddings"),
        ("production",     "shipped to production"),
        ("search",         "built search system"),
        ("pipeline",       "built ML pipeline"),
        ("vector",         "worked on vector systems"),
    ]
    evidence_found = False
    for kw, label in EVIDENCE_PHRASES:
        if kw in career_text:
            parts.append(label)
            evidence_found = True
            break  # one career evidence phrase is enough

    if not evidence_found and career_text:
        parts.append("relevant career experience")

    # 3. Behavioral signals (availability)
    signals = profile_or_metrics.get("redrob_signals", {}) or {}
    open_to_work = signals.get("open_to_work_flag", False) or signals.get("open_to_work", False)
    if open_to_work:
        parts.append("open to work")

    resp_rate = signals.get("recruiter_response_rate", None)
    if resp_rate is not None:
        try:
            rr = float(resp_rate)
            if rr < 0.3:
                parts.append(f"low response rate ({int(rr*100)}%)")
            elif rr > 0.70:
                parts.append(f"response rate >{int(rr*100)}%")
        except (ValueError, TypeError):
            pass

    # 4. Semantic score mention only if high
    sem = profile_or_metrics.get("semantic_score")
    if sem is not None:
        try:
            if float(sem) >= 0.85:
                parts.append("high JD-profile semantic match")
        except (ValueError, TypeError):
            pass

    reasoning = "; ".join(parts)
    reasoning = normalize_string(reasoning)

    # Hard cap at 200 characters
    if len(reasoning) > 200:
        reasoning = reasoning[:197] + "..."

    return reasoning
