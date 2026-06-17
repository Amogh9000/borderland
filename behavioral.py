"""
behavioral.py — Phase 2 Component
Provides:
  calculate_behavioral_multiplier(signals) -> float
  behavioral_multiplier(signals) -> float (alias)
"""

import logging
from datetime import date, datetime
from typing import Union, Dict, Any, Optional

# Configure logging
log = logging.getLogger(__name__)

# Baseline date for recency calculations (per system spec/original file)
BASELINE_DATE = date(2026, 6, 16)

def _parse_date(val: Any) -> Union[date, None]:
    """Parse a date value from various formats."""
    if val is None:
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, str):
        val_str = val.strip()
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                    "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(val_str[:len(fmt)], fmt).date()
            except ValueError:
                continue
        try:
            return date.fromisoformat(val_str[:10])
        except ValueError:
            return None
    return None

def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def _safe_bool(val: Any, default: bool = True) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return bool(val)

def calculate_behavioral_multiplier(signals: dict, profile: Optional[dict] = None) -> float:
    """
    Compute a multiplicative score modifier from the candidate's redrob_signals.
    Accumulate individual candidate signals multiplicatively:
      - last_active_date older than 6 months (180 days) = 0.6x multiplier
      - recruiter_response_rate < 0.15 = 0.7x multiplier; > 0.70 = 1.05x multiplier
      - open_to_work_flag == False = 0.85x multiplier
      - notice_period_days > 90 days = 0.75x multiplier
      - github_activity_score > 70 = 1.05x multiplier
      - interview_completion_rate < 0.4 = 0.85x multiplier
    System Safety Floor: max(calculated_multiplier, 0.50).
    Log active signals for the reasoning engine.
    """
    if not signals or not isinstance(signals, dict):
        log.info("No signals dictionary found. Returning neutral default: 0.80")
        return 0.80

    multiplier = 1.0
    active_signals = []

    # 1. Activity recency
    last_active_raw = signals.get("last_active_date")
    last_active = _parse_date(last_active_raw)
    if last_active is not None:
        days_inactive = (BASELINE_DATE - last_active).days
        if days_inactive > 180:
            multiplier *= 0.6
            active_signals.append("inactive > 6 months")

    # 2. Recruiter response rate
    response_rate_raw = signals.get("recruiter_response_rate")
    if response_rate_raw is not None:
        response_rate = _safe_float(response_rate_raw, default=-1.0)
        if response_rate >= 0.0:
            if response_rate < 0.15:
                multiplier *= 0.7
                active_signals.append("low response rate")
            elif response_rate > 0.70:
                multiplier *= 1.05
                active_signals.append("high response rate")

    # 3. Open to work flag
    open_to_work_raw = signals.get("open_to_work_flag")
    if open_to_work_raw is not None:
        open_to_work = _safe_bool(open_to_work_raw, default=True)
        if not open_to_work:
            multiplier *= 0.85
            active_signals.append("not open to work")

    # 4. Notice period
    notice_raw = signals.get("notice_period_days")
    if notice_raw is not None:
        notice_days = _safe_float(notice_raw, default=0.0)
        if notice_days > 90:
            multiplier *= 0.75
            active_signals.append("long notice period")

    # 5. GitHub activity score
    github_raw = signals.get("github_activity_score")
    if github_raw is not None:
        github_score = _safe_float(github_raw, default=0.0)
        if github_score > 70:
            multiplier *= 1.05
            active_signals.append("high github activity")

    # 6. Interview completion rate
    interview_raw = signals.get("interview_completion_rate")
    if interview_raw is not None:
        interview_rate = _safe_float(interview_raw, default=1.0)
        if interview_rate < 0.4:
            multiplier *= 0.85
            active_signals.append("low interview completion")

    # 7. Soft Honeypot flag check
    if profile and profile.get('_soft_honeypot_flag'):
        multiplier *= 0.70  # Arbitrary but strong penalty
        active_signals.append("soft honeypot flag")

    # Clamp with the safety floor
    final_mult = max(multiplier, 0.50)
    
    # Cap at 1.10 as standard safety upper bound
    final_mult = min(final_mult, 1.10)

    log.info("Behavioral signals matched: %s. Multiplier calculated: %.4f (final: %.4f)",
             ", ".join(active_signals) if active_signals else "none", multiplier, final_mult)

    return final_mult

# Alias for compatibility with the existing rank.py calling conventions
behavioral_multiplier = calculate_behavioral_multiplier
