"""
reasoning.py — generates a structured narrative for each Top-100 candidate.

Concern severity tiers:
  CRITICAL  — notice > 60d (0.10x), pure IT-services career (0.05x)
  HIGH      — notice 46-60d (0.50x), passive flag, 0 required skills,
               YoE outside [5,9], decoupled title, ghost inactivity
  MODERATE  — notice 31-45d (0.80x), title-chaser, low GitHub, low assessments
  SYSTEM NOTE — positive signals (hidden-gem IR bonus, semantic pardon)
"""

from score import (
    YOE_IDEAL_MIN,
    YOE_IDEAL_MAX,
    SEMANTIC_ZERO_SKILL_PARDON,
    notice_period_multiplier,
    consulting_career_multiplier,
    title_chaser_multiplier,
    classify_title,
)


def generate_reasoning(row, jd_parsed) -> str:
    parts = []

    # ── 1. Semantic alignment headline ───────────────────────────────────────
    sim_pct = round(row.get('semantic_score', 0) * 100)
    parts.append(f"Profile text aligns {sim_pct}% with the JD.")

    # ── 2. Skills verdict ────────────────────────────────────────────────────
    matched = row.get('matched_required_skills', [])
    n_req   = len(jd_parsed.get('required_skills', []))
    if matched:
        named = ', '.join(matched[:4])
        parts.append(f"Matches {len(matched)}/{n_req} required skills: {named}.")
    else:
        sem = row.get('semantic_score', 0)
        if sem > SEMANTIC_ZERO_SKILL_PARDON:
            parts.append(
                f"No required skills matched; high semantic score ({sim_pct}%) partially pardons."
            )
        else:
            parts.append(
                f"0/{n_req} required skills matched — jd_fit capped + skill signal halved."
            )

    # ── 3. Experience & title ────────────────────────────────────────────────
    yoe   = row.get('years_of_experience', 0)
    title = row.get('current_title', '')
    co    = row.get('current_company', '')
    parts.append(f"{yoe} yrs as {title} at {co}.")

    # ── 4. Top assessment ────────────────────────────────────────────────────
    scores = row.get('skill_assessment_scores', {})
    if isinstance(scores, dict) and scores:
        top_s, top_v = max(scores.items(), key=lambda x: x[1])
        parts.append(f"Top assessment: {top_s} {top_v}/100.")

    # ── 5. Availability line ─────────────────────────────────────────────────
    otw    = row.get('open_to_work_flag', True)
    notice = int(row.get('notice_period_days', 0))
    np_m   = notice_period_multiplier(notice)
    otw_str = "Active & open to work." if otw else "NOT open to work."
    parts.append(f"{otw_str} Notice: {notice}d (×{np_m:.2f}).")

    # ── 6. Concerns — ordered by severity ────────────────────────────────────
    concerns = []

    # CRITICAL
    if notice > 60:
        concerns.append(
            f"CRITICAL: {notice}-day notice → 0.10× multiplier; founding-team SLA violated"
        )

    career_companies = row.get('career_companies', [])
    _, svc_flag = consulting_career_multiplier(career_companies)
    if svc_flag == "pure_services":
        concerns.append(
            "CRITICAL: Career history is 100% IT Services (TCS/Infosys/Wipro/etc.) — "
            "fails product-experience constraint; 0.05× multiplier applied"
        )

    # HIGH
    if 45 < notice <= 60:
        concerns.append(f"HIGH: {notice}-day notice → 0.50× multiplier")

    if not otw:
        concerns.append(
            "HIGH: passive candidate (open_to_work=False) → platform score ×0.20 "
            "unless top-1% semantic"
        )

    if len(matched) == 0 and row.get('semantic_score', 0) <= SEMANTIC_ZERO_SKILL_PARDON:
        concerns.append(
            f"HIGH: 0 required skills & semantic ≤{int(SEMANTIC_ZERO_SKILL_PARDON*100)}% "
            "→ jd_fit capped + skill halved"
        )

    if yoe < YOE_IDEAL_MIN:
        deficit    = round(YOE_IDEAL_MIN - yoe, 1)
        deduction  = round(deficit * 15, 1)
        concerns.append(
            f"HIGH: {yoe} yrs is {deficit} yrs below 5-yr floor → -{deduction} pt YoE penalty"
        )
    elif yoe > YOE_IDEAL_MAX:
        excess = round(yoe - YOE_IDEAL_MAX, 1)
        concerns.append(f"MODERATE: {yoe} yrs exceeds 9-yr cap by {excess} yrs → minor taper")

    title_mult, title_tier = classify_title(title)
    if title_tier == "decoupled":
        concerns.append(
            f"HIGH CONCERN: Current role as '{title}' is entirely decoupled from the "
            "required ML/Systems engineering profile → jd_fit ×0.00"
        )
    elif title_tier == "soft_eng":
        concerns.append(
            f"HIGH: Title '{title}' is adjacent SWE, not core ML/AI → jd_fit ×0.35"
        )

    score_flags = row.get('score_flags', '')
    if 'ghost_180d' in score_flags or 'inactive_' in score_flags:
        days_inactive = 0
        if hasattr(row.get('last_active_date'), 'days'):
            import datetime
            days_inactive = (datetime.date.today() - row['last_active_date']).days
        if days_inactive > 180:
            concerns.append(
                f"HIGH: Profile inactive for {days_inactive} days → 0.20× engagement multiplier"
            )
        elif days_inactive > 90:
            concerns.append(
                f"HIGH: Profile inactive for {days_inactive} days → exponential engagement decay"
            )

    rrr = row.get('recruiter_response_rate', 0.5)
    if rrr < 0.15:
        concerns.append(
            f"HIGH: Recruiter response rate {round(rrr*100)}% < 15% → platform score zeroed"
        )

    # MODERATE
    if 30 < notice <= 45:
        concerns.append(f"MODERATE: {notice}-day notice → 0.80× multiplier")

    if 'title_chaser' in score_flags:
        n_cos = row.get('n_unique_companies', 1)
        avg_tenure = round(yoe / max(1, n_cos), 1)
        concerns.append(
            f"CRITICAL: Title-chaser velocity detected — avg tenure {avg_tenure} yrs across "
            f"{n_cos} companies with title inflation → 0.60× multiplier"
        )

    github_score = row.get('github_activity_score', 100)
    if 0 <= github_score < 20:
        concerns.append(f"low GitHub activity ({github_score}/100)")

    if isinstance(scores, dict):
        low_scores = [f"{k}:{v}" for k, v in scores.items() if v < 40]
        if low_scores:
            concerns.append("assessment scores <40: " + ', '.join(low_scores))

    oar = row.get('offer_acceptance_rate', 1.0)
    if 0 <= oar < 0.4:
        concerns.append("low offer acceptance rate")

    if concerns:
        parts.append("Concerns: " + '; '.join(concerns) + ".")

    # ── 7. Positive system notes ─────────────────────────────────────────────
    if row.get('is_hidden_gem', False):
        parts.append(
            "SYSTEM NOTE: Protected Core IR Engineer — foundational "
            "search/recommendation/ranking expertise recognized; +15 pt bonus applied."
        )

    return ' '.join(parts)[:600]
