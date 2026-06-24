import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# ── Tier 1: Core ML / AI / IR engineering titles ────────────────────────────
_TITLE_CORE_ML = [
    "ai", "ml", "machine learning", "nlp", "deep learning",
    "computer vision", "recommendation", "search engineer",
    "applied scientist", "research engineer", "data scientist",
    "ranking", "retrieval", "llm", "language model",
]

# ── Tier 2: General software / systems engineering (adjacent but not ideal) ──
_TITLE_SOFT_ENG = [
    "software engineer", "software developer", "swe", "backend",
    "systems engineer", "infrastructure", "platform engineer",
    "full stack", "fullstack", "tech lead", "engineering manager",
    "principal engineer", "staff engineer", "senior engineer",
    "data engineer",
]

# ── Tier 3: Non-technical / entirely decoupled roles → catastrophic score ────
_TITLE_DECOUPLED = [
    "accountant", "accounting", "sales", "business development",
    "hr ", "human resource", "recruiter", "talent acquisition",
    "marketing", "content ", "seo", "copywriter",
    "product manager", "program manager", "project manager",
    "operations manager", "supply chain", "logistics",
    "graphic designer", "ui designer", "ux designer",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "customer support", "customer success", "account manager",
    "finance", "investment", "analyst",
    "lawyer", "legal", "doctor", "physician",
]

# ── IT-services / consulting firms blocklist ─────────────────────────────────
SERVICES_BLOCKLIST = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "mphasis", "hexaware", "tech mahindra",
    "hcl technologies", "hcl tech", "l&t infotech", "ltimindtree",
    "mindtree", "niit technologies", "mastech",
}

# ── Title-inflation keywords that signal title-chasing ───────────────────────
_CHASER_TITLE_SIGNALS = ["staff", "principal", "director", "vp ", "vice president", "head of"]

# ── Hidden-gem IR / foundational systems keywords ────────────────────────────
_IR_FOUNDATION_KEYWORDS = [
    "search engine", "information retrieval", "ranking system",
    "recommendation engine", "matching platform", "solr",
    "elasticsearch", "opensearch", "lucene", "bm25",
    "collaborative filtering", "matrix factorization",
    "learning to rank", "reranking", "recall@", "ndcg", "mrr",
]

# ── JD bracket hard limits ───────────────────────────────────────────────────
YOE_IDEAL_MIN = 5.0
YOE_IDEAL_MAX = 9.0
SEMANTIC_ZERO_SKILL_PARDON = 0.65
OTW_PASSIVE_MULTIPLIER = 0.20


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def classify_title(title: str) -> tuple[float, str]:
    if not title:
        return 0.30, "unknown"
    tl = title.lower()
    if any(k in tl for k in _TITLE_CORE_ML):
        return 1.00, "core_ml"
    if any(k in tl for k in _TITLE_DECOUPLED):
        return 0.00, "decoupled"
    if any(k in tl for k in _TITLE_SOFT_ENG):
        return 0.35, "soft_eng"
    return 0.30, "unknown"


def title_relevance_score(title: str) -> float:
    """Backward-compatibility shim."""
    mult, _ = classify_title(title)
    return mult


def _is_services_company(name: str) -> bool:
    nl = name.lower()
    return any(s in nl for s in SERVICES_BLOCKLIST)


def consulting_career_multiplier(career_companies: list) -> tuple[float, str]:
    if not career_companies:
        return 1.00, "ok"
    non_empty = [c for c in career_companies if c]
    if not non_empty:
        return 1.00, "ok"
    services_flags = [_is_services_company(c) for c in non_empty]
    if all(services_flags):
        return 0.05, "pure_services"
    return 1.00, "ok"


def title_chaser_multiplier(yoe: float, n_unique_companies: int, career_titles: list) -> tuple[float, bool]:
    if n_unique_companies <= 1 or yoe <= 0:
        return 1.00, False
    avg_tenure = yoe / n_unique_companies
    if avg_tenure > 1.7:
        return 1.00, False
    titles_blob = " ".join(career_titles).lower()
    if any(sig in titles_blob for sig in _CHASER_TITLE_SIGNALS):
        return 0.60, True
    return 1.00, False


def hidden_gem_bonus(career_text_blob: str, yoe: float, career_companies: list) -> float:
    blob = (career_text_blob or "").lower()
    has_ir_signal = any(kw in blob for kw in _IR_FOUNDATION_KEYWORDS)
    if not has_ir_signal:
        return 0.0
    _, svc_flag = consulting_career_multiplier(career_companies)
    non_svc_yoe = 0.0 if svc_flag == "pure_services" else yoe
    if non_svc_yoe >= 4.0:
        return 15.0
    return 0.0


def engagement_multiplier(days_inactive: int, recruiter_response_rate: float) -> tuple[float, str]:
    if days_inactive > 180:
        return 0.20, "ghost_180d"
    if days_inactive > 90:
        mult = 0.20 + 0.60 * np.exp(-(days_inactive - 90) / 60.0)
        mult = float(np.clip(mult, 0.20, 0.80))
        return mult, f"inactive_{days_inactive}d"
    return 1.00, "ok"


# ════════════════════════════════════════════════════════════════════════════
# SCORE COMPONENTS
# ════════════════════════════════════════════════════════════════════════════

def skill_match_score(candidate_skills, jd_required, jd_preferred):
    if not jd_required and not jd_preferred:
        return 1.0, []
    cand_names = [s.get('name', '').lower() for s in candidate_skills]
    if not cand_names:
        return 0.0, []

    req_hits, matched_skills = 0, []
    for req in jd_required:
        best = max([fuzz.token_sort_ratio(req.lower(), cs) for cs in cand_names] or [0])
        if best > 80:
            req_hits += 1
            matched_skills.append(req)

    pref_hits = 0
    for pref in jd_preferred:
        best = max([fuzz.token_sort_ratio(pref.lower(), cs) for cs in cand_names] or [0])
        if best > 80:
            pref_hits += 1

    req_total = len(jd_required) if jd_required else 1
    pref_total = len(jd_preferred) if jd_preferred else 1
    score = (req_hits / req_total) * 0.7 + (pref_hits / pref_total) * 0.3
    return min(1.0, score), matched_skills


def salary_overlap_score(c_min, c_max, jd_range) -> float:
    jd_min, jd_max = jd_range
    if c_max < jd_min or c_min > jd_max:
        return 0.0
    if c_min >= jd_min and c_max <= jd_max:
        return 1.0
    return 0.5


def notice_period_multiplier(days: int) -> float:
    if days <= 30:
        return 1.00
    elif days <= 45:
        return 0.80
    elif days <= 60:
        return 0.50
    else:
        return 0.10


def yoe_bracket_penalty(yoe: float) -> float:
    if yoe < YOE_IDEAL_MIN:
        return max(0.0, (YOE_IDEAL_MIN - yoe) * 15.0)
    elif yoe > YOE_IDEAL_MAX:
        return min(10.0, (yoe - YOE_IDEAL_MAX) * 2.0)
    return 0.0


def jd_fit_score(row, jd_parsed):
    yoe = row['years_of_experience']
    jd_min = jd_parsed['min_years_experience']
    exp_gap = abs(yoe - jd_min)
    exp_score = max(0.0, 1.0 - exp_gap * 0.12)

    mode_match = 1.0 if row['preferred_work_mode'] == jd_parsed['work_mode'] else 0.5

    skill_match, matched_req = skill_match_score(
        row['skills'], jd_parsed['required_skills'], jd_parsed['preferred_skills']
    )

    title_mult, title_tier = classify_title(row.get('current_title', ''))
    semantic = row['semantic_score']
    
    if len(matched_req) == 0 and semantic <= SEMANTIC_ZERO_SKILL_PARDON:
        skill_match *= 0.50

    raw_fit = (
        0.45 * semantic
      + 0.30 * skill_match
      + 0.10 * exp_score
      + 0.10 * (1.0 if row.get('github_activity_score', -1) >= 50 else 0.0)
      + 0.05 * mode_match
    )

    if len(matched_req) == 0 and semantic <= SEMANTIC_ZERO_SKILL_PARDON:
        raw_fit = min(raw_fit, 0.55)

    gated_fit = raw_fit * title_mult
    return gated_fit, matched_req, title_tier


def quality_score(row) -> float:
    prof_map = {'advanced': 1.0, 'intermediate': 0.6, 'beginner': 0.3}
    avg_prof = (
        np.mean([prof_map.get(s.get('proficiency', 'intermediate'), 0.6) for s in row['skills']])
        if row['skills'] else 0.5
    )
    assessment_scores = (
        list(row['skill_assessment_scores'].values())
        if isinstance(row['skill_assessment_scores'], dict) else []
    )
    assessment_avg = (np.mean(assessment_scores) / 100.0) if assessment_scores else 0.5

    edu_map = {'tier_1': 1.0, 'tier_2': 0.7, 'tier_3': 0.45}
    edu_score = edu_map.get(row['education_tier'], 0.4)

    github_val = row['github_activity_score']
    github = 0.5 if github_val < 0 else github_val / 100.0

    endorse = min(np.log1p(row['endorsements_received']) / np.log1p(200), 1.0)

    return float(
        0.35 * avg_prof
      + 0.30 * assessment_avg
      + 0.15 * edu_score
      + 0.10 * github
      + 0.10 * endorse
    )


def platform_score(row, today) -> float:
    days_inactive = (today - row['last_active_date']).days if hasattr(row['last_active_date'], 'days') else 0
    recency = np.exp(-days_inactive / 30.0)

    oar_raw = row['offer_acceptance_rate']
    offer_acceptance = 0.6 if oar_raw < 0 else oar_raw

    reliability = (
        0.4 * row['interview_completion_rate']
      + 0.3 * offer_acceptance
      + 0.3 * (1.0 - min(row['avg_response_time_hours'], 168.0) / 168.0)
    )

    market = min(row['saved_by_recruiters_30d'] / 10.0, 1.0)
    complete = row['profile_completeness_score'] / 100.0

    base_plat = (
        0.25 * recency
      + 0.25 * reliability
      + 0.20 * market
      + 0.15 * complete
      + 0.15 * 1.0
    )

    rrr = row.get('recruiter_response_rate', 0.5)
    if rrr < 0.15:
        base_plat = 0.0

    if not row.get('open_to_work_flag', True):
        base_plat *= OTW_PASSIVE_MULTIPLIER

    return base_plat


# ════════════════════════════════════════════════════════════════════════════
# FINAL ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════

def compute_scores(df: pd.DataFrame, jd: dict) -> pd.DataFrame:
    import datetime
    today = datetime.date.today()

    if len(df) == 0:
        for col in ['jd_fit_score', 'matched_required_skills',
                    'matched_preferred_skills', 'quality_score',
                    'platform_score', 'final_score', 'title_tier',
                    'score_flags']:
            df[col] = []
        return df

    def process_row(row):
        jd_fit, matched_req, title_tier = jd_fit_score(row, jd)
        qual = quality_score(row)
        plat = platform_score(row, today)

        # Non-linear scaling applied via exponentiation
        base_score = (jd_fit ** 2 * 0.55 + qual * 0.25 + plat * 0.20) * 100.0

        yoe = row['years_of_experience']
        yoe_deduction = yoe_bracket_penalty(yoe)
        base_score = max(0.0, base_score - yoe_deduction)

        gem_bonus = hidden_gem_bonus(
            row.get('career_text_blob', ''),
            yoe,
            row.get('career_companies', []),
        )
        base_score = min(100.0, base_score + gem_bonus)

        flags = []

        np_mult = notice_period_multiplier(int(row.get('notice_period_days', 0)))
        base_score *= np_mult

        days_inactive = (today - row['last_active_date']).days if hasattr(row['last_active_date'], 'days') else 0
        eng_mult, eng_flag = engagement_multiplier(days_inactive, row.get('recruiter_response_rate', 0.5))
        if eng_flag != "ok":
            flags.append(eng_flag)
        base_score *= eng_mult

        svc_mult, svc_flag = consulting_career_multiplier(row.get('career_companies', []))
        if svc_flag != "ok":
            flags.append(svc_flag)
        base_score *= svc_mult

        tc_mult, tc_flagged = title_chaser_multiplier(
            yoe,
            row.get('n_unique_companies', 1),
            row.get('career_titles', []),
        )
        if tc_flagged:
            flags.append("title_chaser")
        base_score *= tc_mult

        final = min(100.0, max(0.0, base_score))

        return (
            jd_fit * 100.0,
            matched_req,
            title_tier,
            qual * 100.0,
            plat * 100.0,
            final,
            gem_bonus > 0,
            '; '.join(flags),
        )

    results = df.apply(process_row, axis=1, result_type='expand')
    df['jd_fit_score'] = results[0]
    df['matched_required_skills'] = results[1]
    df['title_tier'] = results[2]
    df['matched_preferred_skills'] = [[]] * len(df)
    df['quality_score'] = results[3]
    df['platform_score'] = results[4]
    df['final_score'] = results[5]
    df['is_hidden_gem'] = results[6]
    df['score_flags'] = results[7]

    # ── 99th-percentile OTW pardon ────────────────────────────────────────
    if 'semantic_score' in df.columns:
        p99 = df['semantic_score'].quantile(0.99)
        pardon_mask = (~df['open_to_work_flag']) & (df['semantic_score'] >= p99)
        if pardon_mask.any():
            def recompute_pardoned(row):
                jd_fit, matched_req, _tier = jd_fit_score(row, jd)
                qual = quality_score(row)
                days_inactive = (today - row['last_active_date']).days if hasattr(row['last_active_date'], 'days') else 0
                recency = np.exp(-days_inactive / 30.0)
                oar_raw = row['offer_acceptance_rate']
                offer_acceptance = 0.6 if oar_raw < 0 else oar_raw
                reliability = (
                    0.4 * row['interview_completion_rate']
                  + 0.3 * offer_acceptance
                  + 0.3 * (1.0 - min(row['avg_response_time_hours'], 168.0) / 168.0)
                )
                market = min(row['saved_by_recruiters_30d'] / 10.0, 1.0)
                complete = row['profile_completeness_score'] / 100.0
                rrr = row.get('recruiter_response_rate', 0.5)
                
                plat_pardoned = (
                    (0.25*recency + 0.25*reliability + 0.20*market + 0.15*complete + 0.15)
                    if rrr >= 0.15 else 0.0
                )

                yoe = row['years_of_experience']
                base_score = (jd_fit**2 * 0.55 + qual * 0.25 + plat_pardoned * 0.20) * 100.0
                base_score = max(0.0, base_score - yoe_bracket_penalty(yoe))
                base_score = min(100.0, base_score + hidden_gem_bonus(
                    row.get('career_text_blob', ''), yoe, row.get('career_companies', [])))

                np_mult = notice_period_multiplier(int(row.get('notice_period_days', 0)))
                eng_mult, _ = engagement_multiplier(days_inactive, rrr)
                svc_mult, _ = consulting_career_multiplier(row.get('career_companies', []))
                tc_mult, _ = title_chaser_multiplier(yoe, row.get('n_unique_companies', 1), row.get('career_titles', []))

                final = min(100.0, max(0.0, base_score * np_mult * eng_mult * svc_mult * tc_mult))
                return plat_pardoned * 100.0, final

            pardoned = df[pardon_mask].apply(recompute_pardoned, axis=1, result_type='expand')
            df.loc[pardon_mask, 'platform_score'] = pardoned[0].values
            df.loc[pardon_mask, 'final_score'] = pardoned[1].values

    return df