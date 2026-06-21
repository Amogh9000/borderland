import numpy as np
import pandas as pd
from rapidfuzz import fuzz

RELEVANT_TITLE_KEYWORDS = ["ai", "ml", "machine learning", "nlp", "data scientist", "research engineer", "applied scientist", "deep learning", "computer vision", "recommendation", "search engineer"]
IRRELEVANT_TITLES = ["hr manager", "operations manager", "civil engineer", "mechanical engineer", "graphic designer", "customer support", "business analyst", "data analyst", "java developer", "devops engineer", "backend engineer", "project manager"]

def title_relevance_score(title):
    if not title:
        return 0.5
    title_lower = str(title).lower()
    if any(k in title_lower for k in RELEVANT_TITLE_KEYWORDS):
        return 1.0
    if any(k in title_lower for k in IRRELEVANT_TITLES):
        return 0.15
    return 0.5

def skill_match_score(candidate_skills, jd_required, jd_preferred):
    if not jd_required and not jd_preferred:
        return 1.0, []
        
    cand_skill_names = [s.get('name', '').lower() for s in candidate_skills]
    if not cand_skill_names:
        return 0.0, []
        
    req_hits = 0
    matched_skills = []
    for req in jd_required:
        score = max([fuzz.token_sort_ratio(req.lower(), cs) for cs in cand_skill_names] or [0])
        if score > 80:
            req_hits += 1
            matched_skills.append(req)
            
    pref_hits = 0
    for pref in jd_preferred:
        score = max([fuzz.token_sort_ratio(pref.lower(), cs) for cs in cand_skill_names] or [0])
        if score > 80:
            pref_hits += 1
            
    req_total = len(jd_required) if jd_required else 1
    pref_total = len(jd_preferred) if jd_preferred else 1
    
    score = (req_hits / req_total) * 0.7 + (pref_hits / pref_total) * 0.3
    return min(1.0, score), matched_skills

def salary_overlap_score(c_min, c_max, jd_range):
    jd_min, jd_max = jd_range
    if c_max < jd_min or c_min > jd_max:
        return 0.0
    if c_min >= jd_min and c_max <= jd_max:
        return 1.0
    return 0.5

def jd_fit_score(row, jd_parsed):
    exp_gap = abs(row['years_of_experience'] - jd_parsed['min_years_experience'])
    exp_score = max(0, 1 - exp_gap * 0.12)
    
    mode_match = 1.0 if row['preferred_work_mode'] == jd_parsed['work_mode'] else 0.5
    
    sal_overlap = salary_overlap_score(
        row['expected_salary_min_lpa'], 
        row['expected_salary_max_lpa'],
        jd_parsed['salary_range_lpa']
    )
    
    skill_match, matched_req = skill_match_score(row['skills'], jd_parsed['required_skills'], jd_parsed['preferred_skills'])
    title_score = title_relevance_score(row.get('current_title', ''))
    
    return (
        0.35 * row['semantic_score']
      + 0.25 * skill_match
      + 0.20 * title_score
      + 0.10 * exp_score
      + 0.05 * mode_match
      + 0.05 * sal_overlap
    ), matched_req

def quality_score(row):
    prof_map = {'advanced': 1.0, 'intermediate': 0.6, 'beginner': 0.3}
    if row['skills']:
        avg_prof = np.mean([prof_map.get(s.get('proficiency', 'intermediate'), 0.6) for s in row['skills']])
    else:
        avg_prof = 0.5
        
    assessment_scores = list(row['skill_assessment_scores'].values()) if isinstance(row['skill_assessment_scores'], dict) else []
    assessment_avg = (np.mean(assessment_scores) / 100.0) if assessment_scores else 0.5
    
    edu_map = {'tier_1': 1.0, 'tier_2': 0.7, 'tier_3': 0.45}
    edu_score = edu_map.get(row['education_tier'], 0.4)
    
    github_val = row['github_activity_score']
    github = 0.5 if github_val < 0 else github_val / 100.0
    
    endorse = min(np.log1p(row['endorsements_received']) / np.log1p(200), 1.0)
    
    return (
        0.35 * avg_prof
      + 0.30 * assessment_avg
      + 0.15 * edu_score
      + 0.10 * github
      + 0.10 * endorse
    )

def notice_period_score(days):
    if days <= 30:
        return 1.0
    elif days <= 60:
        return 0.85
    elif days <= 90:
        return 0.65
    else:
        return 0.45

def platform_score(row, today):
    days_inactive = (today - row['last_active_date']).days if hasattr(row['last_active_date'], 'days') else 0
    recency = np.exp(-days_inactive / 30.0)
    otw = 1.0 if row['open_to_work_flag'] else 0.3

    oar_raw = row['offer_acceptance_rate']
    offer_acceptance = 0.6 if oar_raw < 0 else oar_raw  # neutral-ish default when no offer history exists

    reliability = (
        0.4 * row['interview_completion_rate']
      + 0.3 * offer_acceptance
      + 0.3 * (1 - min(row['avg_response_time_hours'], 168.0) / 168.0)
    )

    market = min(row['saved_by_recruiters_30d'] / 10.0, 1.0)
    complete = row['profile_completeness_score'] / 100.0

    return (
        0.20 * recency
      + 0.20 * otw
      + 0.10 * notice_period_score(row.get('notice_period_days', 0))
      + 0.20 * reliability
      + 0.15 * market
      + 0.15 * complete
    )

def compute_scores(df: pd.DataFrame, jd: dict) -> pd.DataFrame:
    import datetime
    today = datetime.date.today()
    
    def process_row(row):
        jd_fit, matched_req = jd_fit_score(row, jd)
        qual = quality_score(row)
        plat = platform_score(row, today)
        
        final = (0.50 * jd_fit + 0.30 * qual + 0.20 * plat) * 100.0
        return jd_fit * 100.0, matched_req, [], qual * 100.0, plat * 100.0, final

    if len(df) == 0:
        df['jd_fit_score'] = []
        df['matched_required_skills'] = []
        df['matched_preferred_skills'] = []
        df['quality_score'] = []
        df['platform_score'] = []
        df['final_score'] = []
        return df

    results = df.apply(process_row, axis=1, result_type='expand')
    df['jd_fit_score'] = results[0]
    df['matched_required_skills'] = results[1]
    df['matched_preferred_skills'] = results[2]
    df['quality_score'] = results[3]
    df['platform_score'] = results[4]
    df['final_score'] = results[5]
    
    return df
