import json
import pandas as pd
from dateutil import parser as date_parser
import datetime
from tqdm import tqdm

def parse_candidates(jsonl_path: str) -> pd.DataFrame:
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Parsing candidates", unit="cand"):
        if not line.strip(): continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        prof = c.get('profile', {})
        career = c.get('career_history', [])
        edu = c.get('education', [])
        skills = c.get('skills', [])
        signals = c.get('redrob_signals', {})
            
        # flatten skills
        skill_list = []
        if isinstance(skills, list):
            for s in skills:
                if isinstance(s, dict):
                    skill_list.append(s)
        elif isinstance(skills, dict):
            for k, v in skills.items():
                if isinstance(v, dict):
                    skill_list.append({"name": k, "proficiency": v.get("proficiency", "intermediate")})
                else:
                    skill_list.append({"name": k, "proficiency": str(v)})
        
        career_desc = " ".join([j.get("description", "") for j in career if isinstance(j, dict)])
        skills_desc = " ".join([s.get("name", "") for s in skill_list])
        career_text_blob = f"{prof.get('headline', '')} {prof.get('summary', '')} {career_desc} {skills_desc}"
        
        sum_career_months = sum([float(j.get("duration_months") or 0) for j in career if isinstance(j, dict)])
        expert_skills_under_6m = sum(1 for s in skill_list if str(s.get("proficiency", "")).lower() in ["expert", "advanced"] and float(s.get("duration_months") or 0) < 6)

        # Career history arrays — used by consulting-firm filter & tenure-velocity check
        career_companies = [j.get("company", "").lower().strip() for j in career if isinstance(j, dict)]
        career_titles    = [j.get("title",   "").lower().strip() for j in career if isinstance(j, dict)]
        n_unique_cos     = max(1, len(set(c for c in career_companies if c)))
        

        try:
            last_active_date = date_parser.parse(str(signals.get("last_active_date", "2026-06-19"))).date()
        except:
            last_active_date = datetime.date.today()
        
        exp_yoe = prof.get("years_of_experience")
        if exp_yoe is None:
            exp_yoe = c.get("years_of_experience", 0.0)
        
        try:
            exp_yoe = float(exp_yoe)
        except:
            exp_yoe = 0.0
            
        salary_min = c.get("expected_salary_min_lpa", 0)
        if salary_min is None: salary_min = 0
        salary_max = c.get("expected_salary_max_lpa", 100)
        if salary_max is None: salary_max = 100
        
        record = {
            "candidate_id": c.get("candidate_id") or c.get("id", "Unknown"),
            "years_of_experience": exp_yoe,
            "preferred_work_mode": prof.get("preferred_work_mode", "remote").lower(),
            "expected_salary_min_lpa": float(salary_min),
            "expected_salary_max_lpa": float(salary_max),
            "skills": skill_list,
            "skill_assessment_scores": c.get("skill_assessment_scores", {}),
            "education_tier": edu[0].get("tier", "tier_2") if edu and isinstance(edu[0], dict) else "tier_2",
            "github_activity_score": float(signals.get("github_activity_score") or 0),
            "endorsements_received": float(signals.get("endorsements_received") or 0),
            "last_active_date": last_active_date,
            "open_to_work_flag": bool(signals.get("open_to_work_flag", True)),
            "interview_completion_rate": float(signals.get("interview_completion_rate") or 0.8),
            "offer_acceptance_rate": float(signals.get("offer_acceptance_rate") or 0.8),
            "avg_response_time_hours": float(signals.get("avg_response_time_hours") or 24),
            "saved_by_recruiters_30d": float(signals.get("saved_by_recruiters_30d") or 0),
            "profile_completeness_score": float(signals.get("profile_completeness_score") or 50),
            "notice_period_days": int(signals.get("notice_period_days") or 30),
            "recruiter_response_rate": float(signals.get("recruiter_response_rate") or 0.5),
            "career_text_blob": career_text_blob,
            "current_title": prof.get("current_title", ""),
            "current_company": prof.get("current_company", ""),
            "sum_career_months": sum_career_months,
            "expert_skills_under_6m": expert_skills_under_6m,
            "career_companies": career_companies,   # list[str] — all employer names, lowercased
            "career_titles":    career_titles,      # list[str] — all role titles, lowercased
            "n_unique_companies": n_unique_cos,     # for tenure-velocity calc
        }
        records.append(record)
    return pd.DataFrame(records)
