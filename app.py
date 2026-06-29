import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
from dateutil import parser as date_parser
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# App Configuration & JD
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Redrob AI - Candidate Ranking Demo", layout="wide")

def get_parsed_jd() -> dict:
    return {
        "required_skills": [
            "embeddings", "sentence-transformers", "vector databases", "faiss", 
            "pinecone", "qdrant", "weaviate", "milvus", "evaluation frameworks", 
            "ndcg", "python"
        ],
        "preferred_skills": ["lora", "peft", "xgboost", "distributed systems"],
        "min_years_experience": 5,
        "max_years_experience": 9,
        "work_mode": "hybrid",
        "salary_range_lpa": (25, 55),
        "raw_jd_text": "We are looking for a Senior ML Engineer focusing on search and retrieval. Required skills: embeddings, sentence-transformers, vector databases, faiss, pinecone, qdrant, weaviate, milvus, evaluation frameworks, ndcg, python. Preferred: lora, peft, xgboost, distributed systems. 5-9 years of experience, hybrid work."
    }

JD_PARSED = get_parsed_jd()

from score import compute_scores
from rank_filter import get_top_100
from reasoning import generate_reasoning
# -----------------------------------------------------------------------------
# Data Parsing & Pipeline
# -----------------------------------------------------------------------------
@st.cache_data
def parse_candidates_from_text(lines: list) -> pd.DataFrame:
    records = []
    for line in lines:
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
            "career_companies": career_companies,
            "career_titles": career_titles,
            "n_unique_companies": n_unique_cos,
        }
        records.append(record)
    return pd.DataFrame(records)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("Redrob AI — Candidate Ranking Demo")
st.markdown("This sandbox runs the complete Redrob candidate ranking pipeline offline. Upload a candidate JSON or JSONL file to generate a prioritized hiring queue based on semantic JD matching, experience gating, and behavioral signal evaluation.")

with st.sidebar:
    st.header("Job Description Requirements")
    st.markdown("**Required Skills**")
    st.write(", ".join(JD_PARSED['required_skills']))
    st.markdown("**Preferred Skills**")
    st.write(", ".join(JD_PARSED['preferred_skills']))
    st.markdown(f"**Experience Required:** {JD_PARSED['min_years_experience']} - {JD_PARSED['max_years_experience']} years")
    st.markdown(f"**Work Mode:** {JD_PARSED['work_mode'].capitalize()}")
    st.markdown(f"**Salary Range (LPA):** {JD_PARSED['salary_range_lpa'][0]} - {JD_PARSED['salary_range_lpa'][1]}")

uploaded_file = st.file_uploader("Upload candidates file (.json or .jsonl)", type=["json", "jsonl"])

if uploaded_file is not None:
    # 1. Parse file content
    content = uploaded_file.getvalue().decode("utf-8").strip()
    
    is_json_array = False
    lines = []
    try:
        parsed_json = json.loads(content)
        if isinstance(parsed_json, list):
            lines = [json.dumps(c) for c in parsed_json]
            is_json_array = True
    except json.JSONDecodeError:
        pass
        
    if not is_json_array:
        lines = content.split('\n')
        
    if not lines or not lines[0].strip():
        st.error("Uploaded file has 0 candidates.")
        st.stop()
        
    # Process all uploaded candidates
    pass
        
    try:
        df = parse_candidates_from_text(lines)
    except Exception as e:
        st.error(f"Error parsing candidates: {str(e)}")
        st.stop()
        
    if df.empty:
        st.error("Uploaded file yielded 0 valid candidates.")
        st.stop()
        
    st.info(f"Successfully loaded {len(df)} candidates. Running pipeline...")
    
    progress_bar = st.progress(0, text="Initializing Model...")
    
    # 2. Embedding
    model = load_model()
    progress_bar.progress(20, text="Embedding Job Description...")
    
    jd_emb = model.encode(JD_PARSED['raw_jd_text'], normalize_embeddings=True)
    
    progress_bar.progress(40, text="Embedding Candidates...")
    cand_embs = model.encode(df['career_text_blob'].tolist(), normalize_embeddings=True)
    
    progress_bar.progress(70, text="Scoring Candidates...")
    # Dot product for semantic similarity
    df['semantic_score'] = np.dot(cand_embs, jd_emb)
    
    # Run full scoring
    df = compute_scores(df, JD_PARSED)
    
    progress_bar.progress(90, text="Ranking and Filtering...")
    top_df = get_top_100(df, JD_PARSED, top_k=min(100, len(df)))
    
    progress_bar.progress(100, text="Generating Reasoning...")
    top_df['reasoning'] = top_df.apply(lambda r: generate_reasoning(r, JD_PARSED), axis=1)
    
    # Done
    st.success("Ranking pipeline completed.")
    
    # 3. Display Results
    st.subheader("Ranked Results")
    
    # Prepare display dataframe
    display_df = top_df[['rank', 'candidate_id', 'final_score', 'current_title', 'current_company', 'years_of_experience', 'notice_period_days', 'reasoning']].copy()
    display_df = display_df.rename(columns={
        'rank': 'Rank',
        'candidate_id': 'Candidate ID',
        'final_score': 'Score',
        'current_title': 'Title',
        'current_company': 'Company',
        'years_of_experience': 'Years of Experience',
        'notice_period_days': 'Notice Period',
        'reasoning': 'Reasoning'
    })
    
    st.dataframe(display_df, use_container_width=True)
    
    # 4. Score Breakdown
    st.subheader("Score Breakdown")
    breakdown_df = top_df[['candidate_id', 'jd_fit_score', 'quality_score', 'platform_score', 'final_score']].copy()
    st.dataframe(breakdown_df, use_container_width=True)
    
    # 5. CSV Download
    csv_out = top_df[['candidate_id', 'rank', 'final_score', 'reasoning']].copy()
    csv_out = csv_out.rename(columns={'final_score': 'score'})
    csv_data = csv_out.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download results as CSV",
        data=csv_data,
        file_name='redrob_ranked_output.csv',
        mime='text/csv',
    )
