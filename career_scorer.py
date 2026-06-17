"""
career_scorer.py — Phase 2 Component
Provides:
  score_career(career_history, current_title)  -> float in [0.0, 1.0]
  is_honeypot(profile)   -> bool (True = disqualify immediately)
"""

import re
import logging
from typing import List, Dict, Union, Any
from rapidfuzz import fuzz

# Configure logging
log = logging.getLogger(__name__)

# Pre-compiled lowercase sets of consulting firms and product companies
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "tata consultancy services",
    "infosys", "infosys technologies",
    "wipro", "wipro technologies",
    "accenture", "accenture india",
    "cognizant", "cognizant technology solutions", "cts",
    "capgemini", "capgemini india",
    "hcl", "hcl technologies", "hcltech",
    "tech mahindra", "techm",
    "l&t infotech", "lnt infotech", "lti", "ltimindtree",
    "mindtree", "mphasis", "hexaware", "hexaware technologies",
    "ust", "ust global", "persistent", "persistent systems",
    "coforge", "cyient", "virtusa", "dxc", "dxc technology",
    "ntt data", "genpact", "tcs e-serve", "capgemini consulting"
}

PRODUCT_COMPANIES = {
    "razorpay", "zepto", "cred", "meesho", "swiggy", "flipkart",
    "zomato", "paytm", "phonepe", "ola", "uber", "inmobi", "ola electric",
    "groww", "nykaa", "lenskart", "bms", "bookmyshow", "freshworks",
    "google", "microsoft", "amazon", "meta", "facebook", "apple", "netflix",
    "nvidia", "amd", "intel", "adobe", "salesforce", "atlassian", "oracle",
    "postman", "browserstack", "hasura", "chargebee", "uniphore", "druva"
}

POSITIVE_TITLE_KEYWORDS = {
    "machine learning", "ml", "artificial intelligence", "ai",
    "nlp", "natural language", "research engineer", "research scientist",
    "data scientist", "deep learning", "computer vision",
    "applied scientist", "applied ml", "applied ai",
    "embedding", "retrieval", "ranking",
}

NON_TECH_DOMAINS = {"hr", "marketing", "recruiter", "sales", "recruitment", "finance", "billing"}

AI_ML_SKILLS = {
    "faiss", "embeddings", "semantic search", "sentence-transformers",
    "pytorch", "torch", "nlp", "natural language processing", "rag",
    "retrieval augmented generation", "llm fine-tuning", "llm finetuning",
    "fine-tuning", "finetuning", "vector db", "vector database", "vector store",
    "qdrant", "pinecone", "weaviate", "deep learning", "machine learning", "ml",
    "artificial intelligence", "ai", "computer vision"
}

NON_ML_TITLES = {
    'marketing manager', 'hr manager', 'human resources', 'business analyst',
    'sales manager', 'product marketing', 'content manager', 'content writer',
    'operations manager', 'account manager', 'finance manager', 'brand manager',
    'social media manager', 'growth manager', 'customer success', 'recruiter',
    'talent acquisition', 'office manager', 'executive assistant', 'pr manager',
    'legal counsel', 'procurement manager', 'supply chain manager',
}

def normalize_string(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def score_career(career_history: Union[List[Dict[str, Any]], Dict[str, Any]], current_title: str = "") -> float:
    """
    Evaluate career quality signal.
    Accepts either a profile dictionary, or career_history list and current_title string.
    - Use rapidfuzz.fuzz.token_set_ratio to robustly cross-reference company names.
    - Multipliers: Entirely consulting = 0.5x penalty; mixed = 1.0; product-dominant = 1.2x bonus.
    - Additive +0.05 bonus if raw operational text ("shipped", "production", "deployed", "A/B test", "scale") found in description.
    - Returns normalized ultimate score in [0.0, 1.0].
    """
    # Overload support: if the first argument is a dictionary (profile), unpack it.
    if isinstance(career_history, dict):
        profile = career_history
        career_list = profile.get("career_history", []) or []
        curr_title = str(profile.get("profile", {}).get("current_title", "") or profile.get("current_title", "") or "")
    else:
        career_list = career_history or []
        curr_title = current_title

    career_history_list: List[Dict[str, Any]] = [job for job in career_list if isinstance(job, dict)]
    curr_title_norm = normalize_string(curr_title)

    # Hard floor: non-technical title overrides company scoring
    if any(t in curr_title_norm for t in NON_ML_TITLES):
        return 0.15

    # 1. Base Score calculation (Relevance of titles)
    base_score = 0.5
    has_relevant_title = False

    if any(kw in curr_title_norm for kw in POSITIVE_TITLE_KEYWORDS):
        base_score += 0.2
        has_relevant_title = True

    for job in career_history_list:
        title_norm = normalize_string(job.get("title", ""))
        if any(kw in title_norm for kw in POSITIVE_TITLE_KEYWORDS):
            base_score += 0.1
            has_relevant_title = True

    base_score = min(base_score, 1.0)

    # 2. Company Classification via rapidfuzz
    consulting_count = 0
    product_count = 0

    for job in career_history_list:
        company = normalize_string(job.get("company", "") or job.get("employer", ""))
        if not company:
            continue

        best_consulting = 0
        for firm in CONSULTING_FIRMS:
            sim = fuzz.token_set_ratio(firm, company)
            if sim > best_consulting:
                best_consulting = sim

        best_product = 0
        for prod in PRODUCT_COMPANIES:
            sim = fuzz.token_set_ratio(prod, company)
            if sim > best_product:
                best_product = sim

        if max(best_consulting, best_product) >= 85.0:
            if best_product > best_consulting:
                product_count += 1
            else:
                consulting_count += 1

    # 3. Apply Multiplier
    if consulting_count > 0 and product_count == 0:
        multiplier = 0.5
    elif product_count > 0 and consulting_count == 0:
        multiplier = 1.2
    elif product_count > 0 and consulting_count > 0:
        if product_count > consulting_count:
            multiplier = 1.2
        elif consulting_count > product_count:
            multiplier = 0.5
        else:
            multiplier = 1.0
    else:
        multiplier = 1.0

    score = base_score * multiplier

    # 4. Additive Operational Execution Bonus (+0.05)
    ops_keywords = ["shipped", "production", "deployed", "a/b test", "scale"]
    has_ops_text = False
    for job in career_history_list:
        desc = normalize_string(job.get("description", ""))
        if any(kw in desc for kw in ops_keywords):
            has_ops_text = True
            break

    if has_ops_text:
        score += 0.05

    # Normalize ultimate score
    score = max(0.0, min(1.0, score))

    # Asset check constraint assertion
    assert 0.0 <= score <= 1.0, f"Career score {score} out of bounds [0.0, 1.0]"
    return score

def is_honeypot(profile: dict) -> bool:
    """
    Return True if any honeypot signatures are detected:
    - Rule 1 (Skill/Duration Mismatch): expert in >= 3 separate skills, but every one shows duration < 3.
    - Rule 2 (Non-Technical Role Mismatch): current_title starts with non-tech token, yet >= 10 expert AI/ML skills.
    - Rule 3 (Missing Timeline Gap): (profile.years_of_experience * 12) - sum_career_months > 48.
    - Bonus Signals: total expert skills > 20 or empty career history with listed experience > 5 years.
    """
    if not profile or not isinstance(profile, dict):
        return False

    skills_raw = profile.get("skills", []) or []
    skills_list: List[Dict[str, Any]] = []

    if isinstance(skills_raw, dict):
        for k, v in skills_raw.items():
            if isinstance(v, dict):
                skills_list.append({"name": k, **v})
            else:
                skills_list.append({"name": k, "proficiency": str(v), "duration_months": None})
    elif isinstance(skills_raw, list):
        for item in skills_raw:
            if isinstance(item, dict):
                skills_list.append(item)

    career_history = profile.get("career_history", []) or []

    # --- Rule 1: Expert Trap ---
    expert_skills = [s for s in skills_list if normalize_string(s.get("proficiency", "")) == "expert"]
    if len(expert_skills) >= 3:
        all_short = True
        for s in expert_skills:
            dur_raw = s.get("duration_months")
            try:
                dur = float(dur_raw) if dur_raw is not None else 0.0
            except (ValueError, TypeError):
                dur = 0.0
            if dur >= 3.0:
                all_short = False
                break
        if all_short:
            log.warning("Honeypot flag: Rule 1 (Skill/Duration Mismatch)")
            return True

    # --- Rule 2: Non-Technical Role Mismatch ---
    AI_SKILL_KEYWORDS = {
        'faiss', 'llm', 'rag', 'langchain', 'pinecone', 'embeddings',
        'transformers', 'bert', 'gpt', 'pytorch', 'tensorflow', 'huggingface',
        'vector database', 'semantic search', 'fine-tuning', 'mlops',
        'kubeflow', 'ray', 'triton', 'vllm', 'lora', 'qlora'
    }
    curr_title = profile.get("profile", {}).get("current_title", "") or profile.get("current_title", "")
    curr_title_norm = normalize_string(curr_title)
    
    is_non_tech = any(t in curr_title_norm for t in NON_ML_TITLES) or any(d in curr_title_norm for d in NON_TECH_DOMAINS)
    
    expert_ai_count = 0
    for s in skills_list:
        name_norm = normalize_string(s.get("name", ""))
        prof_norm = normalize_string(s.get("proficiency", ""))
        if prof_norm == "expert" and any(ai_s in name_norm for ai_s in AI_SKILL_KEYWORDS):
            expert_ai_count += 1
            
    if is_non_tech and expert_ai_count >= 5:
        log.warning("Honeypot flag: Rule 2 (Non-Technical Role Mismatch, threshold 5)")
        return True

    # Bonus signal: non-tech title alone with ANY expert AI skills is suspicious
    if is_non_tech and expert_ai_count >= 2:
        profile['_soft_honeypot_flag'] = True

    # --- Rule 3: Missing Timeline Gap ---
    yoe_raw = profile.get("profile", {}).get("years_of_experience") or profile.get("years_of_experience")
    if yoe_raw is not None:
        try:
            yoe = float(yoe_raw)
        except (ValueError, TypeError):
            yoe = None

        if yoe is not None and career_history:
            sum_months = 0.0
            for job in career_history:
                if isinstance(job, dict):
                    dur_raw = job.get("duration_months")
                    try:
                        sum_months += float(dur_raw) if dur_raw is not None else 0.0
                    except (ValueError, TypeError):
                        pass
            if (yoe * 12.0) - sum_months > 48.0:
                log.warning("Honeypot flag: Rule 3 (Missing Timeline Gap)")
                return True

    # --- Bonus Signals ---
    # Expert count > 20
    expert_count = len(expert_skills)
    if expert_count > 20:
        log.warning("Honeypot flag: Bonus (Expert Count > 20)")
        return True

    # Empty career history while listed experience > 5 years
    if len(career_history) == 0 and yoe_raw is not None:
        try:
            yoe = float(yoe_raw)
            if yoe > 5.0:
                log.warning("Honeypot flag: Bonus (Empty Career History & yoe > 5)")
                return True
        except (ValueError, TypeError):
            pass

    return False
