"""
feature_extractor.py — Phase 2 Component
Provides three deterministic scoring functions:
  score_skills(candidate_skills)  -> [0.0, 1.0]
  score_experience(years)         -> [0.0, 1.0]
  score_location(location_str)    -> constant tier value
"""

import math
import re
import logging
from typing import List, Dict, Union, Any

# Configure logging
log = logging.getLogger(__name__)

# 1. Define comprehensive SKILLS_TAXONOMY mapping variations to standardized buckets
SKILLS_TAXONOMY: Dict[str, str] = {
    # faiss
    "faiss": "faiss",
    "faiss-cpu": "faiss",
    "faiss-gpu": "faiss",
    "approximate nearest neighbor": "faiss",
    "ann search": "faiss",
    "ann": "faiss",
    "vector index": "faiss",
    "ann indexing": "faiss",
    
    # embeddings
    "embeddings": "embeddings",
    "embedding": "embeddings",
    "vector similarity search": "embeddings",
    "vector representation": "embeddings",
    "vector embeddings": "embeddings",
    "similarity search": "embeddings",
    
    # semantic search
    "semantic search": "semantic search",
    "semantic_search": "semantic search",
    "dense retrieval": "semantic search",
    "neural search": "semantic search",
    
    # sentence-transformers
    "sentence-transformers": "sentence-transformers",
    "sentence_transformers": "sentence-transformers",
    "sentence transformers": "sentence-transformers",
    "all-minilm-l6-v2": "sentence-transformers",
    
    # pytorch
    "pytorch": "pytorch",
    "torch": "pytorch",
    
    # python
    "python": "python",
    "py": "python",
    
    # nlp
    "nlp": "nlp",
    "natural language processing": "nlp",
    
    # rag
    "rag": "rag",
    "retrieval augmented generation": "rag",
    
    # fine-tuning
    "llm fine-tuning": "fine-tuning",
    "llm finetuning": "fine-tuning",
    "fine-tuning": "fine-tuning",
    "finetuning": "fine-tuning",
    "model tuning": "fine-tuning",
    
    # vector_db
    "qdrant": "vector_db",
    "pinecone": "vector_db",
    "weaviate": "vector_db",
    "vector db": "vector_db",
    "vector database": "vector_db",
    "milvus": "vector_db",
}

MUST_HAVE_SKILLS: List[str] = ['faiss', 'embeddings', 'semantic search', 'sentence-transformers']

PROFICIENCY_MAP: Dict[str, float] = {
    "expert": 1.0,
    "advanced": 0.8,
    "intermediate": 0.5,
    "beginner": 0.2,
}

def normalize_string(s: str) -> str:
    """Normalize input strings to lowercase and strip whitespace."""
    return re.sub(r"\s+", " ", str(s).strip().lower())

def score_skills(profile: dict) -> float:
    """
    Evidence-first skills scorer.
    Primary signal (75%): applied ML work evident in career descriptions.
    Secondary signal (25%): core skill name presence as tiebreaker only.
    """
    # Primary: scan career descriptions for applied ML evidence
    career_text = ' '.join(
        job.get('description', '') for job in profile.get('career_history', []) if isinstance(job, dict)
    ).lower()
    
    APPLIED_ML_SIGNALS = [
        'production', 'deployed', 'shipped', 'built', 'designed',
        'recommendation', 'search', 'retrieval', 'embedding', 'model',
        'inference', 'pipeline', 'trained', 'fine-tuned', 'served'
    ]
    career_signal_score = sum(
        1 for s in APPLIED_ML_SIGNALS if s in career_text
    ) / len(APPLIED_ML_SIGNALS)
    
    # Secondary: core skill names as tiebreaker only
    skills = profile.get('skills', [])
    if isinstance(skills, dict):
        skills = [{"name": k} for k in skills.keys()]
    
    CORE_SKILLS = {
        'python', 'pytorch', 'tensorflow', 'transformers', 'faiss',
        'embeddings', 'llm', 'nlp', 'ml', 'deep learning', 'rag',
        'langchain', 'huggingface', 'sentence-transformers', 'vector'
    }
    skill_names = {
        (s['name'].lower() if isinstance(s, dict) and 'name' in s else str(s).lower())
        for s in skills if isinstance(s, (dict, str))
    }
    keyword_score = len(skill_names & CORE_SKILLS) / len(CORE_SKILLS)
    
    # Career evidence outweighs keyword listing 3:1
    final = (career_signal_score * 0.75) + (keyword_score * 0.25)
    assert 0.0 <= final <= 1.0, f"skills score out of range: {final}"
    return float(final)

def score_experience(yoe: Any) -> float:
    """
    Gaussian bell curve model:
      score = math.exp(-0.5 * ((yoe - 7.0) / 2.5) ** 2)
    Boundary conditions:
      if yoe < 1 return 0.15; if yoe > 20 return 0.40.
    """
    if yoe is None:
        return 0.15
    
    try:
        y = float(yoe)
    except (ValueError, TypeError):
        return 0.15

    if y < 1.0:
        return 0.15
    if y > 20.0:
        return 0.40

    score = math.exp(-0.5 * ((y - 7.0) / 2.5) ** 2)
    
    # Asset check constraint assertion
    assert 0.0 <= score <= 1.0, f"Experience score {score} out of bounds [0.0, 1.0]"
    return score

def score_location(location: str) -> float:
    """
    Normalize location names and assign to Tier scoring:
      Pune/Noida = 1.0
      Bangalore/Mumbai/Hyderabad = 0.92
      Other India/Remote = 0.82
      International = 0.65
    """
    if not location:
        return 0.65

    loc_lower = normalize_string(location)
    # Normalize common variations
    loc_lower = loc_lower.replace("bengaluru", "bangalore")

    # Tier classification
    if "pune" in loc_lower or "noida" in loc_lower:
        score = 1.0
    elif "bangalore" in loc_lower or "mumbai" in loc_lower or "hyderabad" in loc_lower:
        score = 0.92
    else:
        # Check other India cities/states or remote keywords
        india_keywords = {
            "india", "delhi", "chennai", "kolkata", "ahmedabad", "jaipur",
            "lucknow", "bhopal", "indore", "surat", "chandigarh", "kochi",
            "coimbatore", "nagpur", "vadodara", "patna", "agra", "nashik",
            "meerut", "rajasthan", "gujarat", "maharashtra", "karnataka",
            "tamilnadu", "tamil nadu", "uttar pradesh", "west bengal",
            "andhra pradesh", "telangana", "kerala", "bihar", "odisha",
            "haryana", "punjab", "goa", "assam", "jharkhand", "chhattisgarh",
            "uttarakhand", "himachal", "manipur", "tripura", "meghalaya", "remote"
        }
        if any(kw in loc_lower for kw in india_keywords):
            score = 0.82
        else:
            score = 0.65

    # Asset check constraint assertion
    assert 0.0 <= score <= 1.0, f"Location score {score} out of bounds [0.0, 1.0]"
    return score
