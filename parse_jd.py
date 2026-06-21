def parse_jd(jd_text: str) -> dict:
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
        "raw_jd_text": jd_text
    }
