import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Cache the model globally so it's not reloaded for every call
_model = None

def embed_candidates(df: pd.DataFrame, jd: dict) -> pd.DataFrame:
    global _model
    
    # Load the precomputed embeddings
    embeddings = np.load("candidate_embeddings.npy")
    
    # Load candidate_id_order.pkl and assert alignment
    # Note: embed_candidates(df, jd) must only be called on a freshly-loaded candidates_parsed.pkl DataFrame 
    # (before any filtering or sorting). We compare lists to ignore pandas index differences.
    id_order_df = pd.read_pickle("candidate_id_order.pkl")
    if list(id_order_df['candidate_id']) != list(df['candidate_id']):
        raise ValueError("Misaligned candidate IDs between precomputed order and current dataframe.")
        
    # Load the model once
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        
    # Embed only the job description text
    jd_vec = _model.encode([jd['raw_jd_text']], normalize_embeddings=True, convert_to_numpy=True)[0]
    
    # Compute semantic scores via dot product
    semantic_scores = embeddings @ jd_vec
    
    # Assign and return
    df['semantic_score'] = semantic_scores
    return df
