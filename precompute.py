import pandas as pd
import numpy as np
from parse_candidates import parse_candidates
from sentence_transformers import SentenceTransformer
import os

def main():
    print("Parsing candidates...")
    df = parse_candidates("candidates.jsonl")
    
    print(f"Running precomputation on all {len(df)} candidates...")
    
    print("Embedding candidates...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    corpus = df['career_text_blob'].tolist()
    
    embeddings = model.encode(
        corpus,
        batch_size=512,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print("Saving artifacts...")
    embeddings_np = np.array(embeddings).astype('float32')
    
    np.save("candidate_embeddings.npy", embeddings_np)
    df[['candidate_id']].to_pickle("candidate_id_order.pkl")
    
    import faiss
    d = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner Product = Cosine Similarity since vectors are normalized
    index.add(embeddings_np)
    faiss.write_index(index, "candidates.index")
    
    df.drop(columns=['career_text_blob'], inplace=True)
    df.to_pickle("candidates_parsed.pkl")
    print("Precomputation complete.")

if __name__ == "__main__":
    main()
