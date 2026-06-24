import pandas as pd
import numpy as np
import os
import torch
from parse_candidates import parse_candidates
from sentence_transformers import SentenceTransformer

def main():
    # ── Device detection ─────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))

    # ── Parse candidates ─────────────────────────────────────────────────────
    print("Parsing candidates...")
    df = parse_candidates("candidates.jsonl")
    print(f"Parsed {len(df)} candidates.")

    # ── Embed with GPU ───────────────────────────────────────────────────────
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    corpus = df['career_text_blob'].tolist()

    # Larger batch size on GPU (A100/T4 can handle 2048+ comfortably)
    batch_size = 2048 if device == 'cuda' else 512

    print(f"Embedding {len(corpus)} candidates (batch_size={batch_size})...")
    embeddings = model.encode(
        corpus,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,
    )

    embeddings_np = np.array(embeddings, dtype='float32')

    # ── Build FAISS index ────────────────────────────────────────────────────
    print("Building FAISS index...")
    import faiss

    d = embeddings_np.shape[1]  # 384 for all-MiniLM-L6-v2

    # Use GPU FAISS if available and faiss-gpu is installed
    use_gpu_faiss = False
    index = None
    if device == 'cuda':
        try:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.GpuIndexFlatIP(res, d)
            gpu_index.add(embeddings_np)
            # Transfer back to CPU for serialisation (faiss.write_index requires CPU index)
            index = faiss.index_gpu_to_cpu(gpu_index)
            use_gpu_faiss = True
            print("FAISS GPU index built successfully.")
        except Exception as e:
            print(f"faiss-gpu not available ({e}), falling back to CPU index.")

    if not use_gpu_faiss:
        index = faiss.IndexFlatIP(d)
        index.add(embeddings_np)
        print("FAISS CPU index built.")

    # ── Save artifacts ───────────────────────────────────────────────────────
    print("Saving artifacts...")
    np.save("candidate_embeddings.npy", embeddings_np)
    df[['candidate_id']].to_pickle("candidate_id_order.pkl")
    assert index is not None
    faiss.write_index(index, "candidates.index")

    # Drop the large text blob before pickling the parsed DataFrame
    df.drop(columns=['career_text_blob'], inplace=True)
    df.to_pickle("candidates_parsed.pkl")

    print("Precomputation complete.")
    print(f"  embeddings : candidate_embeddings.npy  ({embeddings_np.nbytes // 1_000_000} MB)")
    print(f"  index      : candidates.index")
    print(f"  parsed df  : candidates_parsed.pkl")

if __name__ == "__main__":
    main()
