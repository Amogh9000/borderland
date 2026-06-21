# Redrob AI Candidate Ranking Pipeline

An enterprise-grade, two-phase candidate ranking system built for the **Redrob AI Hiring Challenge**. This pipeline processes 100,000 candidate profiles using semantic search and multi-factor scoring to output the top 100 candidates in under 5 minutes on CPU-only hardware.

## 🚀 Architecture Overview

The system is split into two phases to adhere to strict latency and hardware constraints (<16GB RAM, <5 minutes runtime):

1. **Phase 1: Offline Precompute (`precompute.py`)**
   - Parses the raw 100k `candidates.jsonl` dataset.
   - Generates 384-dimensional semantic embeddings using `all-MiniLM-L6-v2`.
   - Embeds **career history and descriptions** (explicitly excluding keyword-stuffed skills lists).
   - Builds an optimized `FAISS IndexFlatIP` binary index.

2. **Phase 2: Online Ranking (`rank.py`)**
   - Loads the pre-built FAISS index and performs an ultra-fast nearest-neighbor search for the top 500 semantic matches.
   - Executes the **Multi-Factor Scoring Engine**.
   - Filters out fraudulent profiles using the **Honeypot Detector**.
   - Generates dynamic, 200-character reasoning strings based on actual career evidence.
   - Exports the `top100_candidates.csv`.

## 🧠 Scoring Logic (Amended Playbook §1)

The final score is a weighted combination of multiple signals, heavily prioritizing actual career evidence over easily-faked skills lists:

- **Semantic Match (35%)**: JD alignment based on career history embeddings.
- **Career Score (30%)**: Evaluates employer tier (e.g., product vs. consulting) and role relevance.
- **Experience (15%)**: Gaussian-weighted scoring curve peaking around 5–8 years.
- **Location (10%)**: Tier-based geographic proximity scoring.
- **Skills (10%)**: Tiebreaker signal only.
- **Behavioral Multiplier**: A multiplicative layer (0.50x to 1.10x) based on GitHub activity, login recency, notice periods, and recruiter response rates.

## 🛡️ Anti-Fraud Honeypot Detection

The pipeline includes a robust set of countermeasures to disqualify "imposter" profiles:
1. **Rule 1 (Duration Fake)**: Disqualifies candidates claiming "expert" status in AI/ML skills with `< 3 months` of actual usage.
2. **Rule 2 (Non-Technical Mismatch)**: Disqualifies candidates with titles like "Marketing Manager" or "HR" who claim 5+ expert AI skills. Applies a soft-penalty (0.70x) for border cases.
3. **Rule 3 (Timeline Gap)**: Flags candidates whose stated Years of Experience (YOE) drastically exceeds their verifiable career history timeline.

## ⚙️ How to Run

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/Scripts/activate  # (Windows)
pip install -r requirements.txt
```

### 2. Build the FAISS Index (Phase 1)
*Ensure `candidates.jsonl` is in the project root.*
```bash
python precompute.py
```
*This generates `candidate_embeddings.npy`, `candidates.index`, `candidate_id_order.pkl`, and `candidates_parsed.pkl`.*

### 3. Execute the Ranking Pipeline (Phase 2)
```bash
python rank.py --candidates candidates.jsonl --jd job_description.txt
```
*This generates the final output at `top100.csv` and `top100.json`.*

## 📋 Dependencies
- `sentence-transformers`
- `faiss-cpu`
- `rapidfuzz`
- `numpy`
- `torch` (CPU only)
