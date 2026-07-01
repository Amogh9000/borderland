import argparse, json, datetime
import pandas as pd
from parse_candidates import parse_candidates
from parse_jd       import parse_jd
from embed          import embed_candidates
from score          import compute_scores
from rank_filter    import get_top_100
from reasoning      import generate_reasoning

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidates', required=True)
    p.add_argument('--jd',         required=True)
    p.add_argument('--output',     default='Team Borderland survivors.csv')
    p.add_argument('--top-k',      type=int, default=100)
    args = p.parse_args()
    
    print('[1/4] Parsing JD and Loading Precomputed Candidates...')
    try:
        df = pd.read_pickle("candidates_parsed.pkl")
    except FileNotFoundError:
        print("Error: candidates_parsed.pkl not found. Please run precompute.py first.")
        return

    with open(args.jd, 'r', encoding='utf-8') as f:
        jd_text = f.read()
    jd     = parse_jd(jd_text)
    
    print('[2/4] Embedding JD and Computing Similarities...')
    df     = embed_candidates(df, jd)
    
    print('[3/4] Scoring...')
    df     = compute_scores(df, jd)
    
    print('[4/4] Ranking...')
    top    = get_top_100(df, jd, args.top_k)
    
    if len(top) > 0:
        top['reasoning'] = top.apply(
            lambda r: generate_reasoning(r, jd), axis=1
        )
        submission_df = top[['candidate_id', 'rank', 'final_score', 'reasoning']].copy()
        submission_df.rename(columns={'final_score': 'score'}, inplace=True)
        submission_df.to_csv(args.output, index=False)
        json_out = args.output.replace('.csv', '.json')
        submission_df.to_json(json_out, orient='records', indent=2)
        excel_out = args.output.replace('.csv', '.xlsx')
        submission_df.to_excel(excel_out, index=False)
        print(f'Done. Results written to {args.output}, {json_out}, and {excel_out}')
    else:
        print("No candidates matched the filters.")

if __name__ == '__main__':
    main()
