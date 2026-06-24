import pandas as pd

# Hard bracket limits (soft penalties are processed upstream in score.py)
HARD_YOE_MIN = 4.0
HARD_YOE_MAX = 11.0

def get_top_100(df: pd.DataFrame, jd: dict, top_k: int = 100) -> pd.DataFrame:
    # ── Hard YoE Gate ─────────────────────────────────────────────────────
    df = df[
        (df['years_of_experience'] >= HARD_YOE_MIN) &
        (df['years_of_experience'] <= HARD_YOE_MAX)
    ]

    # ── Honeypot Filters ──────────────────────────────────────────────────
    # 1. Inflated YoE vs verifiable career duration
    df = df[~((df['years_of_experience'] * 12 - df['sum_career_months']) > 48)]
    # 2. Too many expert/advanced skills with < 6 months actual usage
    df = df[df['expert_skills_under_6m'] < 3]

    # ── Notice Period Hard Floor ──────────────────────────────────────────
    # Notice over 90 days acts as a structural violation for founding timelines
    df = df[df['notice_period_days'] <= 90]

    # ── Hard-Eject Title-Decoupled Profiles ──────────────────────────────
    if 'title_tier' in df.columns:
        df = df[df['title_tier'] != 'decoupled']

    # ── Hard-Eject Purely Consulting Careers ──────────────────────────────
    if 'score_flags' in df.columns:
        df = df[~df['score_flags'].str.contains('pure_services', na=False)]

    # ── Sort by final_score desc, candidate_id asc ────────────────────────
    df = df.sort_values(
        by=['final_score', 'candidate_id'],
        ascending=[False, True],
    )

    # ── Company Diversity Cap: max 10 from the same employer ──────────────
    df['current_company_filled'] = (
        df['current_company'].fillna('Unknown').replace('', 'Unknown')
    )
    df = df[df.groupby('current_company_filled').cumcount() < 10]

    top_df = df.head(top_k).copy()
    top_df = top_df.drop(columns=['current_company_filled'])

    if len(top_df) > 0:
        top_df['rank'] = range(1, len(top_df) + 1)

    return top_df