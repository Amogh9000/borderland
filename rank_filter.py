import pandas as pd

def get_top_100(df: pd.DataFrame, jd: dict, top_k: int = 100) -> pd.DataFrame:
    # Hard filters
    # experience tolerance +-2yr
    exp_min = max(0, jd['min_years_experience'] - 1)
    exp_max = jd['max_years_experience'] + 2
    
    # filter
    df = df[(df['years_of_experience'] >= exp_min) & (df['years_of_experience'] <= exp_max)]
    
    # Explicit Honeypot Filters
    # 1. Experience mismatch: stated years vs sum of career duration > 4 years
    df = df[~((df['years_of_experience'] * 12 - df['sum_career_months']) > 48)]
    # 2. Too many expert skills with less than 6 months of use
    df = df[df['expert_skills_under_6m'] < 3]

    
    # sort
    df = df.sort_values(by=['final_score', 'candidate_id'], ascending=[False, True])
    
    # Company diversity cap: max 15 from same employer
    df['current_company_filled'] = df['current_company'].fillna("Unknown").replace("", "Unknown")
    df = df[df.groupby('current_company_filled').cumcount() < 15]
    top_df = df.head(top_k).copy()
    top_df = top_df.drop(columns=['current_company_filled'])
    
    if len(top_df) > 0:
        top_df['rank'] = range(1, len(top_df) + 1)
    
    return top_df
