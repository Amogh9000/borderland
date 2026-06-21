def generate_reasoning(row, jd_parsed):
    parts = []
    sim_pct = round(row.get('semantic_score', 0) * 100)
    parts.append(f"Profile text aligns {sim_pct}% with the JD.")
    
    matched = row.get('matched_required_skills', [])
    n_req   = len(jd_parsed.get('required_skills', []))
    if matched:
        named = ', '.join(matched[:4])
        parts.append(f"Matches {len(matched)} of {n_req} required skills: {named}.")
    else:
        parts.append("No required skills matched from the JD.")
        
    yoe = row.get('years_of_experience', 0)
    parts.append(f"{yoe} years exp as {row.get('current_title', '')} at {row.get('current_company', '')}.")
    
    scores = row.get('skill_assessment_scores', {})
    if isinstance(scores, dict) and scores:
        top_s, top_v = max(scores.items(), key=lambda x: x[1])
        parts.append(f"Top assessment: {top_s} scored {top_v}/100.")
        
    otw = "Open to work." if row.get('open_to_work_flag') else "Not currently open to work."
    parts.append(f"{otw} Notice period: {row.get('notice_period_days', 0)} days.")
    
    # Mandatory concerns block
    concerns = []
    github_score = row.get('github_activity_score', 100)
    if github_score >= 0 and github_score < 20:
        concerns.append(f"low GitHub activity ({github_score}/100)")
        
    if isinstance(scores, dict):
        low_scores = [f"{k}: {v}" for k,v in scores.items() if v < 40]
        if low_scores:
            concerns.append("assessment scores below 40: " + ', '.join(low_scores))
            
    notice = row.get('notice_period_days', 0)
    if notice > 30:
        concerns.append(f"notice period of {notice} days is above the sub-30-day preference")
        
    if not row.get('open_to_work_flag'):
        concerns.append("not marked as open to work")
        
    oar = row.get('offer_acceptance_rate', 1.0)
    if oar >= 0 and oar < 0.4:
        concerns.append("low offer acceptance rate")
        
    if concerns:
        parts.append("Concerns: " + '; '.join(concerns) + ".")
        
    return ' '.join(parts)
