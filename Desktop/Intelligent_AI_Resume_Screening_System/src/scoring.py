from src.rjas_metric import calculate_rjas

def calculate_readability(text):
    """
    Calculates a 'Clarity Score' using Flesch Reading Ease.
    Higher is easier to read (0-100).
    """
    try:
        score = textstat.flesch_reading_ease(text)
        # Normalize to 0-1 range roughly
        return max(0, min(100, score)) / 100.0
    except:
        return 0.5

def extract_education_level(text):
    """
    Heuristic to detect education level.
    Returns score 0.0 to 1.0.
    """
    text_lower = text.lower()
    degrees = {
        "phd": 1.0, "ph.d": 1.0, "doctorate": 1.0,
        "master": 0.8, "m.sc": 0.8, "m.tech": 0.8, "mba": 0.8,
        "bachelor": 0.6, "b.sc": 0.6, "b.tech": 0.6, "b.e": 0.6, "bs": 0.6
    }
    
    max_score = 0.0
    for deg, score in degrees.items():
        if deg in text_lower:
            max_score = max(max_score, score)
            
    return max_score # If no degree found, 0 (severe penalty in real ATS) or fallback.

def extract_experience_relevance(text):
    """
    Heuristic to estimate experience relevance.
    Simple regex for 'X years' or '20xx - 20xx'.
    """
    # Look for "X years" pattern
    years_pattern = re.findall(r'(\d+)\+?\s*years?', text.lower())
    years = [int(y) for y in years_pattern if y.isdigit()]
    
    max_years = max(years) if years else 0
    
    # Cap experience at 10 years for scoring 1.0
    return min(max_years / 5.0, 1.0) # Assume 5 years is "full score" baseline for general mid-level

def calculate_composite_score(resume_text, jd_text, skill_match_count, total_jd_skills, nlp_similarity, sbert_similarity, weights, bias_penalty=0, alpha=0.9):
    """
    Calculates the Final Composite Score using Multi-Objective RJAS.
    
    Returns:
    - total_score (0-100)
    - breakdown (dict)
    """
    
    # 1. Skill Match Ratio (0-1)
    skill_ratio = skill_match_count / total_jd_skills if total_jd_skills > 0 else 0
    
    # 2. Experience (0-1)
    exp_score = extract_experience_relevance(resume_text)
    
    # 3. Education (0-1)
    edu_score = extract_education_level(resume_text)
    
    # 4. Semantic (0-1) - already averaged/passed in
    sem_score = sbert_similarity
    
    # Calculate RJAS
    final_score, accuracy_score, fairness_score = calculate_rjas(
        sbert_score=sem_score,
        skill_score=skill_ratio,
        experience_score=exp_score,
        education_score=edu_score,
        bias_penalty=bias_penalty,
        weights=weights,
        alpha=alpha
    )
    
    return final_score, {
        "Skills": round(skill_ratio * 100, 2),
        "Semantic": round(sem_score * 100, 2),
        "Experience": round(exp_score * 100, 2),
        "Education": round(edu_score * 100, 2),
        "RJAS": final_score,
        "Accuracy": accuracy_score,
        "Fairness": fairness_score
    }
