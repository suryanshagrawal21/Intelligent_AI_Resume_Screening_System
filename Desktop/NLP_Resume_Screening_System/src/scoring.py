import textstat
import re

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

def calculate_ats_score(resume_text, jd_text, skill_match_count, total_jd_skills, nlp_similarity):
    """
    Calculates ATS Score based on weighted formula:
    - Skill matching (40%)
    - Keyword relevance (30%) - Mapped to NLP Cosine Similarity
    - Experience relevance (20%)
    - Education match (10%)
    
    Returns:
    - total_score (0-100)
    - breakdown (dict)
    """
    
    # 1. Skill Match (40%)
    skill_ratio = skill_match_count / total_jd_skills if total_jd_skills > 0 else 0
    score_skills = skill_ratio * 40
    
    # 2. Keyword Relevance (30%)
    # Using existing NLP similarity as proxy for "Relevance"
    score_keywords = nlp_similarity * 30
    
    # 3. Experience Relevance (20%)
    exp_factor = extract_experience_relevance(resume_text)
    score_experience = exp_factor * 20
    
    # 4. Education Match (10%)
    edu_factor = extract_education_level(resume_text)
    score_education = edu_factor * 10
    
    total_score = score_skills + score_keywords + score_experience + score_education
    
    return round(total_score, 2), {
        "Skills": round(score_skills, 2),
        "Keywords": round(score_keywords, 2),
        "Experience": round(score_experience, 2),
        "Education": round(score_education, 2)
    }
