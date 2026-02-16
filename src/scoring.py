import re

try:
    import textstat
except ImportError:
    textstat = None

from src.rjas_metric import calculate_rjas


# ---------------------------------------------------------------------------
# Sub-score Helpers
# ---------------------------------------------------------------------------

def calculate_readability(text):
    """Returns a 0-1 clarity score based on Flesch Reading Ease."""
    if not textstat:
        return 0.5  # sensible default when library is missing

    try:
        raw_score = textstat.flesch_reading_ease(text)
        # Flesch gives ~0-100; clamp then normalise
        return max(0, min(100, raw_score)) / 100.0
    except Exception:
        return 0.5


def extract_education_level(text):
    """Heuristic: scans for degree keywords and returns 0.0-1.0."""
    text_lower = text.lower()

    degrees = {
        "phd": 1.0, "ph.d": 1.0, "doctorate": 1.0,
        "master": 0.8, "m.sc": 0.8, "m.tech": 0.8, "mba": 0.8,
        "bachelor": 0.6, "b.sc": 0.6, "b.tech": 0.6, "b.e": 0.6, "bs": 0.6,
    }

    best = 0.0
    for keyword, score in degrees.items():
        if keyword in text_lower:
            best = max(best, score)

    return best  # 0 if no degree mentioned


def extract_experience_relevance(text):
    """Heuristic: looks for 'X years' patterns and scores 0.0-1.0."""
    matches = re.findall(r'(\d+)\+?\s*years?', text.lower())
    years = [int(y) for y in matches if y.isdigit()]

    max_years = max(years) if years else 0

    # Treat 5+ years as full score (mid-level baseline)
    return min(max_years / 5.0, 1.0)


# ---------------------------------------------------------------------------
# Composite Score
# ---------------------------------------------------------------------------

def calculate_composite_score(resume_text, jd_text, skill_match_count,
                              total_jd_skills, nlp_similarity,
                              sbert_similarity, weights,
                              bias_penalty=0, alpha=0.9):
    """Builds the multi-objective RJAS from all sub-scores.

    Returns (total_score 0-100, breakdown_dict).
    """
    # Individual component scores (all normalised to 0-1)
    skill_ratio = skill_match_count / total_jd_skills if total_jd_skills > 0 else 0
    experience = extract_experience_relevance(resume_text)
    education = extract_education_level(resume_text)
    semantic = sbert_similarity  # already 0-1 from SBERT cosine sim

    # Feed into the RJAS formula
    final_score, accuracy, fairness = calculate_rjas(
        sbert_score=semantic,
        skill_score=skill_ratio,
        experience_score=experience,
        education_score=education,
        bias_penalty=bias_penalty,
        weights=weights,
        alpha=alpha,
    )

    breakdown = {
        "Skills": round(skill_ratio * 100, 2),
        "Semantic": round(semantic * 100, 2),
        "Experience": round(experience * 100, 2),
        "Education": round(education * 100, 2),
        "RJAS": final_score,
        "Accuracy": accuracy,
        "Fairness": fairness,
    }

    return final_score, breakdown
