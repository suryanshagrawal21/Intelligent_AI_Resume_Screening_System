import numpy as np

def calculate_rjas(sbert_score, skill_score, experience_score, education_score, bias_penalty=0.0, weights=None, alpha=0.9):
    """
    Calculates the Multi-Objective Resume-Job Alignment Score (RJAS).
    
    Formula:
    Accuracy = (w_sem * SBERT) + (w_skill * Skill) + (w_exp * Exp) + (w_edu * Edu)
    Fairness = 1.0 - BiasPenalty
    RJAS = alpha * Accuracy + (1-alpha) * Fairness
    
    Args:
        ...
        alpha (float): Fairness Preference (0.0=Pure Fairness, 1.0=Pure Accuracy). Default 0.9.
        
    Returns:
        tuple: (CompositeScore, AccuracyScore, FairnessScore) - All 0-100
    """
    if weights is None:
         weights = {"Semantic": 0.25, "Skills": 0.25, "Experience": 0.25, "Education": 0.25}
         
    # Ensure keys match (fallback for 'Skill' vs 'Skills')
    w_sem = weights.get("Semantic", 0.25)
    w_skill = weights.get("Skills", weights.get("Skill", 0.25))
    w_exp = weights.get("Experience", 0.25)
    w_edu = weights.get("Education", 0.25)

    # 1. Accuracy Component (Predictive Power)
    accuracy_score = (
        sbert_score * w_sem +
        skill_score * w_skill +
        experience_score * w_exp +
        education_score * w_edu
    )
    
    # 2. Fairness Component (Bias Minimization)
    # If bias is detected (penalty=1.0), Fairness = 0.0. If clean, Fairness = 1.0.
    fairness_score = 1.0 - bias_penalty
    
    # 3. Multi-Objective Composite Score
    # alpha * Accuracy + (1-alpha) * Fairness
    # e.g., 0.8 * 0.9 (Good Candidate) + 0.2 * 1.0 (Clean) = 0.72 + 0.2 = 0.92
    
    final_score = (alpha * accuracy_score) + ((1.0 - alpha) * fairness_score)
    
    # Clip and Scale
    return (
        round(max(0.0, min(1.0, final_score)) * 100, 2),
        round(max(0.0, min(1.0, accuracy_score)) * 100, 2),
        round(max(0.0, min(1.0, fairness_score)) * 100, 2)
    )
