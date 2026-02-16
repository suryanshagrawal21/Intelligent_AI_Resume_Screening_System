import numpy as np


def calculate_rjas(sbert_score, skill_score, experience_score, education_score,
                   bias_penalty=0.0, weights=None, alpha=0.9):
    """Computes the Multi-Objective Resume-Job Alignment Score (RJAS).

    RJAS balances two objectives:
      Accuracy  = weighted sum of individual component scores
      Fairness  = 1 - bias_penalty  (penalises detected PII / demographic info)
      RJAS      = alpha * Accuracy + (1-alpha) * Fairness

    Alpha controls the accuracy-fairness trade-off:
      alpha=1.0  -> pure accuracy (ignore bias)
      alpha=0.0  -> pure fairness (ignore skills/experience)

    Returns (composite_score, accuracy_score, fairness_score) each scaled 0-100.
    """
    # Fall back to equal weights if none provided
    if weights is None:
        weights = {"Semantic": 0.25, "Skills": 0.25, "Experience": 0.25, "Education": 0.25}

    # Unpack weights (handle both 'Skill' and 'Skills' key variants)
    w_sem = weights.get("Semantic", 0.25)
    w_skill = weights.get("Skills", weights.get("Skill", 0.25))
    w_exp = weights.get("Experience", 0.25)
    w_edu = weights.get("Education", 0.25)

    # --- Accuracy: weighted combination of all matching dimensions ---
    accuracy = (
        sbert_score * w_sem
        + skill_score * w_skill
        + experience_score * w_exp
        + education_score * w_edu
    )

    # --- Fairness: 0 if bias detected (penalty=1), 1 if clean ---
    fairness = 1.0 - bias_penalty

    # --- Composite: blend accuracy and fairness via alpha ---
    composite = (alpha * accuracy) + ((1.0 - alpha) * fairness)

    # Clip to [0, 1] then scale to a 0-100 percentage
    return (
        round(max(0.0, min(1.0, composite)) * 100, 2),
        round(max(0.0, min(1.0, accuracy)) * 100, 2),
        round(max(0.0, min(1.0, fairness)) * 100, 2),
    )
