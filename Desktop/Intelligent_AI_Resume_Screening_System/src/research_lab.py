import spacy
from collections import Counter
import pandas as pd
import plotly.express as px

nlp = spacy.load("en_core_web_sm")

def get_jaccard_similarity(set1, set2):
    """
    Calculates Jaccard Similarity (Intersection over Union).
    Used as the 'Traditional Keyword Matching' baseline.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def compare_algorithms(resume_text, jd_text, nlp_score, rjas_score=0):
    """
    Compares traditional Jaccard matching vs NLP Cosine matching vs RJAS.
    Returns the scores for visualization.
    """
    # Simple tokenization for Jaccard
    r_tokens = set(resume_text.lower().split())
    j_tokens = set(jd_text.lower().split())
    
    jaccard_score = get_jaccard_similarity(r_tokens, j_tokens)
    
    return {
        "Keyword (Jaccard)": round(jaccard_score, 4),
        "NLP (TF-IDF/Cosine)": round(nlp_score, 4),
        "Novel RJAS (Composite)": round(rjas_score / 100.0, 4) # Normalize RJAS to 0-1
    }

def detect_bias_entities(text):
    """
    Checks if Name/Gender/Age entities are still detectable.
    This demonstrates the component of the system that filters them out.
    """
    doc = nlp(text)
    biased_entities = []
    
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "DATE", "AGE", "GPE"]: # GPE location can sometimes bias
            biased_entities.append((ent.text, ent.label_))
            
    return biased_entities

def analyze_fairness(df):
    """
    Analyzes the 'Fairness' of the ranking by checking if detected bias entities 
    correlate with lower scores (which would be bad, but here we want to ensure 
    they don't correlate with higher scores arbitrarily).
    
    In a real system, we would check demographic parity. 
    Here, we calculate the average score of candidates WITH detected bias vs WITHOUT.
    """
    if df.empty or "Detected Bias" not in df.columns:
        return None
        
    df["Has_Bias"] = df["Detected Bias"].apply(lambda x: len(x) > 0)
    
    avg_score_bias = df[df["Has_Bias"]]["Final Score"].mean()
    avg_score_clean = df[~df["Has_Bias"]]["Final Score"].mean()
    
    return {
        "Avg Score (Bias Detected)": round(avg_score_bias, 2) if not pd.isna(avg_score_bias) else 0,
        "Avg Score (Clean)": round(avg_score_clean, 2) if not pd.isna(avg_score_clean) else 0,
        "Gap": round(abs(avg_score_bias - avg_score_clean), 2) if not pd.isna(avg_score_bias) and not pd.isna(avg_score_clean) else 0
    }

def generate_pareto_frontier(results_data, weights):
    """
    Simulates Ranking for varying Alpha (0.0 to 1.0) to visualize Accuracy vs Fairness trade-off.
    Returns DataFrame for Pareto Plot.
    """
    from src.rjas_metric import calculate_rjas
    
    pareto_points = []
    
    # Iterate alpha from 0.0 to 1.0
    for alpha in [i/10 for i in range(11)]:
        # Calculate scores for ALL candidates under this alpha
        total_accuracy = 0
        total_fairness = 0
        count = 0
        
        for res in results_data:
            if res.get("Processing Status") != "Success":
                 continue
                 
            # Re-calculate standard scores
            skill_ratio = len(res["Matched Skills"]) / 20 # Approximation or pass JD skills count
            # Use cached scores if avail, else approx
            acc_score = res["Breakdown"]["Accuracy"] # Pre-calculated at current alpha? No, need to recalculate.
            
            # Actually, calculate_rjas is stateless, so we can just re-run it
            rjas, acc, fair = calculate_rjas(
                sbert_score=res["SBERT Score"],
                skill_score=res["Breakdown"]["Skills"]/100.0,
                experience_score=res["Breakdown"]["Experience"]/100.0,
                education_score=res["Breakdown"]["Education"]/100.0,
                bias_penalty=res["Bias Penalty"],
                weights=weights,
                alpha=alpha
            )
            
            total_accuracy += acc
            total_fairness += fair
            count += 1
            
        if count > 0:
            pareto_points.append({
                "Alpha (Preference)": alpha,
                "Msg": f"α={alpha}",
                "Global Accuracy": round(total_accuracy / count, 2),
                "Global Fairness": round(total_fairness / count, 2)
            })
            
    return pd.DataFrame(pareto_points)
