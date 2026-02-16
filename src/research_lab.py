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

def calculate_statistics(results_df):
    """
    Computes rigorous statistical metrics for the research paper.
    - T-Test (Paired) between RJAS and TF-IDF
    - Descriptive Stats (Mean, Std)
    """
    import numpy as np
    from scipy import stats
    
    if results_df.empty or "RJAS" not in results_df.columns or "NLP Score" not in results_df.columns:
        return {}
        
    rjas_scores = results_df["RJAS"].values
    # NLP Score is 0-1, scale to 0-100 for valid comparison
    nlp_scores = results_df["NLP Score"].values * 100 
    
    # 1. Descriptive
    stats_data = {
        "RJAS Mean": np.mean(rjas_scores),
        "RJAS Std": np.std(rjas_scores),
        "NLP Mean": np.mean(nlp_scores),
        "NLP Std": np.std(nlp_scores)
    }
    
    # 2. T-Test (Paired)
    # Null Hypothesis: The mean difference between RJAS and NLP Score is zero.
    try:
        t_stat, p_val = stats.ttest_rel(rjas_scores, nlp_scores)
        stats_data["T-Statistic"] = t_stat
        stats_data["P-Value"] = p_val
        stats_data["Significant"] = p_val < 0.05
    except Exception as e:
        stats_data["Error"] = str(e)
        
    return stats_data

def simulate_rl_convergence(iterations=100, role="Developer"):
    """
    Simulates the RL Agent's learning process over time to generate a convergence plot.
    Returns DataFrame with [Iteration, Reward, Weight_Skill, Weight_Semantic, ...].
    """
    import random
    from src.adaptive_engine import RLRankingAgent
    
    agent = RLRankingAgent()
    agent.current_role = role
    
    history = []
    
    # "Ideal" candidate profile for this simulation (Ground Truth)
    # Let's say a Developer role favors Skills (0.6) and Experience (0.3)
    # We simulate feedback where candidates high in these get hired (Reward=1)
    
    for i in range(iterations):
        # 1. Get current weights (Action)
        weights = agent.get_weights(explore=True)
        
        # 2. Simulate User Feedback
        # We generate a random candidate score profile
        # If the candidate aligns with "Ground Truth", user hires (Reward=1)
        # Otherwise, user rejects (Reward=-1 or 0) - Simplified: we only update on Hires for now
        
        # specific context: Developer
        # Candidate strengths
        cand_skills = random.uniform(0.5, 1.0)
        cand_exp = random.uniform(0.0, 1.0)
        cand_sem = random.uniform(0.4, 0.9)
        cand_edu = random.uniform(0.2, 0.8)
        
        # Oracle Score (Hidden user preference)
        oracle_score = (0.6 * cand_skills) + (0.3 * cand_exp) + (0.05 * cand_sem) + (0.05 * cand_edu)
        
        reward = 0
        if oracle_score > 0.75: # User Hires
            reward = 1.0
            # Construct feedback payload
            feedback = {
                "Breakdown": {
                    "Skills": cand_skills * 100,
                    "Experience": cand_exp * 100,
                    "Semantic": cand_sem * 100,
                    "Education": cand_edu * 100
                }
            }
            agent.update_policy(feedback, reward)
        
        # Log state
        log_entry = {"Iteration": i+1, "Cumulative Reward": agent.iterations} # approximation
        log_entry.update(agent.q_table[role])
        history.append(log_entry)
        
    return pd.DataFrame(history)
