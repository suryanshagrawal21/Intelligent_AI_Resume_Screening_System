import random
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import spacy
from scipy import stats

nlp = spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------------
# Baseline Comparison
# ---------------------------------------------------------------------------

def get_jaccard_similarity(set1, set2):
    """Jaccard = |intersection| / |union|. Our 'traditional keyword' baseline."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def compare_algorithms(resume_text, jd_text, nlp_score, rjas_score=0):
    """Runs all three matchers and returns their scores side-by-side."""
    # Tokenise for the simple Jaccard baseline
    resume_tokens = set(resume_text.lower().split())
    jd_tokens = set(jd_text.lower().split())

    jaccard = get_jaccard_similarity(resume_tokens, jd_tokens)

    return {
        "Keyword (Jaccard)": round(jaccard, 4),
        "NLP (TF-IDF/Cosine)": round(nlp_score, 4),
        "Novel RJAS (Composite)": round(rjas_score / 100.0, 4),  # normalise RJAS to 0-1
    }


# ---------------------------------------------------------------------------
# Bias Detection & Fairness
# ---------------------------------------------------------------------------

def detect_bias_entities(text):
    """Flags potentially bias-inducing entities (name, age, location)."""
    doc = nlp(text)
    biased = []

    for ent in doc.ents:
        # PERSON / DATE / AGE → demographic info; GPE → location bias
        if ent.label_ in ["PERSON", "DATE", "AGE", "GPE"]:
            biased.append((ent.text, ent.label_))

    return biased


def analyze_fairness(df):
    """Compares average RJAS for candidates *with* vs *without* detected bias.

    A large gap would suggest the system is unfairly penalising certain groups.
    """
    if df.empty or "Detected Bias" not in df.columns:
        return None

    df["Has_Bias"] = df["Detected Bias"].apply(lambda x: len(x) > 0)

    avg_bias = df[df["Has_Bias"]]["Final Score"].mean()
    avg_clean = df[~df["Has_Bias"]]["Final Score"].mean()

    # Handle cases where one group might be empty
    avg_bias = round(avg_bias, 2) if not pd.isna(avg_bias) else 0
    avg_clean = round(avg_clean, 2) if not pd.isna(avg_clean) else 0

    gap = round(abs(avg_bias - avg_clean), 2) if (avg_bias and avg_clean) else 0

    return {
        "Avg Score (Bias Detected)": avg_bias,
        "Avg Score (Clean)": avg_clean,
        "Gap": gap,
    }


# ---------------------------------------------------------------------------
# Pareto Frontier (Accuracy vs Fairness trade-off)
# ---------------------------------------------------------------------------

def generate_pareto_frontier(results_data, weights):
    """Sweeps alpha from 0 to 1 and records global accuracy vs fairness.

    Returns a DataFrame suitable for a Pareto scatter plot.
    """
    from src.rjas_metric import calculate_rjas  # local import to avoid circular dep

    pareto_points = []

    for alpha in [i / 10 for i in range(11)]:
        total_accuracy = 0
        total_fairness = 0
        count = 0

        for res in results_data:
            if res.get("Processing Status") != "Success":
                continue

            rjas, acc, fair = calculate_rjas(
                sbert_score=res["SBERT Score"],
                skill_score=res["Breakdown"]["Skills"] / 100.0,
                experience_score=res["Breakdown"]["Experience"] / 100.0,
                education_score=res["Breakdown"]["Education"] / 100.0,
                bias_penalty=res["Bias Penalty"],
                weights=weights,
                alpha=alpha,
            )

            total_accuracy += acc
            total_fairness += fair
            count += 1

        if count > 0:
            pareto_points.append({
                "Alpha (Preference)": alpha,
                "Msg": f"α={alpha}",
                "Global Accuracy": round(total_accuracy / count, 2),
                "Global Fairness": round(total_fairness / count, 2),
            })

    return pd.DataFrame(pareto_points)


# ---------------------------------------------------------------------------
# Statistical Validation
# ---------------------------------------------------------------------------

def calculate_statistics(results_df):
    """Computes descriptive stats and a paired t-test (RJAS vs TF-IDF).

    Used to back the claim that RJAS significantly outperforms baseline NLP.
    """
    if results_df.empty or "RJAS" not in results_df.columns or "NLP Score" not in results_df.columns:
        return {}

    rjas_scores = results_df["RJAS"].values
    nlp_scores = results_df["NLP Score"].values * 100  # scale to 0-100 for fair comparison

    stats_data = {
        "RJAS Mean": np.mean(rjas_scores),
        "RJAS Std": np.std(rjas_scores),
        "NLP Mean": np.mean(nlp_scores),
        "NLP Std": np.std(nlp_scores),
    }

    # Paired t-test — H₀: mean(RJAS) = mean(NLP)
    try:
        t_stat, p_val = stats.ttest_rel(rjas_scores, nlp_scores)
        stats_data["T-Statistic"] = t_stat
        stats_data["P-Value"] = p_val
        stats_data["Significant"] = p_val < 0.05
    except Exception as e:
        stats_data["Error"] = str(e)

    return stats_data


# ---------------------------------------------------------------------------
# RL Convergence Simulation
# ---------------------------------------------------------------------------

def simulate_rl_convergence(iterations=100, role="Developer"):
    """Simulates the RL agent learning over many hire/reject cycles.

    Ground truth: a Developer role ideally favours Skills (0.6) and Experience (0.3).
    Candidates scoring above an oracle threshold get 'hired' (reward=1).
    Returns a DataFrame tracking weight evolution over iterations.
    """
    from src.adaptive_engine import RLRankingAgent

    agent = RLRankingAgent()
    agent.current_role = role
    history = []

    for i in range(iterations):
        # 1. Agent selects current weights (with exploration)
        weights = agent.get_weights(explore=True)

        # 2. Generate a random candidate profile
        cand_skills = random.uniform(0.5, 1.0)
        cand_exp = random.uniform(0.0, 1.0)
        cand_sem = random.uniform(0.4, 0.9)
        cand_edu = random.uniform(0.2, 0.8)

        # 3. Hidden oracle decides if this candidate is "good enough" to hire
        oracle_score = (0.6 * cand_skills) + (0.3 * cand_exp) + (0.05 * cand_sem) + (0.05 * cand_edu)

        if oracle_score > 0.75:
            # Simulate a positive hire signal
            feedback = {
                "Breakdown": {
                    "Skills": cand_skills * 100,
                    "Experience": cand_exp * 100,
                    "Semantic": cand_sem * 100,
                    "Education": cand_edu * 100,
                }
            }
            agent.update_policy(feedback, reward=1.0)

        # Log weights at this iteration
        log_entry = {"Iteration": i + 1, "Cumulative Reward": agent.iterations}
        log_entry.update(agent.q_table[role])
        history.append(log_entry)

    return pd.DataFrame(history)
