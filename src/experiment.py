import random

import pandas as pd

from src.research_lab import compare_algorithms


class ExperimentLab:
    """Runs comparative experiments across matching algorithms."""

    def __init__(self):
        self.results_log = []

    def run_comparative_study(self, candidates, jd_text, sbert_model, tfidf_matcher):
        """Compares Jaccard, TF-IDF, and SBERT scores for each candidate."""
        study_data = []

        for cand in candidates:
            # Jaccard baseline (keyword overlap)
            comparison = compare_algorithms(cand["Raw Text"], jd_text, 0)
            jaccard_score = comparison["Keyword (Jaccard)"]

            # TF-IDF and SBERT scores are pre-computed in the main pipeline
            tfidf_score = cand.get("NLP Score", 0)
            sbert_score = cand.get("SBERT Score", 0)

            study_data.append({
                "Name": cand["Name"],
                "Jaccard": jaccard_score,
                "TF-IDF": tfidf_score,
                "SBERT": sbert_score,
            })

        return pd.DataFrame(study_data)

    def generate_simulated_metrics(self, study_df):
        """Estimates Precision@K and Recall@K using SBERT top-K as ground truth.

        Since we don't have labelled relevance data for user-uploaded resumes,
        we treat the top-3 SBERT candidates as the 'correct' set and measure
        how well Jaccard and TF-IDF recover them.
        """
        top_k = 3
        ground_truth = set(
            study_df.sort_values(by="SBERT", ascending=False).head(top_k)["Name"]
        )

        metrics = []
        for method in ["Jaccard", "TF-IDF", "SBERT"]:
            top_by_method = set(
                study_df.sort_values(by=method, ascending=False).head(top_k)["Name"]
            )
            overlap = len(ground_truth.intersection(top_by_method))
            precision = overlap / top_k

            metrics.append({
                "Method": method,
                "Precision@3": precision,
                "Recall@3": precision,  # equal when |relevant| == K
            })

        return pd.DataFrame(metrics)
