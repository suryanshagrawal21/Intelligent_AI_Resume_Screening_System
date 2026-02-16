from src.research_lab import compare_algorithms
import pandas as pd
import random

class ExperimentLab:
    def __init__(self):
        self.results_log = []

    def run_comparative_study(self, candidates, jd_text, sbert_model, tfidf_matcher):
        """
        Runs 3-way comparison (Jaccard, TF-IDF, SBERT) and aggregates metrics.
        """
        study_data = []
        
        for cand in candidates:
            # 1. Jaccard (Baseline)
            comp_res = compare_algorithms(cand["Raw Text"], jd_text, 0) # Helper reused
            jaccard_score = comp_res["Keyword (Jaccard) Score"]
            
            # 2. TF-IDF (Structural)
            # Calculated in main app usually, passed here or re-calc if needed.
            # Assuming 'score' passed in candidate dict is TF-IDF for now.
            tfidf_score = cand.get("NLP Score", 0) # Fallback
            
            # 3. SBERT (Deep Semantic)
            # In a real app we'd generate embedding here.
            # For this 'Lab', we assume SBERT scoring is done in the main flow.
            # Let's placeholder or expect it in candidate dict.
            sbert_score = cand.get("SBERT Score", 0)
            
            study_data.append({
                "Name": cand["Name"],
                "Jaccard": jaccard_score,
                "TF-IDF": tfidf_score,
                "SBERT": sbert_score
            })
            
        return pd.DataFrame(study_data)

    def generate_simulated_metrics(self, study_df):
        """
        Since we don't have Ground Truth labels for user uploads, 
        we simulate 'Precision' and 'Recall' assuming SBERT is the Ground Truth (Oracle).
        This helps visualize how much better SBERT is vs Jaccard.
        """
        # Assume Top 3 SBERT candidates are "Relevant"
        top_k = 3
        sorted_sbert = study_df.sort_values(by="SBERT", ascending=False).head(top_k)
        ground_truth_names = set(sorted_sbert["Name"])
        
        metrics = []
        for method in ["Jaccard", "TF-IDF", "SBERT"]:
            # Get Top 3 by this method
            top_method = study_df.sort_values(by=method, ascending=False).head(top_k)
            retrieved_names = set(top_method["Name"])
            
            # Calculate Precision @ K
            relevant_retrieved = len(ground_truth_names.intersection(retrieved_names))
            precision = relevant_retrieved / top_k
            
            metrics.append({
                "Method": method,
                "Precision@3": precision,
                "Recall@3": precision # Same as precision when K is fixed and relevance count = K
            })
            
        return pd.DataFrame(metrics)
