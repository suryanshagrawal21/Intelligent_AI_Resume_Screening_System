import json
import os

class AdaptiveWeights:
    def __init__(self):
        # Default Weights (Standard)
        # Weights: [Semantic, Skill, Experience, Education]
        self.default_weights = {
            "Developer": {"Semantic": 0.3, "Skill": 0.5, "Experience": 0.15, "Education": 0.05},
            "Manager": {"Semantic": 0.3, "Skill": 0.2, "Experience": 0.4, "Education": 0.1},
            "Analyst": {"Semantic": 0.4, "Skill": 0.4, "Experience": 0.1, "Education": 0.1},
            "General": {"Semantic": 0.3, "Skill": 0.4, "Experience": 0.2, "Education": 0.1}
        }
        self.current_role = "General"
        self.current_weights = self.default_weights["General"].copy()
        self.learning_rate = 0.05 # How much weights shift on feedback

    def detect_role(self, jd_text):
        """
        Simple keyword-based role detection to initialize weights.
        """
        jd_lower = jd_text.lower()
        if any(w in jd_lower for w in ["manager", "lead", "director", "head"]):
            self.current_role = "Manager"
        elif any(w in jd_lower for w in ["developer", "engineer", "software", "programmer"]):
            self.current_role = "Developer"
        elif any(w in jd_lower for w in ["analyst", "data", "scientist", "research"]):
            self.current_role = "Analyst"
        else:
            self.current_role = "General"
            
        # Initialize with base weights for this role
        self.current_weights = self.default_weights[self.current_role].copy()
        return self.current_role

    def get_weights(self):
        return self.current_weights

    def update_weights_from_feedback(self, selected_candidate_data):
        """
        Reinforcement Learning-lite:
        If a user selects a candidate, we verify what was their strongest suit.
        If they had high Experience but low Skills, we boost Experience weight.
        """
        scores = selected_candidate_data["Breakdown"] # {Skills, Keywords, Experience, Education}
        # Normalize to 0-1 range roughly if they are out of 30, 40 etc.
        # Current Breakdown is raw points. We need relative strength.
        
        # Heuristic: Find the max scoring component
        best_component = max(scores, key=scores.get)
        
        # Boost the weight of the best component
        if best_component in self.current_weights:
            self.current_weights[best_component] += self.learning_rate
            
            # Normalize so sum is 1.0 (approximately)
            total = sum(self.current_weights.values())
            for k in self.current_weights:
                self.current_weights[k] /= total
                
        return self.current_weights
