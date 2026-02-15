import json
import os
import random
import numpy as np

class RLRankingAgent:
    """
    Reinforcement Learning Agent (Contextual Bandit) for Adaptive Resume Ranking.
    
    State: Job Role (Developer, Manager, Analyst, General)
    Action: Weight Configuration (Exploitation) or Random Perturbation (Exploration)
    Reward: User Feedback (Hire/Reject)
    """
    def __init__(self):
        # Base Policies (The "Arms" of the bandit for each Context)
        self.q_table = {
            "Developer": {"Semantic": 0.3, "Skills": 0.5, "Experience": 0.15, "Education": 0.05},
            "Manager": {"Semantic": 0.3, "Skills": 0.2, "Experience": 0.4, "Education": 0.1},
            "Analyst": {"Semantic": 0.4, "Skills": 0.4, "Experience": 0.1, "Education": 0.1},
            "General": {"Semantic": 0.3, "Skills": 0.4, "Experience": 0.2, "Education": 0.1}
        }
        self.current_role = "General"
        self.epsilon = 0.1 # Exploration rate (10% chance to try random weights)
        self.learning_rate = 0.1 # Alpha
        self.discount_factor = 0.9 # Gamma (Not fully used in Bandit, but good for future)
        
        # Track iterations
        self.iterations = 0

    def detect_role(self, jd_text):
        """
        Simple keyword-based role detection (The 'Context' for the Bandit).
        """
        jd_lower = jd_text.lower()
        if any(w in jd_lower for w in ["manager", "lead", "director", "head", "chief"]):
            self.current_role = "Manager"
        elif any(w in jd_lower for w in ["developer", "engineer", "software", "programmer", "coder"]):
            self.current_role = "Developer"
        elif any(w in jd_lower for w in ["analyst", "data", "scientist", "research", "statistics"]):
            self.current_role = "Analyst"
        else:
            self.current_role = "General"
        return self.current_role

    def get_weights(self, explore=True):
        """
        Epsilon-Greedy Policy:
        - With probability 1-epsilon: Exploit (Return best known weights from Q-table)
        - With probability epsilon: Explore (Perturb weights slightly)
        """
        base_weights = self.q_table.get(self.current_role, self.q_table["General"]).copy()
        
        if explore and random.random() < self.epsilon:
            # Exploration: Randomly adjust weights by +/- 0.05
            keys = list(base_weights.keys())
            target = random.choice(keys)
            delta = random.uniform(-0.05, 0.05)
            base_weights[target] = max(0.05, min(0.8, base_weights[target] + delta))
            
            # Re-normalize
            total = sum(base_weights.values())
            for k in base_weights:
                base_weights[k] /= total
            
            return base_weights
        
        return base_weights

    def update_policy(self, feedback_data, reward=1.0):
        """
        Update Q-Values based on Reward.
        
        Args:
            feedback_data (dict): Data about the selected candidate (Strengths).
            reward (float): Positive for Hire, Negative for Reject (not implemented yet).
        """
        current_weights = self.q_table[self.current_role]
        
        # Analyze what made the candidate successful
        # feedback_data = {"Breakdown": {"Skills": 0.8, "Experience": 0.4 ...}}
        scores = feedback_data.get("Breakdown", {})
        
        if not scores:
            return current_weights

        # Find the dominant factor in the successful candidate
        best_component = max(scores, key=scores.get)
        
        # RL Update Rule: Q(s,a) = Q(s,a) + alpha * (Reward - Q(s,a))
        # Here we simplify: Shift weight towards the component that yielded the reward.
        
        old_value = current_weights.get(best_component, 0.25)
        new_value = old_value + (self.learning_rate * reward)
        
        # Update and Normalize
        current_weights[best_component] = new_value
        
        total = sum(current_weights.values())
        for k in current_weights:
            current_weights[k] /= total
            
        self.q_table[self.current_role] = current_weights
        self.iterations += 1
        
        return current_weights
