import json
import os
import random

import numpy as np


class RLRankingAgent:
    """Contextual Bandit for adaptive resume ranking.

    The agent learns which scoring weights work best for each job role
    by observing recruiter hire/reject decisions over time.

    - State  : detected job role (Developer, Manager, Analyst, General)
    - Action : weight configuration (exploit best-known or explore neighbours)
    - Reward : +1 on hire, negative on reject (future work)
    """

    def __init__(self):
        # Q-table: one weight profile per role (the "arms" of the bandit)
        self.q_table = {
            "Developer": {"Semantic": 0.3, "Skills": 0.5, "Experience": 0.15, "Education": 0.05},
            "Manager":   {"Semantic": 0.3, "Skills": 0.2, "Experience": 0.4,  "Education": 0.1},
            "Analyst":   {"Semantic": 0.4, "Skills": 0.4, "Experience": 0.1,  "Education": 0.1},
            "General":   {"Semantic": 0.3, "Skills": 0.4, "Experience": 0.2,  "Education": 0.1},
        }
        self.current_role = "General"

        # Hyper-parameters
        self.epsilon = 0.1          # 10% chance to explore random weights
        self.learning_rate = 0.1    # step size (alpha)
        self.discount_factor = 0.9  # gamma — reserved for future MDP extension

        self.iterations = 0

    # ------------------------------------------------------------------
    # Context detection
    # ------------------------------------------------------------------

    def detect_role(self, jd_text):
        """Simple keyword-based role detection (provides the bandit 'context')."""
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

    # ------------------------------------------------------------------
    # Epsilon-greedy action selection
    # ------------------------------------------------------------------

    def get_weights(self, explore=True):
        """Returns scoring weights for the current role.

        With probability (1-epsilon): exploit best-known weights.
        With probability epsilon:     explore by perturbing one weight randomly.
        """
        weights = self.q_table.get(self.current_role, self.q_table["General"]).copy()

        if explore and random.random() < self.epsilon:
            # Pick a random dimension and nudge it
            keys = list(weights.keys())
            target_key = random.choice(keys)
            delta = random.uniform(-0.05, 0.05)
            weights[target_key] = max(0.05, min(0.8, weights[target_key] + delta))

            # Re-normalise so weights still sum to 1
            total = sum(weights.values())
            for k in weights:
                weights[k] /= total

        return weights

    # ------------------------------------------------------------------
    # Policy update (Q-value learning)
    # ------------------------------------------------------------------

    def update_policy(self, feedback_data, reward=1.0):
        """Shifts weight toward the strongest component of a hired candidate.

        Uses a simplified Q-learning update:
            Q(role, component) += learning_rate * reward
        then re-normalises the weight vector.
        """
        current_weights = self.q_table[self.current_role]
        scores = feedback_data.get("Breakdown", {})

        if not scores:
            return current_weights

        # Only consider actual weight dimensions (ignore derived fields like RJAS)
        valid_keys = [k for k in scores if k in current_weights]
        if not valid_keys:
            return current_weights

        # Find which component contributed most to the hire decision
        best_component = max(valid_keys, key=scores.get)

        # Q-update: increase weight for the best component
        old_value = current_weights.get(best_component, 0.25)
        current_weights[best_component] = old_value + (self.learning_rate * reward)

        # Re-normalise
        total = sum(current_weights.values())
        for k in current_weights:
            current_weights[k] /= total

        self.q_table[self.current_role] = current_weights
        self.iterations += 1

        return current_weights
