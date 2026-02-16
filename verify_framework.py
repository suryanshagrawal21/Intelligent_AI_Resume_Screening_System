"""Quick verification that RJAS and the RL Agent work correctly."""

import sys
import os

sys.path.append(os.getcwd())

import pandas as pd
from src.rjas_metric import calculate_rjas
from src.adaptive_engine import RLRankingAgent
from src.research_lab import analyze_fairness


def test_rjas():
    """Checks the multi-objective RJAS formula with known inputs."""
    print("Testing Multi-Objective RJAS...")

    # alpha=0.5 means equal weight to accuracy and fairness
    score, acc, fair = calculate_rjas(
        sbert_score=0.9,
        skill_score=0.8,
        experience_score=1.0,
        education_score=0.5,
        bias_penalty=1.0,   # full penalty → Fairness = 0
        alpha=0.5,
    )

    # Expected:
    #   Accuracy = (0.9+0.8+1.0+0.5) * 0.25 = 0.8  → 80.0
    #   Fairness = 1.0 - 1.0 = 0.0                  → 0.0
    #   Final    = 0.5*0.8 + 0.5*0.0 = 0.4           → 40.0
    print(f"RJAS Output: {score}, Acc: {acc}, Fair: {fair}")
    assert score == 40.0, f"Expected 40.0, got {score}"
    assert acc == 80.0, f"Expected Acc 80.0, got {acc}"
    assert fair == 0.0, f"Expected Fair 0.0, got {fair}"
    print("✅ Multi-Objective RJAS Test Passed")


def test_rl_agent():
    """Verifies role detection and weight evolution after feedback."""
    print("Testing RL Agent...")

    agent = RLRankingAgent()
    role = agent.detect_role("Senior Python Developer needed")
    assert role == "Developer", f"Role detection failed: {role}"

    initial_weights = agent.get_weights(explore=False)
    print(f"Initial Weights: {initial_weights}")

    # Simulate 10 rounds of hiring a high-skill candidate
    for _ in range(10):
        feedback = {"Breakdown": {"Skills": 90, "Semantic": 60, "Experience": 50, "Education": 40}}
        agent.update_policy(feedback)

    updated_weights = agent.get_weights(explore=False)
    print(f"Updated Weights: {updated_weights}")

    assert "Skills" in updated_weights, "Key 'Skills' missing from weights"
    print("✅ RL Agent Test Passed")


if __name__ == "__main__":
    test_rjas()
    test_rl_agent()
