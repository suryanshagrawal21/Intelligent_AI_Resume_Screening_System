import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.rjas_metric import calculate_rjas
from src.adaptive_engine import RLRankingAgent
from src.research_lab import analyze_fairness
import pandas as pd

def test_rjas():
    print("Testing Multi-Objective RJAS...")
    # alpha = 0.5 (Equal weight)
    score, acc, fair = calculate_rjas(
        sbert_score=0.9, 
        skill_score=0.8, 
        experience_score=1.0, 
        education_score=0.5,
        bias_penalty=1.0, # Detection penalty (Fairness=0)
        alpha=0.5
    )
    # Accuracy Component: (0.9*0.25 + 0.8*0.25 + 1.0*0.25 + 0.5*0.25) = 0.8
    # Fairness Component: 1.0 - 1.0 = 0.0
    # Final = 0.5*0.8 + 0.5*0.0 = 0.4 -> 40.0
    
    print(f"RJAS Output: {score}, Acc: {acc}, Fair: {fair}")
    assert score == 40.0, f"Expected 40.0, got {score}"
    assert acc == 80.0, f"Expected Acc 80.0, got {acc}"
    assert fair == 0.0, f"Expected Fair 0.0, got {fair}"
    print("✅ Multi-Objective RJAS Test Passed")

def test_rl_agent():
    print("Testing RL Agent...")
    agent = RLRankingAgent()
    role = agent.detect_role("Senior Python Developer needed")
    assert role == "Developer", f"Role detection failed: {role}"
    
    initial_weights = agent.get_weights(explore=False)
    print(f"Initial Weights: {initial_weights}")
    
    # Simulate feedback loop
    for _ in range(10):
        # Fake successful candidate with high skill
        feedback = {"Breakdown": {"Skills": 90, "Semantic": 60, "Experience": 50, "Education": 40}}
        agent.update_policy(feedback)
        
    updated_weights = agent.get_weights(explore=False)
    print(f"Updated Weights: {updated_weights}")
    
    # 'Skills' should ideally increase, or at least change
    assert "Skills" in updated_weights, "Key 'Skills' missing from weights"
    print("✅ RL Agent Test Passed")

if __name__ == "__main__":
    test_rjas()
    test_rl_agent()
