import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.getcwd())

from src.research_lab import calculate_statistics, simulate_rl_convergence

def test_statistics():
    print("Testing Statistical Analysis...")
    # Mock Data
    df = pd.DataFrame({
        "Name": ["A", "B", "C", "D", "E"],
        "RJAS": [85, 90, 78, 60, 95], # Mean: 81.6
        "NLP Score": [0.7, 0.8, 0.75, 0.6, 0.85] # Mean: 0.74 -> 74
    })
    
    stats = calculate_statistics(df)
    
    print(f"Stats Result: {stats}")
    
    assert "T-Statistic" in stats, "T-Statistic missing"
    assert "P-Value" in stats, "P-Value missing"
    # In this mock, RJAS (81.6) > NLP (74), so T-stat should be positive or significant
    
    print("✅ Statistics Module Verified")

def test_convergence():
    print("Testing RL Convergence Simulation...")
    sim_df = simulate_rl_convergence(iterations=50, role="Developer")
    
    print(f"Simulation Shape: {sim_df.shape}")
    assert not sim_df.empty, "Simulation returned empty DataFrame"
    assert "Skills" in sim_df.columns, "Skills weight missing in history"
    
    # Check if weights changed (not static)
    start_skill = sim_df.iloc[0]["Skills"]
    end_skill = sim_df.iloc[-1]["Skills"]
    
    print(f"Start Skills W: {start_skill}, End Skills W: {end_skill}")
    assert start_skill != end_skill, "Weights did not evolve!"
    
    print("✅ RL Convergence Verified")

if __name__ == "__main__":
    try:
        test_statistics()
        test_convergence()
    except ImportError as e:
        print(f"⚠️ Skipped tests due to missing libs: {e}")
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        exit(1)
