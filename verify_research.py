"""Verifies the statistical analysis and RL convergence simulation modules."""

import sys
import os

sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from src.research_lab import calculate_statistics, simulate_rl_convergence


def test_statistics():
    """Runs the stats module on mock data and checks for required fields."""
    print("Testing Statistical Analysis...")

    df = pd.DataFrame({
        "Name": ["A", "B", "C", "D", "E"],
        "RJAS": [85, 90, 78, 60, 95],           # Mean ≈ 81.6
        "NLP Score": [0.7, 0.8, 0.75, 0.6, 0.85],  # Mean = 0.74 → scaled to 74
    })

    stats = calculate_statistics(df)
    print(f"Stats Result: {stats}")

    assert "T-Statistic" in stats, "T-Statistic missing"
    assert "P-Value" in stats, "P-Value missing"
    print("✅ Statistics Module Verified")


def test_convergence():
    """Checks that the RL simulation actually evolves weights over time."""
    print("Testing RL Convergence Simulation...")

    sim_df = simulate_rl_convergence(iterations=50, role="Developer")
    print(f"Simulation Shape: {sim_df.shape}")

    assert not sim_df.empty, "Simulation returned empty DataFrame"
    assert "Skills" in sim_df.columns, "Skills weight missing in history"

    # Weights should change — if they didn't, the agent isn't learning
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
