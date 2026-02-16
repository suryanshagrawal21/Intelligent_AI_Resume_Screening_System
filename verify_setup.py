
import sys
import os

print("--- Verifying Setup (Granular) ---")

try:
    print("Importing src.adaptive_engine...")
    import src.adaptive_engine
    print("✅ src.adaptive_engine imported")

    print("Importing src.matcher...")
    import src.matcher
    print("✅ src.matcher imported")

    print("Importing src.parser...")
    import src.parser
    print("✅ src.parser imported")

    print("Importing src.preprocessing...")
    import src.preprocessing
    print("✅ src.preprocessing imported")

    print("Importing src.scoring...")
    import src.scoring
    print("✅ src.scoring imported")

    print("Importing src.experiment...")
    import src.experiment
    print("✅ src.experiment imported")

    print("Importing src.research_lab...")
    import src.research_lab
    print("✅ src.research_lab imported")

except ImportError as e:
    print(f"❌ Import failed at step '{sys.exc_info()[2].tb_next}': {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

print("--- Verification Complete ---")
