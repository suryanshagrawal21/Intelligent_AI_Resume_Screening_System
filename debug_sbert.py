"""Quick diagnostic — checks that sentence-transformers and torch are importable."""

import sys

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

try:
    import sentence_transformers
    print(f"✅ sentence_transformers version: {sentence_transformers.__version__}")
    from sentence_transformers import SentenceTransformer
    print("✅ SentenceTransformer class imported successfully.")
except ImportError as e:
    print(f"❌ ImportError: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

try:
    import torch
    print(f"✅ torch version: {torch.__version__}")
except ImportError as e:
    print(f"❌ Torch ImportError: {e}")
