import sys
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

try:
    import sentence_transformers
    print(f"Success! sentence_transformers version: {sentence_transformers.__version__}")
    from sentence_transformers import SentenceTransformer
    print("Successfully imported SentenceTransformer class.")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

try:
    import torch
    print(f"Success! torch version: {torch.__version__}")
except ImportError as e:
    print(f"Torch ImportError: {e}")
