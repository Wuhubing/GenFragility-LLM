import sys
try:
    from vllm import LLM
    print("VLLM IMPORT SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
