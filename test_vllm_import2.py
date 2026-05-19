import sys
try:
    from vllm import LLM
    print("VLLM IMPORT SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FAILED: {e}")
