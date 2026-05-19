import sys
# Create dummy pyairports module
import sys, types
module = types.ModuleType('pyairports')
module.airports = types.ModuleType('pyairports.airports')
module.airports.AIRPORT_LIST = []
sys.modules['pyairports'] = module
sys.modules['pyairports.airports'] = module.airports

try:
    from vllm import LLM
    print("VLLM IMPORT SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FAILED: {e}")
