import os
import transformers

moe_path = os.path.join(os.path.dirname(transformers.__file__), 'integrations', 'moe.py')
if os.path.exists(moe_path):
    with open(moe_path, 'r') as f:
        code = f.read()
    
    # Wrap the custom_op registration in a try-except block
    if 'torch.library.custom_op("transformers::grouped_mm_fallback"' in code and 'try:' not in code:
        code = code.replace(
            'torch.library.custom_op("transformers::grouped_mm_fallback", _grouped_mm_fallback, mutates_args=())',
            'try:\n    torch.library.custom_op("transformers::grouped_mm_fallback", _grouped_mm_fallback, mutates_args=())\nexcept Exception:\n    pass'
        )
        with open(moe_path, 'w') as f:
            f.write(code)
        print("Successfully patched moe.py!")
    else:
        print("Already patched or pattern not found.")
else:
    print(f"moe.py not found at {moe_path}")
