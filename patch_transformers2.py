import os
import transformers

moe_path = os.path.join(os.path.dirname(transformers.__file__), 'integrations', 'moe.py')
with open(moe_path, 'r') as f:
    lines = f.readlines()

with open(moe_path, 'w') as f:
    for line in lines:
        if 'torch.library.custom_op("transformers::grouped_mm_fallback"' in line:
            indent = line[:len(line) - len(line.lstrip())]
            f.write(indent + 'try:\n')
            f.write(indent + '    ' + line.lstrip())
            f.write(indent + 'except Exception:\n')
            f.write(indent + '    pass\n')
        else:
            f.write(line)
