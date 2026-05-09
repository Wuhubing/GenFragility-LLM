import re

file_path = 'Makefile'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace HF_CACHE
content = re.sub(r'HF_CACHE \?= /tmp/hf_cache', r'HF_CACHE ?= /scratch/weibing_wang/huggingface_cache', content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Makefile patched successfully")
