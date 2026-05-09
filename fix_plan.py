import re

with open("/home/weibing_wang/GenFragility-LLM/.hermes/plans/model_scale_up_plan.md", "r") as f:
    content = f.read()

# Fix the broken part around 7.4
broken_marker = "5. 删 LoRA checkpoint(节省磁盘)"
# I'll just rewrite the sections from 6.3 onwards to ensure it's clean and integrates the new logic.
