import os

filepath = '/home/weibing_wang/GenFragility-LLM/pipeline_trial_main.py'
with open(filepath, 'r') as f:
    content = f.read()

# Replace the folder naming logic to include the specific model name
old_code = """    # Define experiment parameters
    MODEL_SIZE_STR = "0.5b"
    BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Read targets"""

new_code = """    # Define experiment parameters
    BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    MODEL_NAME_SAFE = BASE_MODEL.split('/')[-1] # e.g., Qwen2.5-0.5B-Instruct
    
    # Read targets"""

content = content.replace(old_code, new_code)

old_dir_code = """    exp_dir_name = f"{MODEL_SIZE_STR}_hub{NUM_HUBS}_tail{NUM_TAILS}_experiment\""""
new_dir_code = """    exp_dir_name = f"{MODEL_NAME_SAFE}_hub{NUM_HUBS}_tail{NUM_TAILS}_experiment\""""

content = content.replace(old_dir_code, new_dir_code)

old_run_id_code = """        run_id = f"{MODEL_SIZE_STR}_trial_{target_id}\""""
new_run_id_code = """        run_id = f"{MODEL_NAME_SAFE}_trial_{target_id}\""""

content = content.replace(old_run_id_code, new_run_id_code)

with open(filepath, 'w') as f:
    f.write(content)
