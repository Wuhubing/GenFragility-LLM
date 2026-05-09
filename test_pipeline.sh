source ~/miniconda3/etc/profile.d/conda.sh
conda activate genfragility
export HF_HOME="/home/weibing_wang/huggingface_cache_large"

python pipeline_70b_main.py
