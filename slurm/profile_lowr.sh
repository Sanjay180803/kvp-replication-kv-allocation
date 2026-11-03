#!/bin/bash
#SBATCH --account=skadiri_1836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/profile_lowr_%j.out
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=/scratch2/$USER/kvp/hf_cache UV_CACHE_DIR=/scratch2/$USER/kvp/uv_cache
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd $HOME/kvp-team
nvidia-smi --query-gpu=name,memory.total --format=csv
RUN="uv run --no-project --with-requirements scripts/requirements-profiling.txt python"
$RUN scripts/profile_heads.py --model /scratch2/$USER/kvp/models/Qwen2.5-7B-Instruct \
  --out_dir results/profiling_lowr --n_per_task 3 --scorers snapkv \
  --ratios 0.02,0.05,0.1,0.25,0.5 --batch_size 16 --skip_sanity
$RUN scripts/analyze_profiling.py --dir results/profiling_lowr
