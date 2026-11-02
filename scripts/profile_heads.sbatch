#!/bin/bash
#SBATCH --job-name=kv-head-profile
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/profile_%j.out

# Usage (from the repo root):  mkdir -p logs && sbatch scripts/profile_heads.sbatch
# Extra args pass through, e.g. sbatch scripts/profile_heads.sbatch --n_per_task 5 --scorers snapkv,streaming
set -euo pipefail
export HF_HOME=/scratch2/$USER/hf_cache
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# activate your env here, e.g.:
# source /scratch2/$USER/envs/kvp/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/profile_heads.py --out_dir results/profiling "$@"
python scripts/analyze_profiling.py --dir results/profiling
