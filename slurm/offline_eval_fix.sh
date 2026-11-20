#!/bin/bash
# Re-run kv_offline_rl.py --step evaluate after the indexing-bug fix (CARC).
#SBATCH --account=skadiri_1836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=logs/offline_eval_fix_%j.out
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=/scratch2/$USER/kvp/hf_cache UV_CACHE_DIR=/scratch2/$USER/kvp/uv_cache
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd $HOME/kvp-team
uv run --no-project --with-requirements scripts/requirements-profiling.txt python scripts/kv_offline_rl.py \
  --target 0.05 --init_alloc allocations/profile-share_snapkv_t0.05.json \
  --run_dir rl_runs/offline_t0.05 --step evaluate --policies awr+dpo+mbpo_k1.0+mbpo_k0.0
