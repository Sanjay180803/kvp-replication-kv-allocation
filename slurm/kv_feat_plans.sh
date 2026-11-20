#!/bin/bash
# Feature-shuffle / long-context allocations. LENGTH=4096|8192|16384, SHUFFLE=1 to add the shuffle controls.
#SBATCH --account=skadiri_1836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/kv_feat_plans_%j.out
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=/scratch2/$USER/kvp/hf_cache UV_CACHE_DIR=/scratch2/$USER/kvp/uv_cache
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd $HOME/kvp-team
uv run --no-project --with-requirements scripts/requirements-profiling.txt python scripts/kv_feat_plans.py \
  --length ${LENGTH:-4096} ${SHUFFLE:+--shuffle}
