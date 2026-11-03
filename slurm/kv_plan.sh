#!/bin/bash
# STEP=collect|train|evaluate|dump  TARGET=0.05   (collect as an array = shards)
#SBATCH --account=skadiri_1836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/plan_%x_%A_%a.out
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=/scratch2/$USER/kvp/hf_cache UV_CACHE_DIR=/scratch2/$USER/kvp/uv_cache
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd $HOME/kvp-team
T=${TARGET:-0.05}; STEP=${STEP:-collect}
SHARD=""; [ -n "$SLURM_ARRAY_TASK_ID" ] && SHARD="--shard $SLURM_ARRAY_TASK_ID"
echo "=== step $STEP target $T $SHARD ==="
uv run --no-project --with-requirements scripts/requirements-profiling.txt python scripts/kv_plan.py \
  --step $STEP --target $T --init_alloc allocations/profile-share_snapkv_t$T.json $SHARD ${EXTRA_ARGS}
