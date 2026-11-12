#!/bin/bash
# STEP=collect|train_critic|policies|evaluate|online  TARGET=0.05  (collect as an array = shards)
#SBATCH --account=skadiri_1836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/off_%x_%A_%a.out
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=/scratch2/$USER/kvp/hf_cache UV_CACHE_DIR=/scratch2/$USER/kvp/uv_cache
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd $HOME/kvp-team
T=${TARGET:-0.05}; STEP=${STEP:-collect}
PY="uv run --no-project --with-requirements scripts/requirements-profiling.txt python scripts/kv_offline_rl.py --target $T --init_alloc allocations/profile-share_snapkv_t$T.json"
SHARD=""; [ -n "$SLURM_ARRAY_TASK_ID" ] && SHARD="--shard $SLURM_ARRAY_TASK_ID"
echo "=== step $STEP target $T $SHARD ==="
if [ "$STEP" = "policies" ]; then
  for A in bc awr dpo; do $PY --step train_policy --algo $A ${EXTRA_ARGS}; done
  $PY --step train_policy --algo mbpo --kappa 1.0 ${EXTRA_ARGS}
  $PY --step train_policy --algo mbpo --kappa 0.0 ${EXTRA_ARGS}      # ablation: no pessimism
else
  $PY --step $STEP $SHARD ${EXTRA_ARGS}
fi
