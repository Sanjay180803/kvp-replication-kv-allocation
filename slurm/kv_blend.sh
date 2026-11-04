#!/bin/bash
# Shrinkage planner: tune alpha on val KL, then dump per-prompt allocations. Array task 0 -> 5%, 1 -> 10%.
# STEPS env var picks steps (default "tune dump").
#SBATCH --account=skadiri_1836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/kv_blend_%A_%a.out
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=/scratch2/$USER/kvp/hf_cache UV_CACHE_DIR=/scratch2/$USER/kvp/uv_cache
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd $HOME/kvp-team
TS=(0.05 0.1); T=${TS[$SLURM_ARRAY_TASK_ID]}
RUN="uv run --no-project --with-requirements scripts/requirements-profiling.txt python"
ARGS="--target $T --init_alloc allocations/profile-share_snapkv_t$T.json --run_dir rl_runs/plan"
for S in ${STEPS:-tune dump}; do $RUN scripts/kv_plan.py --step $S $ARGS || exit 1; done
