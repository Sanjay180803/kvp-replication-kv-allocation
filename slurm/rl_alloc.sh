#!/bin/bash
# MODE=oracle|train|dump  TARGET=0.05|0.1   e.g. sbatch --export=ALL,MODE=oracle,TARGET=0.05 slurm/rl_alloc.sh
#SBATCH --account=skadiri_1836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/rl_%x_%j.out
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=/scratch2/$USER/kvp/hf_cache UV_CACHE_DIR=/scratch2/$USER/kvp/uv_cache
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd $HOME/kvp-team
MODE=${MODE:-train}; T=${TARGET:-0.05}
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=== mode $MODE target $T ==="
uv run --no-project --with-requirements scripts/requirements-profiling.txt python scripts/rl_alloc.py \
  --mode $MODE --target $T --init_alloc allocations/profile-share_snapkv_t$T.json --out_dir rl_runs/t$T ${EXTRA_ARGS}
