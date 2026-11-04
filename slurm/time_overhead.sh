#!/bin/bash
#SBATCH --account=skadiri_1836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/time_overhead_%j.out
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=/scratch2/$USER/kvp/hf_cache UV_CACHE_DIR=/scratch2/$USER/kvp/uv_cache
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd $HOME/kvp-team
RUN="uv run --no-project --with-requirements scripts/requirements-profiling.txt python"
$RUN scripts/check_plan.py && $RUN scripts/time_overhead.py
