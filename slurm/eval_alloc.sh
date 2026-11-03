#!/bin/bash
# RULER eval for every allocation JSON in allocations/ (one array task per file, sorted by name).
# Submit:  N=$(ls allocations/*.json | wc -l); sbatch --array=0-$((N-1)) slurm/eval_alloc.sh
#SBATCH --account=rashidin_1753
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=eval_alloc_%A_%a.log
source $HOME/ml-learning-to-evict/slurm/env.sh
FILES=($(ls ${ALLOC_DIR:-allocations}/*.json | sort))
F=${FILES[$SLURM_ARRAY_TASK_ID]}
echo "=== allocation: $F ==="
uv run python scripts/eval_ruler_alloc.py --alloc $F --per_task ${PER_TASK:-20} ${EXTRA_ARGS}
