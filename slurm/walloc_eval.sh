#!/bin/bash
# One eval per array task: line SLURM_ARRAY_TASK_ID (0-based) of alloc_tasks_L$L.txt, all through Wenbo's eval_alloc.py
#SBATCH --account=skadiri_1836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/walloc_eval_%A_%a.out
source $HOME/ml-learning-to-evict/slurm/env.sh
cd $HOME/ml-learning-to-evict
export RULER_LEN=$L PYTORCH_ALLOC_CONF=expandable_segments:True
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ${TASKS:-alloc_tasks_L$L.txt})
[ -z "$ARGS" ] && { echo "no task $SLURM_ARRAY_TASK_ID"; exit 0; }
echo "=== L$L task $SLURM_ARRAY_TASK_ID: $ARGS ==="
uv run python alloc/eval_alloc.py --splits alloc_ids_L$L.json --split test $ARGS \
  --composite config_composites/ruler845_b128.yaml --out_dir results_walloc_L$L
