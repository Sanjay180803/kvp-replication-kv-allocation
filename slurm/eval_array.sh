#!/bin/bash
# RULER-4K eval. Array index: 0=full 1=streaming 2=snapkv 3=kvp 4=random
# Submit: sbatch --array=0-4 eval_array.sh
# Smoke test: sbatch --array=3 --time=01:00:00 --export=ALL,PER_TASK=1,EXTRA_ARGS="--out_dir results_smoke" eval_array.sh
#SBATCH --account=rashidin_1753
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=eval_%A_%a.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=YOUR_EMAIL
METHODS=(full streaming snapkv kvp random)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
source $HOME/ml-learning-to-evict/slurm/env.sh
echo "=== method: $METHOD ==="
uv run python scripts/eval_ruler.py --method $METHOD --ratios 0.1 0.25 0.5 --per_task ${PER_TASK:-20} ${EXTRA_ARGS}
