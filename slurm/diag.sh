#!/bin/bash
# Training-reward diagnostics (see scripts/diag_rewards*.py). Edit the command at the bottom.
#SBATCH --account=rashidin_1753
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=diag_%j.log
source $HOME/ml-learning-to-evict/slurm/env.sh
export DISABLE_TORCH_COMPILER=1
uv run python scripts/diag_rewards.py 0,4,8,12,16,20,24,27 12
# uv run python scripts/diag_rewards2.py 0,8,16,27 12
# uv run python scripts/diag_rewards2.py 16:1,8:3 44 config_composites/ruler845_b128.yaml
