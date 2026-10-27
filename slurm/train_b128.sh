#!/bin/bash
# Batch-size test: retrain two heads at Apple's batch size 128 (fits on one 48 GB GPU with gumbel chunk 4).
# Submit: sbatch --array=0-1 train_b128.sh    (~6h45 per agent)
#SBATCH --account=rashidin_1753
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=train_b128_%A_%a.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=YOUR_EMAIL
PAIRS=("16 1" "8 3")
read LAYER HEAD <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"
source $HOME/ml-learning-to-evict/slurm/env.sh
echo "=== layer $LAYER head $HEAD, batch 128 ==="
uv run tune run src/kvcompression/entrypoints/rl/train_agent_sampler_distributed.py --config configs/train_single_head.yaml distributed=False target_layer_idx=$LAYER kv_head_idx=$HEAD sweep_name=ruler845_b128 compile_agent=False training.num_epochs_or_steps=4000 loader.train.batch_size=128 loader.eval.batch_size=4 trainer.gumbel_sampling_chunk_size=4
