#!/bin/bash
# Extract Q/K/V traces for 13 RULER-4K chunks (one per task; dataset is ordered by task).
# Chunk 0 was run first on its own; this array covers the other 12. Submit: sbatch --array=0-12 preprocess_array.sh
#SBATCH --account=rashidin_1753
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=preprocess_%A_%a.log
CHUNKS=(0 8 16 24 31 39 47 54 62 70 77 85 93)
C=${CHUNKS[$SLURM_ARRAY_TASK_ID]}
source $HOME/ml-learning-to-evict/slurm/env.sh
uv run tune run src/kvcompression/entrypoints/datagen/unroll_and_store.py --config configs/preprocess_ruler.yaml device=cuda:0 total_num_chunks=100 current_chunk=$C
