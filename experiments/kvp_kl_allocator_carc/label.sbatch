#!/bin/bash
#SBATCH --job-name=kvp-alloc-labels
#SBATCH --account=ctorng_1735
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/labels_%j.log

set -euo pipefail
: "${KVP_ALLOC_EXPERIMENT_DIR:?Use submit_labels.sh}"
srun bash "$KVP_ALLOC_EXPERIMENT_DIR/run_labels.sh" "${KVP_LABEL_MODE:-smoke}"
