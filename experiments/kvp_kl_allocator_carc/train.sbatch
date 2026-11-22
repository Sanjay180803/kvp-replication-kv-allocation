#!/bin/bash
#SBATCH --job-name=kvp-kl-alloc
#SBATCH --account=ctorng_1735
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/train_%A_%a.log

set -euo pipefail
: "${KVP_ALLOC_EXPERIMENT_DIR:?Use submit.sh}"
srun bash "$KVP_ALLOC_EXPERIMENT_DIR/run.sh" "${KVP_ALLOC_MODE:-train}"
