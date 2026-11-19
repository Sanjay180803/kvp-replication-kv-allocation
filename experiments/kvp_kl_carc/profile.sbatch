#!/bin/bash
#SBATCH --job-name=kvp-kl
#SBATCH --account=ctorng_1735
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/kvp_kl_%j.log

set -euo pipefail
: "${KVP_EXPERIMENT_DIR:?Use submit.sh or export the absolute experiment directory}"
srun bash "$KVP_EXPERIMENT_DIR/run.sh" "${KVP_MODE:-smoke}"
