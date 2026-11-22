#!/bin/bash
#SBATCH --job-name=kvp-loto-summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/loto_summary_%j.log
set -euo pipefail
srun bash "$KVP_LOTO_CODE/run.sh" summary "$KVP_LOTO_PIPELINE"
