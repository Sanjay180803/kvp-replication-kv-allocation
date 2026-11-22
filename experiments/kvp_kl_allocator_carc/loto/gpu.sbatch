#!/bin/bash
#SBATCH --job-name=kvp-kl-loto
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/loto_%A_%a.log
set -euo pipefail
: "${KVP_LOTO_PIPELINE:?Use loto/submit_pipeline.sh}"
: "${KVP_LOTO_STAGE:?Missing stage}"
srun bash "$KVP_LOTO_CODE/run.sh" "$KVP_LOTO_STAGE" "$KVP_LOTO_PIPELINE"
