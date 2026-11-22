#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm for GPU label generation}"
export KVP_ALLOC_EXPERIMENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$KVP_ALLOC_EXPERIMENT_DIR/config.sh"
export PYTHONPATH="$APPLE_REPO/src:${PYTHONPATH:-}" PROJECT_ROOT="$APPLE_REPO"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 HF_HOME="/scratch1/chenpaul/kvp_kl/hf_cache"
export PYTORCH_ALLOC_CONF=expandable_segments:True
MODE=${1:-smoke}
case "$MODE" in
    smoke) OUT="$EXPERIMENT_ROOT/label_smoke_$SLURM_JOB_ID" ;;
    full) OUT="$LABELS_DIR" ;;
    *) echo "Unknown mode: $MODE" >&2;exit 2;;
esac
echo "Job $SLURM_JOB_ID | labeling $MODE | output $OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# Public Apple source needs the team's smaller FlexAttention tiles on A40.
# This is idempotent, preserves a backup, and runs before kvcompression imports.
"$PYTHON_BIN" "$KVP_ALLOC_EXPERIMENT_DIR/patch_a40_attention.py" --apple-repo "$APPLE_REPO"
REVISION=()
if [[ -n ${DATASET_REVISION:-} ]];then REVISION=(--dataset-revision "$DATASET_REVISION");fi
"$PYTHON_BIN" "$KVP_ALLOC_EXPERIMENT_DIR/label_carc.py" \
    --splits "$SPLITS_JSON" --model "$MODEL_DIR" \
    --composite "${COMPOSITE:-$KVP_ALLOC_EXPERIMENT_DIR/inputs/ruler845_b128.yaml}" \
    --checkpoint-root "$CHECKPOINT_ROOT" --out "$OUT" --mode "$MODE" "${REVISION[@]}"
