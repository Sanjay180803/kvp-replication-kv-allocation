#!/bin/bash
set -euo pipefail
STAGE=$1
PIPELINE=$2
source "$PIPELINE/base_config.sh"
source "$PIPELINE/settings.sh"
HERE="$PIPELINE/code/loto"
MATCH_SNAPKV_CSVS=()
for f in "$PIPELINE"/match_snapkv/*.csv; do
    if [[ -f "$f" ]]; then MATCH_SNAPKV_CSVS+=("$f"); fi
done
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$APPLE_REPO/src:${PYTHONPATH:-}" PROJECT_ROOT="$APPLE_REPO"
export HF_HOME="/scratch1/chenpaul/kvp_kl/hf_cache" PYTORCH_ALLOC_CONF=expandable_segments:True
if [[ "$STAGE" != summary ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
    "$PYTHON_BIN" "$PIPELINE/code/patch_a40_attention.py" --apple-repo "$APPLE_REPO"
fi
ARGS=(--pipeline "$PIPELINE" --labels "$LABELS_DIR" --model "$MODEL_DIR"
      --checkpoint-root "$CHECKPOINT_ROOT" --seed "$LOTO_SEED"
      --fixed-steps "$FIXED_STEPS" --input-steps "$INPUT_STEPS" --patience "$INPUT_PATIENCE"
      --batch "$BATCH_SIZE" --samples "$RLOO_SAMPLES" --n-weight-train "$N_WEIGHT_TRAIN")
if (( ${#MATCH_SNAPKV_CSVS[@]} )); then ARGS+=(--match-snapkv "${MATCH_SNAPKV_CSVS[@]}"); fi
if [[ -n ${DATASET_REVISION:-} ]]; then ARGS+=(--dataset-revision "$DATASET_REVISION"); fi
"$PYTHON_BIN" "$HERE/worker.py" "$STAGE" "${ARGS[@]}" --index "${SLURM_ARRAY_TASK_ID:-0}"
