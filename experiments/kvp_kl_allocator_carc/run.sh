#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm or run in an allocated GPU session}"
export KVP_ALLOC_EXPERIMENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HERE=$KVP_ALLOC_EXPERIMENT_DIR
source "$HERE/config.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
MODE=${1:-smoke}
case "$MODE" in smoke|train) ;; *) echo "Unknown mode: $MODE" >&2; exit 2;; esac
SEED=${SLURM_ARRAY_TASK_ID:-0}
# Fresh work directory per job/seed: reward-reference caches must never cross weight variants.
RUN="$EXPERIMENT_ROOT/runs/${MODE}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_s$SEED"
mkdir -p "$RUN"
if [[ -e "$RUN/run_config.json" ]]; then
    echo "Run already exists: $RUN. Submit a new job; no implicit overwrite/resume." >&2; exit 2
fi
printf 'Job %s | mode %s | seed %s | output %s\n' "$SLURM_JOB_ID" "$MODE" "$SEED" "$RUN"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU required"; print("torch", torch.__version__)'

MATCH_ARGS=()
if (( ${#MATCH_SNAPKV_CSVS[@]} )); then MATCH_ARGS=(--match-snapkv "${MATCH_SNAPKV_CSVS[@]}"); fi
"$PYTHON_BIN" "$HERE/prepare.py" --profile "$PROFILE_CSV" --splits "$SPLITS_JSON" \
    --labels "$LABELS_DIR" --out "$RUN" --n-train "$N_WEIGHT_TRAIN" \
    --mode "$MODE" --seed "$SEED" "${MATCH_ARGS[@]}"
SPLIT="$RUN/splits.json"
WEIGHTS="$RUN/head_weights_kvp.pt"
WORK="$RUN/work"
FSTEPS=$FIXED_STEPS; ISTEPS=$INPUT_STEPS; BATCH=$BATCH_SIZE; K=$RLOO_SAMPLES
EXTRA=()
if [[ "$MODE" == smoke ]]; then
    FSTEPS=2; ISTEPS=2; BATCH=1; K=2; EXTRA=(--eval_every 1 --patience 2)
fi
COMMON=(--splits "$SPLIT" --labels "$LABELS_DIR" --work "$WORK" --device cuda
        --head_weights "$WEIGHTS" --seed "$SEED" --batch "$BATCH" --K "$K")
"$PYTHON_BIN" "$HERE/vendor/wenbo/train_alloc.py" train "${COMMON[@]}" \
    --mode fixed --steps "$FSTEPS" "${EXTRA[@]}"
"$PYTHON_BIN" "$HERE/vendor/wenbo/train_alloc.py" train "${COMMON[@]}" \
    --mode input --steps "$ISTEPS" --patience "$INPUT_PATIENCE" \
    --init "$WORK/fixed_hw_s$SEED/best.pt" "${EXTRA[@]}"
"$PYTHON_BIN" "$HERE/finish.py" --run "$RUN" --seed "$SEED"
printf 'SUCCESS: %s seed %s. Results: %s\n' "$MODE" "$SEED" "$RUN"
