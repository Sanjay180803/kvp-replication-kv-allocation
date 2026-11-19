#!/bin/bash
# Run directly on an already allocated GPU compute node. No sbatch is called.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${SLURM_JOB_ID:?Run inside an allocated GPU compute session, or use submit.sh}"
source "$HERE/config.sh"
REPO_ROOT="$HERE"
export PYTHONPATH="$REPO_ROOT/scripts:$APPLE_REPO/src:${PYTHONPATH:-}"
export PROJECT_ROOT="$APPLE_REPO"
export HF_HOME="$EXPERIMENT_ROOT/hf_cache"
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLCONFIGDIR="$EXPERIMENT_ROOT/matplotlib"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

MODE=${1:-${KVP_MODE:-smoke}}
OUT_DIR=${KVP_OUT_DIR:-$EXPERIMENT_ROOT/runs/${MODE}_${SLURM_JOB_ID}}
COMPOSITE=${COMPOSITE:-$REPO_ROOT/configs/ruler845_b128.yaml}
mkdir -p "$OUT_DIR"
printf 'Job: %s | node: %s | mode: %s\n' "$SLURM_JOB_ID" "$(hostname)" "$MODE"
printf 'Output directory: %s\n' "$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

"$PYTHON_BIN" "$HERE/preflight.py" --model "$MODEL_DIR" \
    --composite "$COMPOSITE" --checkpoint_root "$CHECKPOINT_ROOT" --require_gpu \
    --out "$OUT_DIR/preflight.json"

ARGS=(--model "$MODEL_DIR" --composite "$COMPOSITE" --checkpoint_root "$CHECKPOINT_ROOT"
      --out_dir "$OUT_DIR" --scorers kvp --batch_size "$PROFILE_BATCH_SIZE"
      --n_per_task "$PROFILE_N_PER_TASK" --ratios "$PROFILE_RATIOS"
      --min_k "$PROFILE_MIN_K" --prompt_style torchtune)
if [[ -n ${DATASET_REVISION:-} ]]; then ARGS+=(--dataset_revision "$DATASET_REVISION"); fi
case "$MODE" in
    smoke) ARGS+=(--max_prompts 1 --heads 0:0,4:1,22:0) ;;
    profile) ;;
    *) printf 'Unknown mode: %s\n' "$MODE" >&2; exit 2 ;;
esac

# Keep the caller's prepared environment; do not activate conda/reset modules here.
"$PYTHON_BIN" "$REPO_ROOT/scripts/profile_heads.py" "${ARGS[@]}"
"$PYTHON_BIN" "$REPO_ROOT/scripts/validate_profile.py" --dir "$OUT_DIR"
if [[ "$MODE" == profile ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/analyze_profiling.py" --dir "$OUT_DIR" --scorer kvp
fi
printf 'SUCCESS: %s completed. Results: %s\n' "$MODE" "$OUT_DIR"
