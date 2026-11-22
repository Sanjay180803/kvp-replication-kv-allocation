#!/bin/bash
set -euo pipefail
export KVP_ALLOC_EXPERIMENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$KVP_ALLOC_EXPERIMENT_DIR/config.sh"
MODE=${1:-smoke}
case "$MODE" in
    smoke) LIMIT=$SMOKE_TIME; ARRAY=0 ;;
    train) LIMIT=$TIME_LIMIT; ARRAY=$SEEDS ;;
    *) printf 'Usage: bash submit.sh smoke|train [extra sbatch options]\n' >&2; exit 2 ;;
esac
if (( $# > 0 )); then shift; fi
test -x "$PYTHON_BIN" || { echo "Missing Python: $PYTHON_BIN" >&2; exit 2; }
test -r "$PROFILE_CSV" || { echo "Missing profile: $PROFILE_CSV" >&2; exit 2; }
test -r "$SPLITS_JSON" || { echo "Download Wenbo's exact splits.json to $SPLITS_JSON" >&2; exit 2; }
test -d "$LABELS_DIR/train" && test -d "$LABELS_DIR/val" || {
    echo "Download Wenbo's train and val labels to $LABELS_DIR" >&2; exit 2;
}
mkdir -p "$KVP_ALLOC_EXPERIMENT_DIR/logs"
export KVP_ALLOC_MODE="$MODE"
sbatch --parsable --account="$CARC_ACCOUNT" --partition="$CARC_PARTITION" \
    --gres="gpu:$CARC_GPU_TYPE:1" --time="$LIMIT" --array="$ARRAY" \
    --chdir="$KVP_ALLOC_EXPERIMENT_DIR" "$@" "$KVP_ALLOC_EXPERIMENT_DIR/train.sbatch"
