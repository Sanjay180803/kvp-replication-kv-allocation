#!/bin/bash
set -euo pipefail
export KVP_ALLOC_EXPERIMENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$KVP_ALLOC_EXPERIMENT_DIR/config.sh"
MODE=${1:-smoke}
case "$MODE" in
    smoke) LIMIT=${LABEL_SMOKE_TIME:-00:15:00} ;;
    full) LIMIT=${LABEL_TIME:-02:00:00} ;;
    *) echo 'Usage: bash submit_labels.sh smoke|full [extra sbatch options]' >&2; exit 2;;
esac
if (( $#>0 ));then shift;fi
test -x "$PYTHON_BIN" || { echo "Missing Python: $PYTHON_BIN" >&2;exit 2; }
test -r "$SPLITS_JSON" || { echo "Missing exact splits: $SPLITS_JSON" >&2;exit 2; }
test -r "$MODEL_DIR/config.json" || { echo "Missing model: $MODEL_DIR" >&2;exit 2; }
test -d "$CHECKPOINT_ROOT" || { echo "Missing KVP checkpoints: $CHECKPOINT_ROOT" >&2;exit 2; }
test -d "$APPLE_REPO/src/kvcompression" || { echo "Missing Apple source: $APPLE_REPO" >&2;exit 2; }
mkdir -p "$KVP_ALLOC_EXPERIMENT_DIR/logs"
export KVP_LABEL_MODE="$MODE"
sbatch --parsable --account="$CARC_ACCOUNT" --partition="$CARC_PARTITION" \
    --gres="gpu:$CARC_GPU_TYPE:1" --time="$LIMIT" --chdir="$KVP_ALLOC_EXPERIMENT_DIR" \
    "$@" "$KVP_ALLOC_EXPERIMENT_DIR/label.sbatch"
