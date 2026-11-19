#!/bin/bash
# Submit only when explicitly run by the user on CARC.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/config.sh"
MODE=${1:-smoke}
case "$MODE" in
    smoke) LIMIT=$SMOKE_TIME ;;
    profile) LIMIT=$PROFILE_TIME ;;
    *) printf 'Usage: bash submit.sh smoke|profile [sbatch options]\n' >&2; exit 2 ;;
esac
if (( $# > 0 )); then shift; fi
test -x "$PYTHON_BIN" || { printf 'Missing Python: %s\n' "$PYTHON_BIN" >&2; exit 2; }
test -d "$MODEL_DIR" || { printf 'Missing model: %s\n' "$MODEL_DIR" >&2; exit 2; }
test -d "$CHECKPOINT_ROOT" || { printf 'Missing checkpoints: %s\n' "$CHECKPOINT_ROOT" >&2; exit 2; }
test -d "$APPLE_REPO/src/kvcompression" || { printf 'Missing Apple source: %s\n' "$APPLE_REPO" >&2; exit 2; }
mkdir -p "$HERE/logs"
export KVP_EXPERIMENT_DIR="$HERE" KVP_MODE="$MODE"
JOB_ID=$(sbatch --parsable --account="$CARC_ACCOUNT" --partition="$CARC_PARTITION" \
    --gres="gpu:$CARC_GPU_TYPE:1" --time="$LIMIT" --chdir="$HERE" "$@" "$HERE/profile.sbatch")
printf 'Submitted %s job %s\nLog: %s/logs/kvp_kl_%s.log\n' "$MODE" "$JOB_ID" "$HERE" "${JOB_ID%%;*}"
