#!/bin/bash
# Queue the entire 4K pipeline. No jobs are submitted until this is run on CARC.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export KVP_ALLOC_EXPERIMENT_DIR=$(cd "$HERE/.." && pwd)
source "$KVP_ALLOC_EXPERIMENT_DIR/config.sh"
SETTINGS="$HERE/settings.example.sh"
if [[ -f "$HERE/settings.sh" ]]; then SETTINGS="$HERE/settings.sh"; fi
source "$SETTINGS"
"$PYTHON_BIN" "$HERE/protocol.py" preflight --splits "$SPLITS_JSON" --labels "$LABELS_DIR" --profile "$PROFILE_CSV"
mkdir -p "$EXPERIMENT_ROOT/loto_runs" "$KVP_ALLOC_EXPERIMENT_DIR/logs"
export KVP_LOTO_PIPELINE=$(mktemp -d "$EXPERIMENT_ROOT/loto_runs/run_XXXXXX")
# Freeze the submitted code/config so later edits do not change queued stages.
mkdir "$KVP_LOTO_PIPELINE/code"
cp -R "$HERE" "$KVP_LOTO_PIPELINE/code/loto"
cp -R "$KVP_ALLOC_EXPERIMENT_DIR/vendor" "$KVP_LOTO_PIPELINE/code/vendor"
for f in prepare.py finish.py label_carc.py label_io.py ruler_scoring.py patch_a40_attention.py; do
    cp "$KVP_ALLOC_EXPERIMENT_DIR/$f" "$KVP_LOTO_PIPELINE/code/$f"
done
cp "$KVP_ALLOC_EXPERIMENT_DIR/config.sh" "$KVP_LOTO_PIPELINE/base_config.sh"
cp "$SETTINGS" "$KVP_LOTO_PIPELINE/settings.sh"
cp "$SPLITS_JSON" "$KVP_LOTO_PIPELINE/splits.original.json"
cp "$PROFILE_CSV" "$KVP_LOTO_PIPELINE/profile.csv"
COMPOSITE=${COMPOSITE:-$KVP_ALLOC_EXPERIMENT_DIR/inputs/ruler845_b128.yaml}
cp "$COMPOSITE" "$KVP_LOTO_PIPELINE/composite.original.yaml"
mkdir "$KVP_LOTO_PIPELINE/match_snapkv"
for i in "${!MATCH_SNAPKV_CSVS[@]}"; do
    cp "${MATCH_SNAPKV_CSVS[$i]}" "$KVP_LOTO_PIPELINE/match_snapkv/$i.csv"
done
export KVP_LOTO_CODE="$KVP_LOTO_PIPELINE/code/loto"
printf 'Pipeline directory: %s\n' "$KVP_LOTO_PIPELINE"
submit_gpu() {
    export KVP_LOTO_STAGE=$1
    shift
    sbatch --parsable --account="$CARC_ACCOUNT" --partition="$CARC_PARTITION" \
        --gres="gpu:$CARC_GPU_TYPE:1" --chdir="$KVP_ALLOC_EXPERIMENT_DIR" \
        --kill-on-invalid-dep=yes "$@" "$KVP_LOTO_CODE/gpu.sbatch"
}
SMOKE=$(submit_gpu smoke --mem=96G --time="$LOTO_SMOKE_TIME" --output=logs/loto_smoke_%j.log)
printf 'smoke %s\n' "$SMOKE" >> "$KVP_LOTO_PIPELINE/submitted_jobs.txt"
FEATURES=$(submit_gpu features --mem=96G --time="$LOTO_FEATURE_TIME" --dependency="afterok:$SMOKE" --output=logs/loto_features_%j.log)
printf 'features %s\n' "$FEATURES" >> "$KVP_LOTO_PIPELINE/submitted_jobs.txt"
FOLDS=$(submit_gpu fold --mem=64G --time="$LOTO_TRAIN_TIME" \
    --array="0-12%$LOTO_MAX_PARALLEL" --dependency="afterok:$FEATURES")
printf 'folds %s\n' "$FOLDS" >> "$KVP_LOTO_PIPELINE/submitted_jobs.txt"
EVAL=$(submit_gpu eval --mem=96G --time="$LOTO_EVAL_TIME" \
    --array="0-5%$LOTO_MAX_PARALLEL" --dependency="afterok:$FOLDS")
printf 'eval %s\n' "$EVAL" >> "$KVP_LOTO_PIPELINE/submitted_jobs.txt"
SUMMARY=$(sbatch --parsable --account="$CARC_ACCOUNT" --partition="$LOTO_CPU_PARTITION" \
    --chdir="$KVP_ALLOC_EXPERIMENT_DIR" --dependency="afterok:$EVAL" \
    --kill-on-invalid-dep=yes "$KVP_LOTO_CODE/summary.sbatch")
printf 'summary %s\n' "$SUMMARY" >> "$KVP_LOTO_PIPELINE/submitted_jobs.txt"
"$PYTHON_BIN" "$HERE/protocol.py" record --out "$KVP_LOTO_PIPELINE" \
    --jobs "$SMOKE" "$FEATURES" "$FOLDS" "$EVAL" "$SUMMARY" --seed "$LOTO_SEED"
printf 'Pipeline: %s\nSmoke: %s\nFeatures: %s\n13 folds: %s\n6 evaluations: %s\nSummary: %s\n' \
    "$KVP_LOTO_PIPELINE" "$SMOKE" "$FEATURES" "$FOLDS" "$EVAL" "$SUMMARY"
