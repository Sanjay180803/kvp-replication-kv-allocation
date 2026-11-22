# Copy to config.sh on CARC. No credentials belong in this file.
CARC_ACCOUNT="ctorng_1735"
CARC_PARTITION="gpu"
CARC_GPU_TYPE="a40"
TIME_LIMIT="00:45:00"
SMOKE_TIME="00:15:00"
SEEDS="0-2"

# Reuse the environment and assets from the successful CARC profiling run.
PYTHON_BIN="/scratch1/chenpaul/kvp_kl/venv/bin/python"
MODEL_DIR="/scratch1/chenpaul/kvp/models/Qwen2.5-7B-Instruct"
CHECKPOINT_ROOT="/scratch1/chenpaul/kvp/agents/ruler845_b128"
APPLE_REPO="$HOME/ml-learning-to-evict"

# Separate writable area. Label generation writes here; policy training reads it.
EXPERIMENT_ROOT="/scratch1/chenpaul/kvp_kl_allocator"
DATA_DIR="/scratch1/chenpaul/kvp/allocator_data"
SPLITS_JSON="$KVP_ALLOC_EXPERIMENT_DIR/inputs/splits.json"
LABELS_DIR="$DATA_DIR/labels_carc"
COMPOSITE="$KVP_ALLOC_EXPERIMENT_DIR/inputs/ruler845_b128.yaml"
LABEL_TIME="02:00:00"
LABEL_SMOKE_TIME="00:15:00"
DATASET_REVISION=""

# The transfer bundle includes this CSV. Local repo use falls back to the sibling experiment.
PROFILE_CSV="$KVP_ALLOC_EXPERIMENT_DIR/profiles/head_kl.csv"
if [[ ! -f "$PROFILE_CSV" ]]; then
    PROFILE_CSV="$KVP_ALLOC_EXPERIMENT_DIR/../kvp_kl_carc/results/profile_14372014/head_kl.csv"
fi

# Optional: match KVP prompt/ratio coverage to Wenbo's actual SnapKV input CSVs.
# Empty = use all 65 prompts at each of 2%, 5%, 10%, 25%.
# Example: MATCH_SNAPKV_CSVS=("/path/profiling/head_kl.csv" "/path/profiling_lowr/head_kl.csv")
MATCH_SNAPKV_CSVS=()

N_WEIGHT_TRAIN=260
FIXED_STEPS=3000
INPUT_STEPS=10000
INPUT_PATIENCE=20
BATCH_SIZE=8
RLOO_SAMPLES=16
