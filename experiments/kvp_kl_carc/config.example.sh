# Copy to config.sh on CARC; edit paths for your own account.
CARC_ACCOUNT="ctorng_1735"
CARC_PARTITION="gpu"
CARC_GPU_TYPE="a40"

EXPERIMENT_ROOT="/scratch1/chenpaul/kvp_kl"
MODEL_DIR="/scratch1/chenpaul/kvp/models/Qwen2.5-7B-Instruct"
CHECKPOINT_ROOT="/scratch1/chenpaul/kvp/agents/ruler845_b128"
APPLE_REPO="$HOME/ml-learning-to-evict"
PYTHON_BIN="$EXPERIMENT_ROOT/venv/bin/python"

PROFILE_BATCH_SIZE=8
PROFILE_N_PER_TASK=5
PROFILE_RATIOS="0.02,0.05,0.1,0.25,0.5"
PROFILE_MIN_K=1
SMOKE_TIME="00:30:00"
PROFILE_TIME="04:00:00"
DATASET_REVISION=""
