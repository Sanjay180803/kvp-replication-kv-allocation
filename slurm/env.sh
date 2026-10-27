# Shared environment for all job scripts. Edit SCRATCH / REPO for your own account.
SCRATCH=/scratch2/$USER/kvp                      # big files: models, data, agents, caches
REPO=$HOME/ml-learning-to-evict                  # Apple's repo checkout (with our scripts/ copied in)
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=$SCRATCH/hf_cache UV_CACHE_DIR=$SCRATCH/uv_cache
export UV_NO_SYNC=1 UV_LINK_MODE=copy PROJECT_ROOT=$REPO
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd $PROJECT_ROOT
source .env
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
