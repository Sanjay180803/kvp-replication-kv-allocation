#!/bin/bash
# Train the 4 KV-head agents of one layer sequentially (array index = layer). Skips finished heads.
# Submit all layers: sbatch --array=0-27%8 train_layer_array.sh   (per-user submit limit: 108-task arrays were rejected)
#SBATCH --account=rashidin_1753
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=train_layer_%A_%a.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=YOUR_EMAIL
LAYER=$SLURM_ARRAY_TASK_ID
source $HOME/ml-learning-to-evict/slurm/env.sh
FAILED=0
for HEAD in 0 1 2 3; do
  DIR=agents/grouped/ruler845_v1/layer_$(printf %06d $LAYER)/kv_head_$(printf %03d $HEAD)
  if ls $DIR/*/figures >/dev/null 2>&1; then echo "=== L$LAYER H$HEAD already done, skipping"; continue; fi
  LOG=train_layer_${SLURM_ARRAY_JOB_ID}_${LAYER}_h${HEAD}.log
  echo "=== L$LAYER H$HEAD -> $LOG"
  uv run tune run src/kvcompression/entrypoints/rl/train_agent_sampler_distributed.py --config configs/train_single_head.yaml distributed=False target_layer_idx=$LAYER kv_head_idx=$HEAD sweep_name=ruler845_v1 compile_agent=False training.num_epochs_or_steps=4000 loader.train.batch_size=16 loader.eval.batch_size=4 > $LOG 2>&1
  STATUS=$?
  if [ $STATUS -ne 0 ] || grep -qi "out of memory" $LOG; then echo "!!! L$LAYER H$HEAD FAILED"; FAILED=1; fi
done
exit $FAILED
