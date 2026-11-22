#!/bin/bash
#SBATCH --account=skadiri_1836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/walloc_raw_%j.out
source $HOME/ml-learning-to-evict/slurm/env.sh
cd $HOME/ml-learning-to-evict
for L in 4096 8192 16384; do
  export RULER_LEN=$L
  OUT=alloc_out_L$L
  uv run python alloc/train_alloc.py dump --splits alloc_ids_L$L.json --labels /scratch2/$USER/kvp/alloc_labels_L$L \
     --split test --source raw --name raw --ratios 0.05 0.1 --out_dir $OUT || exit 1
  printf -- "--alloc $OUT/raw_r0.05.json\n--alloc $OUT/raw_r0.1.json\n" > alloc_tasks_raw_L$L.txt
  cat alloc_tasks_raw_L$L.txt
done
