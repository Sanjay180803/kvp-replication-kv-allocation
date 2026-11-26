#!/bin/bash
#SBATCH --account=skadiri_1836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/walloc_a3_%j.out
source $HOME/ml-learning-to-evict/slurm/env.sh
cd $HOME/ml-learning-to-evict
CK=/project2/rashidin_1753/chenpaul/kvp_kl_allocator3/checkpoints
for L in 4096 8192 16384 LB; do
  if [ $L = LB ]; then export LONGBENCH=$PWD/lb_rows_L4096.jsonl; else unset LONGBENCH; export RULER_LEN=$L; fi
  LAB=/scratch2/$USER/kvp/alloc_labels_L$L; OUT=alloc_out_L$L; IDS=alloc_ids_L$L.json
  D="uv run python alloc/train_alloc.py dump --splits $IDS --labels $LAB --split test --source ckpt --ratios 0.05 0.1 --out_dir $OUT"
  T=alloc_tasks_a3_L$L.txt; : > $T
  for s in 0 1 2; do for m in fixed input; do
    [ -f $CK/${m}_hw_s$s/best.pt ] || { echo "missing $CK/${m}_hw_s$s/best.pt"; exit 1; }
    $D --ckpt $CK/${m}_hw_s$s/best.pt --name k${m}_s$s || exit 1
    for r in 0.05 0.1; do echo "--alloc $OUT/k${m}_s${s}_r$r.json" >> $T; done
  done; done
  cat -n $T; ls -la $(sed 's/--alloc //' $T) || exit 1
done
