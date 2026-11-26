#!/bin/bash
#SBATCH --account=skadiri_1836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/walloc_seeds_%j.out
source $HOME/ml-learning-to-evict/slurm/env.sh
cd $HOME/ml-learning-to-evict
W=wenbo_shared; CURVES=$(find $W -name static_curves.pt | head -1)
for L in 4096 8192 16384 LB; do
  if [ $L = LB ]; then export LONGBENCH=$PWD/lb_rows_L4096.jsonl; else unset LONGBENCH; export RULER_LEN=$L; fi
  LAB=/scratch2/$USER/kvp/alloc_labels_L$L; OUT=alloc_out_L$L; IDS=alloc_ids_L$L.json
  D="uv run python alloc/train_alloc.py dump --splits $IDS --labels $LAB --split test --source ckpt --ratios 0.05 0.1 --out_dir $OUT"
  for s in 0 1; do for m in fixed input; do
    [ -f $OUT/w${m}_s${s}_r0.1.json ] && continue
    $D --ckpt $W/${m}_hw_s$s/best.pt --name w${m}_s$s || exit 1
  done; done
  uv run python alloc/static_plan.py plan --curves $CURVES --labels $LAB/test --ids alloc_idlist_L$L.json \
     --ratios 0.05 0.1 --name wstatic --out_dir $OUT || exit 1
  T=alloc_tasks_seeds_L$L.txt; : > $T
  for n in wfixed_s0 wfixed_s1 winput_s0 winput_s1 wstatic; do
    for r in 0.05 0.1; do echo "--alloc $OUT/${n}_r$r.json" >> $T; done
  done
  cat -n $T; ls -la $(sed 's/--alloc //' $T) || exit 1
done
