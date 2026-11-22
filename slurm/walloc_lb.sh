#!/bin/bash
#SBATCH --account=skadiri_1836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/walloc_lb_%j.out
source $HOME/ml-learning-to-evict/slurm/env.sh
cd $HOME/ml-learning-to-evict
export PYTORCH_ALLOC_CONF=expandable_segments:True
uv run python alloc/lb_data.py build --per_task 40 --maxp 3900 --out lb_rows_L4096.jsonl --ids alloc_ids_LLB.json || exit 1
export LONGBENCH=$PWD/lb_rows_L4096.jsonl
LAB=/scratch2/$USER/kvp/alloc_labels_LLB; OUT=alloc_out_LLB; IDS=alloc_ids_LLB.json
uv run python alloc/label.py --splits $IDS --split test --composite config_composites/ruler845_b128.yaml --out $LAB || exit 1
D="uv run python alloc/train_alloc.py dump --splits $IDS --labels $LAB --split test --ratios 0.05 0.1 --out_dir $OUT"
$D --source ckpt --ckpt alloc_ckpt/input_hw_s2/best.pt --name winput || exit 1
$D --source ckpt --ckpt alloc_ckpt/fixed_hw_s2/best.pt --name wfixed || exit 1
$D --source raw --name raw || exit 1
T=alloc_tasks_LLB.txt; : > $T
for f in $OUT/*.json; do echo "--alloc $f" >> $T; done
for r in 0.05 0.1; do
  echo "--uniform $r" >> $T
  echo "--team_static $HOME/kvp-team/rl_runs/plan/allocs_blend/shareplan_snapkv_t$r.json" >> $T
  echo "--team_static $HOME/kvp-team/allocations/profile-share_snapkv_t$r.json" >> $T
done
echo "eval tasks:"; cat -n $T
