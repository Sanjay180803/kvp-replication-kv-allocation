#!/bin/bash
# Wenbo's allocator at RULER length $L: labels -> allocation dumps -> list of eval jobs (alloc_tasks_L$L.txt)
#SBATCH --account=skadiri_1836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/walloc_prep_%j.out
source $HOME/ml-learning-to-evict/slurm/env.sh
cd $HOME/ml-learning-to-evict
export RULER_LEN=$L PYTORCH_ALLOC_CONF=expandable_segments:True
LAB=/scratch2/$USER/kvp/alloc_labels_L$L; OUT=alloc_out_L$L; IDS=alloc_ids_L$L.json
uv run python alloc/label.py --splits $IDS --split test --composite config_composites/ruler845_b128.yaml --out $LAB || exit 1
D="uv run python alloc/train_alloc.py dump --splits $IDS --labels $LAB --split test --source ckpt --ratios 0.05 0.1 --out_dir $OUT"
$D --ckpt alloc_ckpt/input_hw_s2/best.pt --name winput || exit 1
$D --ckpt alloc_ckpt/fixed_hw_s2/best.pt --name wfixed || exit 1
if [ "$L" = 4096 ]; then
  $D --ckpt alloc_ckpt/input_hw_s2/best.pt --name winput_shufcross --shuffle_feat || exit 1
  $D --ckpt alloc_ckpt/input_hw_s2/best.pt --name winput_shufsame --shuffle_feat --shuffle_same || exit 1
fi
ls -la $OUT
T=alloc_tasks_L$L.txt; : > $T
for f in $OUT/*.json; do echo "--alloc $f" >> $T; done
for r in 0.05 0.1; do
  echo "--uniform $r" >> $T
  echo "--team_static $HOME/kvp-team/rl_runs/plan/allocs_blend/shareplan_snapkv_t$r.json" >> $T
  echo "--team_static $HOME/kvp-team/allocations/profile-share_snapkv_t$r.json" >> $T
done
echo "eval tasks:"; cat -n $T
