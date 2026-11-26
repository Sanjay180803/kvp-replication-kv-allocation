#!/bin/bash
#SBATCH --account=skadiri_1836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/walloc_kstatic_exact_%j.out
source $HOME/ml-learning-to-evict/slurm/env.sh
cd $HOME/ml-learning-to-evict
export OPENBLAS_NUM_THREADS=1
SI=/project2/rashidin_1753/chenpaul/kvp_kl_allocator3/static_inputs
HWF=$(find $SI -name "*head_weight*.pt" | head -1)
SPL=$(find $SI -name "*.json" | xargs grep -l '"train"' | head -1)
LABD=$(dirname "$(find $SI -name '*.pt' ! -name '*head_weight*' | head -1)")
echo "head weights: $HWF"; echo "splits: $SPL"; echo "train labels: $LABD ($(ls $LABD/*.pt | wc -l) files)"
[ -n "$HWF" ] && [ -n "$SPL" ] && [ -d "$LABD" ] || { echo "could not find inputs"; exit 1; }
uv run python alloc/static_plan.py fit --splits $SPL --labels $LABD --head_weights $HWF --out static_curves_kvp.pt || exit 1
for L in 4096 8192 16384 LB; do
  uv run python alloc/static_plan.py plan --curves static_curves_kvp.pt --labels /scratch2/$USER/kvp/alloc_labels_L$L/test \
     --ids alloc_idlist_L$L.json --ratios 0.05 0.1 --name kstatic_exact --out_dir alloc_out_L$L || exit 1
  printf -- "--alloc alloc_out_L$L/kstatic_exact_r0.05.json\n--alloc alloc_out_L$L/kstatic_exact_r0.1.json\n" > alloc_tasks_kstatic_exact_L$L.txt
done
