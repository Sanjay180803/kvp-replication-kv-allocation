# RULER-4K task LOTO: KVP-KL-weighted allocator

Queue the whole experiment once with `bash loto/submit_pipeline.sh` from the
parent CARC experiment folder. Nothing is submitted by local setup or tests.

**Completed run:** `run_AJ4zER` on CARC. Downloaded summaries, fold records,
allocations and predictions are in `../results/loto_run_AJ4zER/`. Verified
RULER-4K results (5% / 10%): static **69.93 / 82.04**, fixed **69.87 / 82.65**,
input **71.03 / 81.69**. Input minus fixed is +1.17 [-1.14, +3.51] / -0.96
[-3.01, +1.04]; neither paired 95% interval excludes zero. No new submission is
needed to reproduce the existing summary from these downloaded predictions.

## Wenbo's actual protocol

Verified from his downloaded `splits.py`, `train_alloc.py`, `head_weights.py`,
`slurm/after_loto.sh` and training logs (including `lane_24102917.log`):

- All **13 RULER tasks** are held out in turn; these are task folds, not random folds.
- Seed **0** per fold, fixed policy up to **3,000** steps, input-aware policy up to
  **10,000** steps with patience **20**, initialized from that fold's fixed checkpoint.
- Train/val IDs are the original split minus the held-out task. Checkpoint choice
  and feature normalization use only that fold's remaining train/val data.
- The held-out task is also removed from the KL profiling weights and from the
  training-label subset used for attention-loss normalization.
- Each fold allocates its held-out task's **20 original test prompts**. Thirteen
  disjoint outputs merge into the original **260-prompt** test set.
- He dumps and evaluates **weighted static split, fixed allocator, input-aware
  allocator**, at **5% and 10%** memory. The fixed allocator has fixed calibration
  parameters but still responds to each prompt's KVP scores.

This setup retains his allocator, RLOO updates, learning rates, training budgets,
stopping rule, feature extraction, reward normalization and exact token budgets.
Each fold gets a derived train/val/test JSON rather than the vendor `--loto`
flag; the checked IDs are exactly those in Wenbo's saved `loto[task]` entries.
Separate fold directories keep reference caches and checkpoints isolated.

The intended change is **KVP-derived KL head weights**. CARC labels were regenerated
and are not claimed to be numerically identical to Delta's. Also, our KVP profile
has 65 prompts per weighting level, while the downloaded SnapKV CSVs have only
nine at 2%/5%. For an ablation that isolates ranking, supply those exact SnapKV
CSV files in the parent's `MATCH_SNAPKV_CSVS`; they are snapshotted and used to
match prompt/ratio coverage before excluding the held-out task. Confirm those
CSV inputs with Wenbo before claiming exact reproduction of his saved weights.
With the default full-KVP profile, each fold uses 60 calibration prompts per level.

`min_k=1` is the KL profiling eligibility threshold, not a memory budget.
Evaluation total budget is always `112 * floor(ratio * prompt_length)`, including
four sink and sixteen recent tokens per head, as in Wenbo's code. Training still
uses his eight-budget grid spanning 2–25%; KL weight construction uses 2/5/10/25%.

## Queued stages

```text
End-to-end smoke
    -> 260-prompt test feature extraction
    -> 13 fold jobs (array, at most two at once)
    -> six RULER evaluations (static/fixed/input x 5%/10%)
    -> CPU summary and paired-bootstrap comparisons
```

Every link uses `afterok` and `--kill-on-invalid-dep=yes`. Full evaluation waits
for all folds; the summary waits for all six evaluations. The queue script
prints the pipeline path immediately and records each submitted job incrementally,
so `submitted_jobs.txt` is available even if a later submission fails.

The smoke holds out task 0, computes real fold-specific head weights, trains two
steps per policy on four non-held-out TRAIN and four non-held-out VAL prompts,
dumps all three methods at both ratios, then evaluates input/10% on two held-out
test prompts. It checks mechanics, not scientific performance. It starts from
scratch and does not use any all-task allocator checkpoint.

Test feature extraction follows Wenbo's **full-answer generation and label
forward pass**, to preserve forward shapes. Only prompt KVP scores/ranks are
saved. Future-attention targets and reference answers are not allocator inputs;
vendor checkpoint dumping uses `need_fut=False`. Model/checkpoint identity,
dataset fingerprint, PyTorch version and key labeling source hashes must match
the completed TRAIN/VAL labeling run.

## Launch on CARC

Upload and extract the latest parent experiment bundle. Existing `config.sh`
and completed TRAIN/VAL labels are reused; do not regenerate the labels.

```bash
cd ~/kvp-kl-allocator-carc
# Optional overrides: seed 0, concurrency 2, and stage time limits are the defaults.
cp -n loto/settings.example.sh loto/settings.sh
nano loto/settings.sh
bash loto/submit_pipeline.sh
squeue -u chenpaul
```

Uses `ctorng_1735`, partition `gpu`, A40 by default; CPU summary uses `main`.
Runtime limits: smoke 25 minutes, features 1 hour, each fold 45 minutes, each
evaluation 1 hour, CPU summary 10 minutes. These are unbenchmarked upper limits,
not expected runtimes. In total, the GPU stages request at most **17h 10m of
GPU runtime** if every task uses its entire limit; actual use ends with each task.
Array concurrency is two, independently of any other jobs you have queued.

At submission, code, configuration, split, KVP profile, composite and optional
coverage-matching CSVs are copied into a new pipeline directory. Queued stages
use that frozen copy. The setup does not change any existing all-task training job.

## Outputs and interpretation

Under `/scratch1/chenpaul/kvp_kl_allocator/loto_runs/run_XXXXXX/`:

- `jobs.json`, `submitted_jobs.txt`: stage job IDs; logs remain in the parent's `logs/`.
- `code/`, `base_config.sh`, `settings.sh`, `splits.original.json`, `profile.csv`:
  submitted source and settings (keep private config out of Git).
- `features/`: test scores/ranks and their provenance; no test reward targets.
- `folds/INDEX_TASK/`: excluded-task record, fold weights, train/val IDs,
  checkpoints, training summaries and 20-prompt allocations.
- `merged/`: each method/ratio's validated, exact-budget 260-prompt allocation.
- `eval/`: per-prompt predictions, scores, coverage and inference provenance.
- `summary.json`, `summary.md`: means and paired 5,000-resample bootstrap CIs
  for input minus fixed/static. The resampling unit is the paired test prompt;
  it does not measure training-seed uncertainty.

Copy back small manifests, summaries, logs and evaluation JSONL/allocations for
review. Keep large label/feature tensors, caches and model weights off Git.

Smoke log: `logs/loto_smoke_JOB_ID.log`; feature log:
`logs/loto_features_JOB_ID.log`; fold/evaluation array logs:
`logs/loto_ARRAY_JOB_ID_TASK_INDEX.log`; summary log:
`logs/loto_summary_JOB_ID.log`.

This implements **4K** task LOTO to match Wenbo's original protocol, not 8K/16K
LOTO or a new train/test split. KVP itself was trained on the RULER task types:
the unseen-task claim applies to the allocator, not the complete model system.
CPU tests cover exclusion invariance, exact split validation, missing-fold and
budget rejection, the future-free feature file contract and mocked Slurm
dependencies. The CARC GPU pipeline completed successfully, including its
end-to-end smoke, all 13 fold jobs and six evaluations. Downloaded evidence was
independently checked for coverage, fold exclusions, token budgets, means and
bootstrap intervals. A fresh reproduction still starts with its own smoke.
