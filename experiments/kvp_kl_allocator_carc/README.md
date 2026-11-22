# Train Wenbo's allocator with KVP KL head weights on CARC

**Task leave-one-out training and evaluation:** see [`loto/README.md`](loto/README.md).
`bash loto/submit_pipeline.sh` queues an independent end-to-end smoke, test-feature
extraction, 13 fold jobs, six 4K evaluations and a summary with success dependencies.

This is a new training experiment. It replaces the SnapKV-derived KL head
importance in Wenbo's attention reward with our KVP-derived KL measurements.
The allocator architecture, RLOO training, label selection and weight normalization
are preserved. CARC label smoke **14417065** and full label job **14417159**
completed successfully with the A40 FlexAttention workaround. All **1,164 TRAIN
and 65 VAL labels** are available on CARC. All-task training for seeds 0–2 and
the full 13-task LOTO training/evaluation pipeline completed successfully.
**All-task checkpoint RULER evaluation remains pending**; the scores below are
LOTO scores, not all-task scores.

## Verified LOTO results (RULER-4K)

Downloaded evidence is in `results/loto_run_AJ4zER/`. Each fold trained on the
other 12 tasks and tested on its excluded task's 20 original test prompts.
All folds used seed 0, with the held-out task excluded from profiling weights,
attention-loss normalization, training and validation.

| Method | 5% memory | 10% memory |
| --- | ---: | ---: |
| Weighted static split | 69.93 | 82.04 |
| Fixed allocator | 69.87 | 82.65 |
| Input-aware allocator | 71.03 | 81.69 |

Input minus fixed: **+1.17** points, paired 95% CI **[-1.14, +3.51]** at 5%;
**-0.96**, **[-3.01, +1.04]** at 10%. Input minus static is also not significant:
+1.10 [-1.58, +3.78] / -0.35 [-2.34, +1.72]. These results do not establish an
input-aware advantage on unseen task types; they do not prove equivalence.

Local checks independently reproduced all six means and all four paired
5,000-resample bootstrap intervals, verified 260 unique IDs (20/task) per
evaluation, exact token budgets, all fold exclusions, 60 calibration prompts
and 260 non-held-out training-label normalization IDs per fold. The profile hash
matches the completed 65-prompt `min_k=1` KVP profile. No coverage-matching mask
was enabled. `verification.json` records the checks. Original test-feature and
fold checkpoint tensors remain on CARC and were not downloaded for this audit.
Intervals reflect test-prompt resampling with one training seed per fold.

## Completed all-task training and checkpoint handoff

All-task jobs `14444657_0`, `_1`, `_2` completed with reported Slurm exit `0:0`,
in 12m30s, 13m16s, and 13m41s. Each trained one fixed and one input-aware
allocator; input seed K starts from fixed seed K. Their six checkpoints still
need matched RULER evaluations (4K/8K/16K) before comparing all-task versus LOTO
or KVP-KL versus SnapKV-KL performance.

Project handoff destination supplied for sharing on CARC:

```text
/project2/rashidin_1753/chenpaul/kvp_kl_allocator3/
  checkpoints/fixed_hw_s{0,1,2}/best.pt
  checkpoints/input_hw_s{0,1,2}/best.pt
  metadata/seed_{0,1,2}/
```

Checkpoint files and remote group permissions were not inspected locally;
confirm all six files are readable before another account evaluates them.
Use paired seeds and the same inference pipeline/IDs/budgets as the baseline.

## Verified label-generation results

Original logs, coverage, split, provenance and per-prompt diagnostics are attached
in `results/labels_14417159/`; `verification.json` records the local checks.

| Split | Completed | Missing/duplicate IDs | Future positions, min / median / max |
| --- | --- | --- | --- |
| TRAIN | 1,164 | 0 / 0 | 1 / 14 / 80 |
| VAL | 65 | 0 / 0 | 1 / 15 / 75 |

All expected IDs and task labels match Wenbo's exact split. The input split
hash and all four recorded generation source hashes match this experiment.
There are no prompts with zero future-attention total. Every validation task
has five prompts; the six original training exclusions remain intact. Logged
per-prompt durations sum to 4,193.54 seconds (about 70 minutes), excluding other
job overhead; a separate Slurm elapsed-time record has not been attached.

The job log confirms validated tensor shapes, rankings, protected tokens and
finite values before successful completion. Only the small records were
downloaded: the `.pt` tensors remain on CARC and were not independently reloaded
locally. The `full_score` values are full-cache Qwen diagnostics on TRAIN/VAL,
not allocator scores or held-out test results.

## Reproduce all-task allocator training (already completed)

Do not resubmit the completed jobs merely to obtain test accuracy. Their next
step is allocation dumping and inference evaluation. To reproduce training, no
label regeneration or new dependency installation is needed. In your
existing CARC folder, check the paths and coverage, then submit a training smoke:

```bash
cd ~/kvp-kl-allocator-carc
export KVP_ALLOC_EXPERIMENT_DIR="$PWD"
source config.sh
echo "$LABELS_DIR"
cat "$LABELS_DIR/label_coverage.json"
SMOKE_JOB_ID=$(bash submit.sh smoke)
echo "Training smoke: $SMOKE_JOB_ID"
```

`LABELS_DIR` should be `/scratch1/chenpaul/kvp/allocator_data/labels_carc`.
The training smoke builds KVP KL weights and trains each policy for two steps
on four TRAIN and four VAL prompts. It is different from the completed label smoke.
Watch `logs/train_SMOKE_JOB_ID_0.log` for checkpoint validation and final SUCCESS.
You can queue the three full training seeds behind its success immediately:

```bash
bash submit.sh train --dependency="afterok:$SMOKE_JOB_ID" --kill-on-invalid-dep=yes
squeue -u chenpaul
```

The array runs seeds 0–2, each on one A40 with a 45-minute maximum and 64 GB
host memory. Each seed trains fixed then input-aware in its own work directory.
The combined requested maximum is 2h 15m for full training plus 15m for the
training smoke, excluding the already-completed labeling jobs. Total elapsed
wall time depends on scheduling and whether seeds run concurrently.

## Inputs and pipeline

1. Use Wenbo's exact `data/splits.json` (included in `inputs/`) and regenerate
   TRAIN/VAL labels on CARC using the existing Qwen and KVP checkpoints.
2. Convert our complete `min_k=1` profile (65 prompts, 36,400 rows) into 112
   importance weights. Following Wenbo, use 2%, 5%, 10%, 25%; 50% is unused.
3. Train a fixed-parameter allocator for up to 3,000 steps, then initialize the
   input-aware allocator from its best checkpoint and train up to 10,000 steps
   with patience 20. Both use RLOO (16 samples, prompt batch 8), validation
   checkpoint selection and the original default learning rates.
4. Repeat seeds 0, 1, 2 in separate Slurm array jobs. Select a seed by validation
   reward, never by test accuracy. Evaluate its allocations on RULER separately.

Weight definition: mean per-level normalized KL share divided by mean per-level
normalized future-attention-loss share, rescaled to mean 1 across heads. The
denominator uses Wenbo's deterministic subset of 260 TRAIN labels. Weighting
changes the reward, not the frozen KVP token scores used as allocator features.

The fixed allocator has 224 learned parameters (a sharpness and weight per
head). Its parameters are input-independent, but its budgets still change with
the input's KVP scores. It is not a fixed per-head budget split.

## Files

| File | Purpose |
| --- | --- |
| `prepare.py` | Validate saved splits/labels/profile, compute KVP weights and provenance |
| `vendor/wenbo/train_alloc.py`, `alloc_lib.py` | Unchanged training and allocation code from Wenbo's download |
| `vendor/wenbo/head_weights.py`, `splits.py` | Original weight/split generators for provenance; not invoked |
| `config.example.sh` | Existing CARC asset paths plus separate data/output paths |
| `submit.sh`, `train.sbatch`, `run.sh` | Smoke or three-seed training on CARC |
| `submit_labels.sh`, `label.sbatch`, `run_labels.sh` | Smoke/full GPU label generation on CARC |
| `label_carc.py`, `label_io.py` | Wenbo's label computation with explicit paths, atomic output and provenance |
| `patch_a40_attention.py` | Apply the team's backed-up, idempotent 64x64 / one-stage FlexAttention patch |
| `inputs/splits.json`, `inputs/ruler845_b128.yaml` | Exact downloaded split and complete KVP checkpoint map |
| `finish.py` | Check saved checkpoints and write `training_summary.json` |
| `build_bundle.py` | Package this code and the measured KVP profile for transfer |
| `tests/test_setup.py` | CPU checks of weighting, split validation and training integration |

Original code comes from the user-downloaded `wenbo-allocator/alloc/` folder;
`SOURCE_MANIFEST.json` records its hashes. This folder uses a new weight builder
because Wenbo's original builder hardcodes `scorer == snapkv` and does not
reject empty/missing profiling levels. The original training code is unchanged.

## Exact splits and CARC label generation

The exact downloaded split is included in `inputs/splits.json`: SHA1
`874e55c37a9a7aa7c87b2939e990c30f07b99355`, 1,164 TRAIN / 65 VAL / 260 TEST.
TRAIN uses offsets 410–499, VAL 405–409, and TEST the team's IDs below 400.
The six training IDs excluded for answer overlap are 2458, 2935, 4420, 4431,
4454 and 4483. The code verifies IDs, exclusions, task mapping, disjointness and
the embedded SHA1. Do not recreate the split or download large labels unless
you prefer Wenbo's original saved labels.

The label job loads the existing four-shard Qwen model and remaps all 112 KVP
checkpoint paths to `CHECKPOINT_ROOT`. It generates the full greedy answer using
the row's generation limit, then captures per-head KVP ranking, raw scores,
future attention and window attention, in rank order, as Wenbo's collector does.
This is not the 32-position KL profiler: it retains Wenbo's full-answer label
policy. Only TRAIN and VAL labels are produced.

Default CARC paths:

```text
Python:      /scratch1/chenpaul/kvp_kl/venv/bin/python
Qwen:        /scratch1/chenpaul/kvp/models/Qwen2.5-7B-Instruct
KVP agents:  /scratch1/chenpaul/kvp/agents/ruler845_b128
Apple code:  ~/ml-learning-to-evict
New labels:  /scratch1/chenpaul/kvp/allocator_data/labels_carc
```

The label manifest records split, source, dataset identity, model/checkpoint
identities and software versions. Validated atomic `.pt` files can be resumed
with the same settings after a timeout. A changed manifest or an old unversioned
label directory is rejected. GPU/software differences can produce numerical
differences from Wenbo's Delta labels; regenerated labels are not claimed to be
bit-identical. Compare reward variants using the same regenerated labels.

The public Apple checkout's default FlexAttention kernel exceeded the A40's
per-block shared-memory limit (114,688 required vs 101,376 available). This is
separate from VRAM exhaustion. Before importing Apple modules, the launcher now
applies the team's existing workaround from `patches/flex_attention_kernel_options.patch`:
`BLOCK_M=64`, `BLOCK_N=64`, `num_stages=1`. It backs up
`src/kvcompression/kv_cache/compressible_attention.py` with suffix
`.before_kvp_alloc_a40`, checks for the expected call, and skips repeat application.
The actual attention source hash is recorded in label provenance. This adjusts
the kernel execution configuration, not the label/reward definition; numerical
equivalence on this GPU is not yet established. No package reinstall is required.

## Transfer this experiment

From the repository root on your Mac:

```bash
python3 experiments/kvp_kl_allocator_carc/build_bundle.py
scp experiments/kvp_kl_allocator_carc/artifacts/kvp-kl-allocator-carc.tar.gz \
    chenpaul@hpc-transfer1.usc.edu:/home1/chenpaul/
```

On **CARC** (this uses a separate folder from the profiler):

```bash
tar -xzf ~/kvp-kl-allocator-carc.tar.gz -C ~/
cd ~/kvp-kl-allocator-carc
cp -n config.example.sh config.sh
nano config.sh
bash submit_labels.sh smoke
```

If updating an earlier extracted bundle, `config.sh` is preserved. Edit it to use:

```bash
SPLITS_JSON="$KVP_ALLOC_EXPERIMENT_DIR/inputs/splits.json"
COMPOSITE="$KVP_ALLOC_EXPERIMENT_DIR/inputs/ruler845_b128.yaml"
LABELS_DIR="$DATA_DIR/labels_carc"
```

The **label smoke** generates two TRAIN and two VAL labels in its own
`$EXPERIMENT_ROOT/label_smoke_JOB_ID` directory and validates shapes, finite values,
rankings and protected tokens. It requests one A40, 96 GB host memory and
15 minutes. This path passed in CARC smoke job 14417065.

After the label smoke succeeds:

```bash
bash submit_labels.sh full
# Or queue behind a successful label smoke:
bash submit_labels.sh full --dependency=afterok:LABEL_SMOKE_JOB_ID --kill-on-invalid-dep=yes
```

The full label job requests two hours and generates all
1,229 labels. Its log is `logs/labels_JOB_ID.log`. Final
`$LABELS_DIR/label_coverage.json` must show complete=true, train=1164 and val=65.
After it succeeds, test policy training:

```bash
bash submit.sh smoke
```

The **training smoke** builds weights from the real labels and runs two optimization
steps per policy on four TRAIN and four VAL examples. Its scores are technical
checks, not scientific results. After it succeeds, submit all three seeds:

```bash
bash submit.sh train
# Or queue behind a smoke job that must succeed:
bash submit.sh train --dependency=afterok:SMOKE_JOB_ID --kill-on-invalid-dep=yes
```

Logs: `logs/train_ARRAY_JOB_ID_SEED.log`. Status: `squeue -u chenpaul`.
Each seed runs fixed then input training sequentially on one A40 with 64 GB host
memory and a 45-minute limit; the completed three seeds took about 13 minutes each.

## Outputs and fair comparison

Each job writes a fresh directory under
`/scratch1/chenpaul/kvp_kl_allocator/runs/train_ARRAY_JOB_ID_sSEED/`:

- `head_weights_kvp.pt`, `splits.json`, `run_config.json` (data/source hashes and settings)
- `work/fixed_hw_sSEED/best.pt` and `work/input_hw_sSEED/best.pt`
- Per-policy validation logs/summaries and `training_summary.json`

Never reuse Wenbo's existing work directory: his reference cache only checks
prompt IDs, not weight identity. Fresh work directories here force new reward
references for the KVP weights and avoid overwriting his baseline checkpoints.
Label identities use paths, sizes and modification times, not full content hashes.

The downloaded baseline profiles contain only nine prompts (three tasks) at
2%/5%, but 65 at 10%/25%. The new KVP profile has 65 at all four levels. To isolate
ranking from profiling coverage, set `MATCH_SNAPKV_CSVS` to the exact baseline
CSV files in `config.sh`; the weight builder restricts KVP prompt/ratio coverage
to their union and records it. Confirm with Wenbo that these are the files used
for his saved weights. Leaving this option empty tests the full new KVP profile,
but the comparison then changes both ranking and coverage.

Compare final RULER answer accuracy at matched budgets. Validation reward under
different head weights is not a comparable accuracy metric. For leave-one-task-out,
the held-out task must also be removed from the profiling weights and training
labels. The parent `submit.sh` remains all-task; the separate `loto/` pipeline
implements and checks those exclusions.

The present launcher covers label generation and allocator training; 4K/8K/16K
inference evaluation is a separate step. Policy training reads saved labels and
does not load Qwen; label generation does.
