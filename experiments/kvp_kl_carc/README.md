# KVP head profiling on CARC (task E)

This experiment asks: **if one KV head keeps only X% of its prompt tokens,
chosen by the trained KVP policy, how much does Qwen's output change?**

Qwen2.5-7B-Instruct and all 112 KVP agents stay frozen. We compress one KV head
at a time, leave the other heads at full cache, and measure mean KL divergence
from an in-batch full-cache control over up to 32 continuation positions.
Lower KL means less output-distribution damage. This is not answer accuracy.

## What is in this folder?

This is a self-contained profiling snapshot for review and reproduction.
The five Python files in `scripts/` and the checkpoint map in `configs/` were
copied byte-for-byte from the local source archive uploaded for these runs.
`SOURCE_MANIFEST.json` records their hashes. The CARC entry points were organized
after the run: they use this folder's local scripts and CARC paths rather than
the original Delta folder layout. This is not a download of the server checkout;
attach its original `experiment.json` and logs to verify exact runtime provenance.

| File | Purpose |
| --- | --- |
| `scripts/profile_heads.py` | Capture prompt K/V, apply per-head eviction, measure KL |
| `scripts/kvp_ranker.py` | Load frozen Apple KVP policies and produce token rankings |
| `scripts/profiling_io.py` | Validate measurements, save prompt shards, support resume |
| `scripts/validate_profile.py` | Check result completeness and valid measurements |
| `scripts/analyze_profiling.py` | Produce heatmaps, correlations and a summary |
| `configs/ruler845_b128.yaml` | Map all 112 agents; remap old paths using `CHECKPOINT_ROOT` |
| `preflight.py` | Check the model, checkpoints, dependencies and GPU |
| `config.example.sh` | CARC paths, account, GPU type and profiling settings |
| `prepare_env.sh`, `requirements.txt` | Build the Python 3.11/3.12 environment |
| `submit.sh`, `profile.sbatch`, `run.sh` | Submit and execute smoke/full runs |
| `run_record.json` | Recorded CARC job IDs, timing and coverage |
| `results/` | Place the original downloaded outputs here |

The broader task E changes to `scripts/rl_alloc.py` and the planner/offline-RL
entry points remain in the repository's root `scripts/` directory. They were
not exercised by this profiling job. This snapshot does not train an allocator
or rebuild the planner's training cost tables.

## What was run?

Both CARC jobs completed with exit code `0:0` on the reported setup:

| Run | Job | Scope | Elapsed |
| --- | --- | --- | --- |
| Smoke | 14364197 | One selected prompt, heads (0,0), (4,1), (22,0), five ratios, plus sanity checks | 2m 39s |
| Full | 14364217 | 65 selected prompts, all 112 KV heads, five ratios | 14m 06s |

Model: Qwen2.5-7B-Instruct, bf16. GPU: NVIDIA A40. Software reported by the log:
PyTorch 2.10.0+cu128 and Transformers 4.57.1. Configured batch size: 8, including
one control item (up to seven compressed variants per batch).

Prompts are RULER-4K offsets 400–404 within each of 13 task blocks.
The five keep ratios are 2%, 5%, 10%, 25%, and 50% **within the selected head**.
These are not whole-model memory budgets.

The full run produced **34,160 measurements = 61 prompts × 112 heads × 5 ratios**.
Four prompts (1900, 1901, 1904, 4402) had only two measured continuation
positions and were explicitly skipped because `min_k=3`. Coverage was complete
with no pending prompts. The completed dataset is therefore not task-balanced.

The profiler caches prompt K/V once per prompt and reuses them for the short
continuation passes. It does not recompute the whole prompt for every variant.
Scorer-specific sanity checks run before measurement.

## Reproduce on CARC

From this folder, copy `config.example.sh` to `config.sh` and edit the paths.
You need your own readable Qwen model, all 112 trained KVP checkpoints with their
layer/head directory hierarchy, Apple's `ml-learning-to-evict` checkout, and a
Python environment with the listed requirements. These large external assets
are not included here. An existing CARC profiling environment can be reused.

If an environment needs installing, with Python 3.12 available:

```bash
cp -n config.example.sh config.sh
# Edit config.sh before continuing.
BOOTSTRAP_PYTHON=/path/to/python3.12 bash prepare_env.sh
```

Submit a smoke test, then a full profile after it succeeds:

```bash
bash submit.sh smoke
bash submit.sh profile --dependency=afterok:SMOKE_JOB_ID --kill-on-invalid-dep=yes
```

Replace `SMOKE_JOB_ID` with the submitted smoke job's number. Logs go to
`logs/kvp_kl_JOB_ID.log`; outputs go to
`$EXPERIMENT_ROOT/runs/smoke_JOB_ID` or `profile_JOB_ID`.
Each new job creates a separate output directory unless `KVP_OUT_DIR` is set.
Resume accepts only matching recorded settings and source fingerprints; keep
using the original server checkout when resuming its historical output.

## Attach results and use them

The downloaded `profile_14364217/` folder and full job log are attached in
`results/`, including the CSV, prompt shards, metadata, coverage and plots.
The attached coverage confirms the counts reported above.

The main table is `head_kl.csv`: prompt/task, layer/KV head, keep ratio, mean/max
KL, top-1 agreement, prompt length, tokens evicted and measured positions.
Use these curves to estimate head importance under KVP ranking. Compare with
the prior SnapKV profile on matched prompt IDs and normalization. Individual
head costs need not add up when many heads are compressed together.

Commit this folder plus the attached small result artifacts. Keep `config.sh`,
model/checkpoint binaries, environments and caches out of Git.
