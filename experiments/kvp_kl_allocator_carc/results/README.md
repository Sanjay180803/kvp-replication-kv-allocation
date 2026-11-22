# Allocator label-generation results

## Completed LOTO training and evaluation

`loto_run_AJ4zER/` contains 13 fold records/training summaries, six merged
allocations and six 260-prompt RULER-4K evaluations. The full downloaded results
were checked locally: correct held-out-task exclusions, matched original test
IDs, exact budgets, and independently reproduced means/bootstrap intervals.
See its `verification.json` and `summary.md`.

Input-aware LOTO scores: **71.03 / 81.69** at 5% / 10%; fixed allocator:
**69.87 / 82.65**. Neither input-minus-fixed confidence interval excludes zero.
All-task checkpoint inference evaluation is separate and remains pending.

## Label-generation evidence

Put downloaded job logs, `label_coverage.json`, `label_manifest.json`,
`splits.original.json`, and per-split `meta_shard0.jsonl` here under the job ID.
The `labels_14417159/` records are generation records and full-cache Qwen
diagnostics; the separate LOTO folder contains actual allocator evaluations.

Full-label job **14417159**, following smoke job **14417065**, reached final
SUCCESS. Its downloaded coverage confirms all **1,164 training and 65 validation
labels**, no missing IDs and `complete: true`. `labels_14417159/verification.json`
records the independent local checks of IDs, tasks, source/split hashes and
per-prompt diagnostics. The `.pt` tensors themselves remain on CARC; their
validation is recorded by the successful generation code/log.

Keep the large label tensors on CARC at
`/scratch1/chenpaul/kvp/allocator_data/labels_carc/{train,val}/*.pt`.
Do not add tensors, model weights, caches or virtual environments to Git.
