# CARC profiling results

Use **`profile_14372014/` (`min_k=1`) as the primary profile**: all 65 prompts
completed, with five per task, no skips and 36,400 measurements. Its log is
`kvp_kl_14372014.log`; profiler runtime was 810.4 seconds on an A40.

`profile_14364217/` preserves the original `min_k=3` run: 61 completed prompts,
four explicit skips and 34,160 measurements. Its log is `kvp_kl_14364217.log`.
All 34,160 shared CSV rows are identical between the runs. The new run adds
IDs 1900, 1901, 1904 (`qa_1`) and 4402 (`qa_2`), each with K=2.
`../run_record.json` records the coverage and available execution evidence.

The folder includes `head_kl.csv`, `coverage.json`, `experiment.json`,
`preflight.json`, per-prompt JSON files, analysis summaries, and PNG plots.
The smoke output/log may be attached separately if available.

Keep the original filenames and metadata so code, settings, checkpoint identity,
prompt IDs, and results can be checked together. Do not add model weights,
KVP checkpoint binaries, environments, or the Hugging Face cache.
