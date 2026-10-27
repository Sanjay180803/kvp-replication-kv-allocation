# Run log (USC CARC, account rashidin_1753)

Every Slurm job we ran, in order, with what happened.

| Job ID | Script | What | Result |
|---|---|---|---|
| 12969801 | gpu_test.sh | GPU check | OK: NVIDIA L40S, torch.cuda True |
| 12971412 | preprocess.sh | RULER-4K chunk 0/100 | OK: 65 samples (all `niah_single_1`, since the dataset is ordered by task), 58 GB |
| 12971771 | train_test.sh | 1 agent (L15 H2), 200 steps, 60 train / 5 val | OK: eval_reward −10.05 → −6.84; 2.45 steps/s |
| 12971914 | preprocess_array.sh | Chunks 8, 16, …, 93 (one per remaining task) | OK: 845 samples total, ~760 GB |
| 13025161 | train_array.sh (batch 32) | Pilot: layer 0, 4 heads | CUDA OOM (another process on the GPU; then OOM at step 1029). Slurm still showed COMPLETED, so scripts now propagate the exit code |
| 13055785 | train_array.sh | Pilot rerun | OOM again: the edited script hadn't been saved (still batch 32) |
| 13077430 | train_array.sh (batch 16) | Pilot: layer 0, 4 heads, 4,000 steps | OK: eval_reward ≈ −3.0; ~1h04 on L40S, 1h26 on A40 |
| 13087071 | train_layer_array.sh --array=1-27%8 | Layers 1–27 (108 agents) | OK: all COMPLETED, ~4h10–5h40 per layer → 112/112 agents |
| 13150534 | eval_array.sh (smoke, KVP) | 13 examples | FAILED: flex-attention "out of resource" (shared memory) on L40S/A40 → kernel_options patch |
| 13151197 | eval_array.sh (smoke, KVP) | 13 examples | OK: 1–10 s per example |
| — | eval_array.sh --array=0-2 (smoke) | full / streaming / snapkv, 13 examples | OK: full 92.3, confirming pipeline and scoring |
| — | eval_array.sh --array=0-4 | Full eval: 5 methods × 260 examples × 3 budgets | OK: see README §5 |
| 13167265 | diag.sh → diag_rewards.py | Training reward, 32 agents | KVP+sw −7.79, StreamingLLM −10.27, random −34.67 |
| 13203342 | diag.sh → diag_rewards2.py | + SnapKV and oracle, 16 agents | KVP+sw −7.86, StreamingLLM −8.09, SnapKV −2.68, oracle −1.00 → under-trained |
| 13352518 | train_b128.sh --array=0-1 | L16H1 and L8H3 at batch 128 | OK: −26.75 → −6.77 and −8.81 → −4.06; ~6h45 each |

## Gotchas we hit

- **uv on the login node** crashes (thread limit, OS error 11). Run `uv sync` on a compute node via `salloc`.
- **OpenBLAS pthread errors** on the login node: `export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- **`kagglesdk` ImportError** (0.1.24 is yanked): pin with `uv add` (see README). `uv pip install` gets reverted by the next `uv run`.
- **P100 GPUs** (sm_60) are unsupported: always `--constraint=a40|a100|l40s`.
- **`PROJECT_ROOT` not set** inside jobs: export it (see `slurm/env.sh`).
- **Submit limit:** a 108-task array was rejected (`QOSMaxSubmitJobPerUserLimit`), so we use one job per layer.
- **Logs** are written to the directory you ran `sbatch` from (the repo root).
- **`squeue --start`** shows only *pending* jobs. In `sacct`, the `.batch` / `.extern` rows belong to the same job.
- **Reruns** create a new timestamped folder per head. Before building the config, each `kv_head_XXX` must have exactly one run folder:
  `for d in agents/grouped/ruler845_v1/layer_*/kv_head_*; do n=$(ls "$d" | wc -l); [ "$n" -ne 1 ] && echo "$d has $n runs"; done`
