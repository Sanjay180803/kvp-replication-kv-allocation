# Final results (Nov 2026): RL allocation of the KV budget across 112 heads

> **See "FINAL RESULTS (Nov 24, 2026)" at the bottom; it supersedes the earlier sections.**


Qwen2.5-7B-Instruct, RULER, KVP ranking inside each head, 260 test prompts per length
(4K: the usual 260 held-out rows; 8K/16K: first 20 rows per task). Allocators trained at 4K only.
All rows run through Wenbo's evaluation pipeline on CARC (`slurm/walloc_prep.sh`, `slurm/walloc_eval.sh`).

| Method | 4K 5% | 4K 10% | 8K 5% | 8K 10% | 16K 5% | 16K 10% |
|---|---|---|---|---|---|---|
| Uniform | 28.3 | 43.9 | 32.1 | 45.0 | 35.9 | 50.0 |
| Profile-share (static) | 66.1 | 78.2 | 65.1 | 75.9 | 68.9 | 76.3 |
| Shareplan (static, learned) | 68.5 | 77.8 | 68.0 | 76.1 | 70.2 | 77.2 |
| RL allocator, fixed (wfixed) | 70.5 | 82.1 | 67.9 | 78.2 | 71.0 | 81.2 |
| **RL allocator, input-aware (winput)** | **71.2** | **83.9** | **68.1** | **79.0** | **72.1** | **81.4** |

Paired bootstrap, winput minus (95% CI):
- shareplan: 4K +2.7 [-0.1, +5.6] / **+6.2 [+2.7, +9.9]**; 8K +0.1 / +2.9 [-0.4, +6.6]; 16K +1.9 / **+4.2 [+1.1, +7.5]**
- wfixed: 4K +0.7 / +1.9 [-0.4, +4.4]; 8K +0.2 / +0.8; 16K +1.1 / +0.2 (all n.s.)

Feature-shuffle control (4K), winput minus shuffled-feature version:
- other task: 5% +1.1 [-0.5, +2.9], 10% **+2.4 [+0.1, +4.9]**
- same task: 5% +0.1 [-1.7, +1.9], 10% -0.3 [-1.4, +0.6]

**Reading:** the RL allocator beats the static splits (significant at 10% at 4K and 16K), and the gain holds at 8K/16K
although it was trained at 4K. Explicit input features add no significant gain over the fixed allocator, and a same-task
shuffle has no effect, so the features mainly encode task type. Our cost-model planner shows the same pattern
(`scripts/kv_feat_plans.py`: same-task shuffle n.s.; `scripts/kv_loto.py`: leave-one-task-out gives no gain).

## Our pipeline (SnapKV ranking), for reference
| | full | uniform | profile-share | shareplan | blendplan |
|---|---|---|---|---|---|
| 8K 5% / 10% | 89.8 | 39.3 / 50.0 | 63.9 / 79.6 | 68.4 / 80.4 | 69.6 / 81.4 |
| 16K 5% / 10% | 89.3 | 53.7 / 62.3 | 70.8 / 84.3 | 76.6 / 85.7 | 76.6 / 86.9 |

Different pipeline and ranker from the table above, so don't compare across the two tables.

## Offline RL: corrected numbers (supersede the earlier table)
Bug: in `kv_offline_rl.py::evaluate()`, the off-policy-evaluation samples were inserted into the candidate list in the middle
of the deterministic candidates, so some true-KL values were assigned to the wrong policy. Fix: all deterministic
candidates first, then the samples in policy order. Buggy version kept as `scripts/kv_offline_rl_buggy.py`;
rerun with `slurm/offline_eval_fix.sh`.

| Policy | Buggy | Fixed | Critic gap (fixed) |
|---|---|---|---|
| static | 2.351 | 2.420 | +0.47 |
| AWR | 2.354 | 2.420 | +0.45 |
| DPO | 1.579 | 2.300 | -1.02 |
| MBPO κ=1 | 1.033 | 1.772 | +3.13 |
| MBPO κ=0 | 1.257 | 2.151 | +3.16 |

Conclusion unchanged: no offline-RL policy beats the static split; MBPO exploits the critic (large positive gap).

## New scripts
- `scripts/kv_plan.py`: alpha tuning, blend/share dump, vectorized greedy planner (~5000x faster, identical output; `scripts/check_plan.py`)
- `scripts/time_overhead.py`: planner overhead (1.5 / 3.1 ms per prompt)
- `scripts/kv_feat_plans.py` + `slurm/kv_feat_plans.sh`: feature-shuffle controls and 8K/16K allocations
- `scripts/kv_loto.py` + `slurm/kv_loto.sh`: leave-one-task-out
- `scripts/eval_ruler_alloc.py`: `--scorer kvp --composite`, `--length {4096,8192,16384}`
- `patches/wenbo_alloc_carc.patch`: `RULER_LEN` env var and `--shuffle_same` for Wenbo's allocator

## Ablation: is the RL calibration needed? (RULER, 260 prompts)
"raw" = global top-B over each head's softmaxed KVP scores, with no learned per-head calibration.

| | 4K 5% | 4K 10% | 8K 5% | 8K 10% | 16K 5% | 16K 10% |
|---|---|---|---|---|---|---|
| raw (no calibration) | 20.1 | 37.9 | 20.5 | 39.1 | 23.1 | 42.4 |
| uniform | 28.3 | 43.9 | 32.1 | 45.0 | 35.9 | 50.0 |
| wfixed (RL-calibrated) | 70.5 | 82.1 | 67.9 | 78.2 | 71.0 | 81.2 |

wfixed − raw: +39 to +50 points, significant everywhere. Without calibration, a few heads with sharp scores take the
budget and global top-B is worse than uniform. wfixed − shareplan at 10%: 4K **+4.3 [+1.1, +7.9]**, 8K +2.1 [−1.0, +5.6],
16K **+4.0 [+0.7, +7.5]**; at 5%: +2.0 / −0.1 / +0.8 (n.s.).

## Downstream: LongBench (4K, 6 English tasks × 40 = 240 prompts, all unseen by the allocators)
Qasper, MultiFieldQA-en, HotpotQA, 2WikiMQA, PassageRetrieval-en, TREC; official prompts and metrics, middle-truncated to
3,900 tokens (`scripts/lb_data.py`, `slurm/walloc_lb.sh`). Full cache = **46.2**.

| | 5% | 10% |
|---|---|---|
| uniform | 29.7 | 32.4 |
| raw | 29.2 | 32.4 |
| profile-share | 33.5 | 41.0 |
| shareplan | 34.6 | 40.3 |
| **wfixed** | **36.5** | 40.6 |
| winput | 31.6 | 39.1 |

Paired (95% CI): wfixed − profile-share 5% **+3.1 [+0.2, +6.1]**, 10% −0.4 [−3.3, +2.3]; wfixed − shareplan 5% +2.0 [−1.3, +5.2],
10% +0.3; wfixed − winput 5% **+4.9 [+1.7, +8.4]**, 10% +1.5 [−1.8, +4.9]. The input-aware allocator loses on unseen tasks
(collapses on PassageRetrieval and TREC at 5%), consistent with the same-task shuffle: its features encode training task type.
Full numbers: `results/walloc/summary_and_ci.txt`.

---
# FINAL RESULTS (Nov 24, 2026) — supersede all sections above
KVP ranking inside heads; RULER 260 prompts per length, LongBench 240 (6 tasks, unseen); fixed/input = 3-seed means;
paired bootstrap 95% CI, * = significant. Full output: `results/walloc/final_ci.txt`.

| 5% / 10% | 4K | 8K | 16K | LongBench |
|---|---|---|---|---|
| uniform | 28.3 / 43.9 | 32.1 / 45.0 | 35.9 / 50.0 | 29.7 / 32.4 |
| raw global top-B (no head weights) | 20.1 / 37.9 | 20.5 / 39.1 | 23.1 / 42.4 | 29.2 / 32.4 |
| profile-share (static) | 66.1 / 78.2 | 65.1 / 75.9 | 68.9 / 76.3 | 33.5 / 41.0 |
| shareplan (static) | 68.5 / 77.8 | 68.0 / 76.1 | 70.2 / 77.2 | 34.6 / 40.3 |
| wstatic (static, SnapKV-KL weights) | 69.5 / 80.9 | 67.5 / 78.5 | 72.4 / 81.1 | 37.1 / 40.5 |
| kstatic (static, KVP-KL weights) | 69.3 / 84.3 | 69.7 / 80.0 | 72.9 / 81.9 | 38.4 / 43.2 |
| RL fixed, SnapKV-KL (wfixed) | 69.7 / 81.3 | 67.2 / 78.2 | 71.4 / 80.9 | 35.9 / 40.5 |
| RL input, SnapKV-KL (winput) | 70.7 / 84.2 | 68.3 / 78.6 | 73.0 / 80.2 | 31.2 / 38.1 |
| RL fixed, KVP-KL (kfixed) | 72.0 / 84.2 | 70.3 / 79.9 | 73.7 / 83.2 | 38.3 / 43.9 |
| RL input, KVP-KL (kinput) | 73.9 / 86.5 | 71.0 / 80.0 | 72.8 / 80.4 | 36.9 / 42.1 |

Full cache: RULER-4K 92.8, LongBench 46.2.

1. **The head split dominates.** Without head weighting, global top-B is below uniform (wfixed − raw +39 to +50*).
2. **The importance weights matter.** KVP-KL (same ranker as inference) beats SnapKV-KL: kstatic − wstatic 4K 10% +3.4*, 8K 5% +2.2*, LB 10% +2.7*; kfixed − wfixed significant in 5 of 8.
3. **With matched weights, RL-fixed equals the static fit:** kfixed − kstatic ties in 7 of 8 (4K 5% +2.7*), consistent with LOTO (−0.06 / +0.60).
4. **Input-aware RL beats the best static split only at the training length:** kinput − kstatic 4K +4.6* / +2.2*; 8K/16K n.s.; LongBench −1.6 / −1.0; winput − wfixed LongBench 5% −4.7*. Same-task shuffle and LOTO are null → it learns training-distribution cues.

**Overhead.** The allocator's decision step (per-head calibration + one global top-B over 112 × P KVP scores) takes
0.36 ms at 4K, 0.39 ms at 8K and 0.42 ms at 16K on an A40 (20 prompts per length; `results/walloc/time_alloc.txt`),
about 0.06% of the 616 ms 4K prefill. KVP scores are computed for eviction regardless.
