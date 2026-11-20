# Final results (Nov 2026): RL allocation of the KV budget across 112 heads

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
