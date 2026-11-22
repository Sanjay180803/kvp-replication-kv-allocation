# KVP KL allocator: RULER-4K leave-one-task-out

13 folds, one seed per fold, 20 held-out test prompts per task.

| Policy | 5% | 10% |
| --- | ---: | ---: |
| static | 69.93 | 82.04 |
| fixed | 69.87 | 82.65 |
| input | 71.03 | 81.69 |

Input minus fixed:
- 0.05: +1.17, paired-bootstrap 95% CI [-1.14, +3.51]
- 0.1: -0.96, paired-bootstrap 95% CI [-3.01, +1.04]

KVP itself saw the RULER task types during its earlier training. This tests allocator transfer, not whole-system unseen-task training.
Confidence intervals cover paired test-prompt resampling, not training-seed uncertainty.
