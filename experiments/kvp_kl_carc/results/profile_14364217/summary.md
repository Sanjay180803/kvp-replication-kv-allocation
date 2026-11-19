# Per-KV-head sensitivity profile (kvp)

61 prompts, 13 tasks, keep ratios [0.02, 0.05, 0.1, 0.25, 0.5]. KL(p_full || p_compressed) in nats, mean over 32 teacher-forced continuation positions.

## Top-10 most sensitive heads (r=0.25)
| rank | layer | kv_head | mean KL | top1 agree |
|---|---|---|---|---|
| 1 | 22 | 0 | 5.310e-02 | 0.980 |
| 2 | 4 | 1 | 2.000e-02 | 0.993 |
| 3 | 26 | 0 | 1.433e-02 | 0.994 |
| 4 | 23 | 1 | 7.324e-03 | 0.996 |
| 5 | 27 | 0 | 7.001e-03 | 0.995 |
| 6 | 24 | 3 | 6.607e-03 | 0.994 |
| 7 | 14 | 1 | 6.185e-03 | 0.995 |
| 8 | 8 | 3 | 5.410e-03 | 0.998 |
| 9 | 27 | 3 | 5.330e-03 | 0.997 |
| 10 | 20 | 3 | 5.030e-03 | 0.994 |

## Bottom-10 least sensitive heads (r=0.25)
| rank | layer | kv_head | mean KL | top1 agree |
|---|---|---|---|---|
| 1 | 25 | 0 | 1.896e-05 | 1.000 |
| 2 | 27 | 1 | 3.412e-05 | 1.000 |
| 3 | 23 | 3 | 3.623e-05 | 0.999 |
| 4 | 19 | 1 | 4.177e-05 | 0.999 |
| 5 | 24 | 0 | 4.437e-05 | 0.998 |
| 6 | 3 | 3 | 4.720e-05 | 0.999 |
| 7 | 6 | 3 | 4.748e-05 | 0.999 |
| 8 | 18 | 0 | 4.802e-05 | 0.998 |
| 9 | 7 | 2 | 5.207e-05 | 0.998 |
| 10 | 7 | 3 | 5.364e-05 | 0.998 |

## Input dependence

Average pairwise Spearman between tasks' 112-head importance vectors: **0.169** (min -0.212, max 0.596).

Prompt-level noise floor: same-task prompt pairs average 0.214, cross-task prompt pairs average 0.075. If cross-task is clearly below same-task, head importance depends on the input type.

![task correlation](task_spearman_r0.25.png)

|                 |   niah_single_1 |   niah_multikey_2 |   fwe |   qa_1 |   niah_single_3 |   niah_multikey_1 |   niah_single_2 |   cwe |   qa_2 |    vt |   niah_multivalue |   niah_multikey_3 |   niah_multiquery |
|:----------------|----------------:|------------------:|------:|-------:|----------------:|------------------:|----------------:|------:|-------:|------:|------------------:|------------------:|------------------:|
| niah_single_1   |            1    |              0.16 | -0.01 |   0.12 |            0.3  |              0.21 |            0.07 | -0.12 |   0.05 |  0.02 |              0.35 |              0.19 |              0.05 |
| niah_multikey_2 |            0.16 |              1    |  0.25 |   0.01 |            0.11 |              0.05 |            0.18 |  0.15 |   0.1  |  0.05 |              0.19 |              0.1  |              0.24 |
| fwe             |           -0.01 |              0.25 |  1    |   0.26 |            0.09 |             -0.01 |            0.12 |  0.6  |   0.4  |  0.39 |              0.16 |              0.35 |              0.1  |
| qa_1            |            0.12 |              0.01 |  0.26 |   1    |            0.02 |              0.16 |            0.13 |  0.11 |   0.34 |  0.14 |              0.22 |              0.43 |              0.15 |
| niah_single_3   |            0.3  |              0.11 |  0.09 |   0.02 |            1    |              0.34 |            0.1  | -0.12 |   0.19 |  0.2  |              0.25 |              0.25 |              0.05 |
| niah_multikey_1 |            0.21 |              0.05 | -0.01 |   0.16 |            0.34 |              1    |            0.28 | -0.21 |   0.17 |  0.12 |              0.44 |              0.13 |              0.19 |
| niah_single_2   |            0.07 |              0.18 |  0.12 |   0.13 |            0.1  |              0.28 |            1    |  0.06 |   0.09 | -0.03 |              0.19 |              0.22 |             -0.02 |
| cwe             |           -0.12 |              0.15 |  0.6  |   0.11 |           -0.12 |             -0.21 |            0.06 |  1    |   0.36 |  0.26 |             -0.09 |              0.21 |              0.03 |
| qa_2            |            0.05 |              0.1  |  0.4  |   0.34 |            0.19 |              0.17 |            0.09 |  0.36 |   1    |  0.31 |              0.28 |              0.52 |              0.19 |
| vt              |            0.02 |              0.05 |  0.39 |   0.14 |            0.2  |              0.12 |           -0.03 |  0.26 |   0.31 |  1    |              0.3  |              0.22 |              0.09 |
| niah_multivalue |            0.35 |              0.19 |  0.16 |   0.22 |            0.25 |              0.44 |            0.19 | -0.09 |   0.28 |  0.3  |              1    |              0.36 |              0.25 |
| niah_multikey_3 |            0.19 |              0.1  |  0.35 |   0.43 |            0.25 |              0.13 |            0.22 |  0.21 |   0.52 |  0.22 |              0.36 |              1    |              0.07 |
| niah_multiquery |            0.05 |              0.24 |  0.1  |   0.15 |            0.05 |              0.19 |           -0.02 |  0.03 |   0.19 |  0.09 |              0.25 |              0.07 |              1    |

## Stability across keep ratios (Spearman, r=0.1 vs r=0.5)

- mean over prompts: 0.802
- per prompt (mean): 0.244

## Run

- GPU: NVIDIA A40
- Runtime: 762.4 s (last invocation)
- Model: /scratch1/chenpaul/kvp/models/Qwen2.5-7B-Instruct, torch 2.10.0+cu128, transformers 4.57.1
