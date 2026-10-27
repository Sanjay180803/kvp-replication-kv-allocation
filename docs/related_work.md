# Related work and novelty

## Instructor feedback on the proposal (−1 point, novelty)

> RLKV already uses RL and answer correctness to learn unequal cache allocation across heads. RateQuant studies per-head precision allocation, while RDKV jointly optimizes eviction and quantization. These close approaches are missing from the comparison. Your potential contribution is input-conditioned, joint budget-and-precision allocation using output KL. This remains a reasonable extension, but its distinction from existing methods should be made explicit.

The instructor's suggestions:
- Compare a fixed learned allocation with the input-conditioned policy. With the same token scorer, test budget allocation, precision selection, and both, to see where the gains come from.
- Compute negative KL over several decoding steps after compression, with matched prefixes. Check that improvements also show up in retrieval accuracy and longer generated answers.
- With GQA, allocate per shared KV head. Include quantization scales and indexing overhead in the memory budget, and report policy overhead alongside quality.

## How the close papers differ

| | What is allocated | Changes per input? | How it is decided | Eviction + precision together? |
|---|---|---|---|---|
| **RLKV** ([arXiv 2510.08525](https://arxiv.org/abs/2510.08525)) | Per KV head, binary: full cache vs sinks + local window (16 + 64) | **No.** Gates are fixed after training; input-adaptive is listed as future work | RL (GRPO) on answer correctness; learned gates mix full and local attention | No quantization |
| **RateQuant** ([arXiv 2605.06675](https://arxiv.org/abs/2605.06675)) | Bits per head | **No.** Calibrated once on a small set | Analytic: rate-distortion, reverse water-filling | Quantization only |
| **RDKV** ([arXiv 2605.08317](https://arxiv.org/abs/2605.08317)) | Bits per **token / channel**, 0 bits = evicted | **Yes.** Computed after each prefill | Analytic: per-layer attention-distortion proxy + water-filling | **Yes** |
| **Ours** | Per KV head: token budget + precision mode | Yes | **Learned RL policy, reward = end-to-end output KL** | Yes |

## Our claimed contribution, stated precisely

"Input-conditioned" and "joint eviction + quantization" are **not new on their own**, since RDKV already does both. What is distinct:

1. **Learned, end-to-end objective vs analytic local proxies.** RDKV and RateQuant measure distortion layer by layer. Our policy is trained on the model's final output distribution (KL vs the full cache), so it can account for errors that compound across layers.
2. **Per-KV-head granularity.** One budget and one bit-width per shared KV head (112 decisions for Qwen2.5-7B) instead of per-token bit maps. That means far fewer decisions, no per-token bit/index tables, and it maps onto uniform-precision attention kernels.
3. **Versus RLKV.** Per input instead of fixed, continuous budgets instead of a binary full/local choice, a real within-head ranker instead of only a local window, and precision selection on top.

These have to be **shown experimentally**:
- Ablation grid with the same token scorer: uniform → fixed learned allocation (RLKV-style) → per-input budget only → per-input precision only → both.
- Baselines: RateQuant-style static bit allocation (reimplementable from calibration + water-filling); RDKV (their code if released, otherwise a simplified reimplementation, clearly labeled).
- Multi-step KL with matched prefixes, checked against RULER retrieval accuracy and long-answer tasks (`cwe`, `fwe`, a LongBench summarization task).
- Honest memory: fp16 scale + zero-point per quantization group (e.g. group 32 at 2-bit adds ~1 bit per value), token-index storage for evicted caches, and policy latency.
