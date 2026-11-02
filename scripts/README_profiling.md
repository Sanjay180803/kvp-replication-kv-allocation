# Per-KV-head sensitivity profiling

Measures how much Qwen2.5-7B-Instruct's next-token distribution changes when only ONE KV head's
prompt cache is shrunk (SnapKV ranking, optional StreamingLLM), for all 112 KV heads at keep
ratios 0.1 / 0.25 / 0.5, on RULER-4K rows `500t+400..` (never the first 400 of a block).

## Files
- `profile_heads.py`: profiling + sanity checks, writes `results/profiling/head_kl.csv` (resumable) and `run_meta.json`
- `analyze_profiling.py`: heatmaps, task Spearman matrix, `summary.md`
- `profile_heads.sbatch`: CARC job (A40/A100/L40S, no P100)
- `test_tiny.py`: CPU smoke test on a tiny random Qwen2 (no download needed)
- `requirements-profiling.txt`: pins transformers 4.57.1 (the custom attention signature depends on it)

## Run on CARC
```bash
salloc --partition=gpu --gres=gpu:1 --constraint="a40|a100|l40s" --time=1:00:00   # installs on a compute node
pip install -r scripts/requirements-profiling.txt
python scripts/profile_heads.py --sanity_only        # checks 1-5 on 3 prompts, ~minutes
exit
mkdir -p logs && sbatch scripts/profile_heads.sbatch                       # 39 prompts
sbatch scripts/profile_heads.sbatch --n_per_task 5 --scorers snapkv,streaming   # 65 prompts + 2nd scorer
```
A crashed job can be resubmitted as is: prompts already in `head_kl.csv` are skipped.
If you hit OOM, lower `--batch_size` (default 48 head/ratio runs per forward).

## How it works (one point worth knowing)
Eviction after prefill only changes what continuation queries (positions >= P) see, and prompt
positions never attend to the continuation. So the prompt hidden states are identical with or
without eviction. The script therefore does ONE full pass per prompt (captures every layer's
post-RoPE prompt K/V and SnapKV scores), then runs each (head, ratio) as a 32-token pass that
attends to the captured prompt K/V with a per-item, per-query-head mask. This is exactly the
"teacher-forced forward on prompt + c_1..c_K with the head masked" from the task spec, just
~100x cheaper. Sanity check 5 confirms it matches the plain full-length masked forward.

The KL reference is the same 32-token path with nothing evicted, so r = 1.0 gives exactly 0 and
bf16 noise does not leak into small head KLs. The log also prints how far this reference is from
the plain full-cache pass (`cached-vs-full ref`).

## Sanity checks (printed with PASS/FAIL; the full run aborts if any fail)
1. r = 1.0 gives KL < 1e-3
2. at r = 0.1 about 0.9 P keys are blocked, the 64-token window is never evicted, KL > 0
3. teacher-forced argmax reproduces the greedy `model.generate` continuation
4. all 112 heads at r = 0.1 gives a larger KL than any single head
5. cached 32-token path == full-length masked forward

## Matching scripts/eval_ruler.py
- SnapKV scores are exactly `SnapKVPress.sort` from eval_ruler.py (window 64, fp32 softmax over all
  P keys, mean over the 7 query heads + window, avg-pool 5 over all P, window set to +inf).
  Verified on random tensors: 0.0 difference, identical keep sets.
- Budget: keep `max(1, int(r * P))` prompt tokens, same rule as the eval.
- StreamingLLM: 4 sinks + most recent tokens, same as `StreamingLLMPress`.
- The question is part of the compressed prefill, same as the eval.
- Prompt: `--prompt_style torchtune` (default) builds `<|im_start|>user ... <|im_end|> <|im_start|>assistant <answer_prefix>`
  with no default system prompt, like the torchtune tokenizer in the eval. `--prompt_style hf` uses
  `apply_chat_template` instead (adds Qwen's default system prompt).
