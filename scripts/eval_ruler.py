"""RULER-4K eval: full cache vs StreamingLLM vs SnapKV vs random vs KVP (trained agents).

Usage (from the ml-learning-to-evict repo root):
  uv run python scripts/eval_ruler.py --method kvp --ratios 0.1 0.25 0.5 --per_task 20

Methods: full | streaming | snapkv | kvp | random
- Eviction happens once after prefill (same as Apple's demo notebook); the question is part of the prefill.
- Budget = ratio * prompt_len tokens kept per KV head (uniform across heads).
- Only RULER rows NOT used for training are evaluated (every sample id under data/gqa_safetensors is excluded).
- Results -> <out_dir>/ruler_<method>.jsonl (resumable: finished (id, ratio) pairs are skipped).
"""
import argparse, json, math, time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import OmegaConf
from torchtune import config, training
from torchtune.data import Message
from torchtune.models.qwen2_5 import qwen2_5_tokenizer

from kvcompression import PROJECT_ROOT
from kvcompression.hooks.compressor import KVCompressor, LayerHeadCompressionConfig
from kvcompression.kv_cache.compression_strategy_protocol import (
    DummyCompressionStrategy, PressCompressionStrategy)
from kvcompression.presses.base_press import BasePress
from kvcompression.presses.random_press import RandomPress
from kvcompression.utils.generation import generate_with_compression

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
MODEL_DIR = PROJECT_ROOT / "models" / "Qwen2.5-7B-Instruct"


class StreamingLLMPress(BasePress):
    """Keep attention sinks + most recent tokens."""
    def __init__(self, n_sinks=4):
        super().__init__()
        self.n_sinks = n_sinks

    def sort(self, hidden_states, keys, values, queries, attention_weights=None):
        B, H, S, _ = keys.shape
        scores = torch.arange(S, device=keys.device, dtype=torch.float32)
        scores[: min(self.n_sinks, S)] = float("inf")
        return scores.expand(B, H, S).argsort(-1, descending=True)


class SnapKVPress(BasePress):
    """Score tokens by attention from the last `window` queries (all GQA query heads), pooled."""
    def __init__(self, window=64, kernel=5):
        super().__init__()
        self.window, self.kernel = window, kernel

    def sort(self, hidden_states, keys, values, queries, attention_weights=None):
        B, H, S, D = keys.shape  # H == 1 (called per KV head)
        w = min(self.window, S)
        q = queries[..., -w:, :].float()                       # [B, Q, w, D]
        k = keys.float()                                       # [B, 1, S, D]
        attn = q @ k.transpose(-1, -2) / math.sqrt(D)          # [B, Q, w, S]
        pos = torch.arange(S, device=keys.device)
        rows = torch.arange(S - w, S, device=keys.device)
        attn = attn.masked_fill(pos[None, :] > rows[:, None], float("-inf"))
        scores = attn.softmax(-1).mean(dim=(1, 2))              # [B, S]
        scores = F.avg_pool1d(scores[:, None], self.kernel, stride=1, padding=self.kernel // 2)[:, 0]
        scores[:, S - w:] = float("inf")                       # always keep the window
        return scores[:, None, :].expand(B, H, S).argsort(-1, descending=True)


def build_configs(method, composite):
    if method == "full":
        return [LayerHeadCompressionConfig(None, None, DummyCompressionStrategy(), "full")]
    if method == "streaming":
        return [LayerHeadCompressionConfig(None, None, PressCompressionStrategy(StreamingLLMPress()), "streaming")]
    if method == "random":
        return [LayerHeadCompressionConfig(None, None, PressCompressionStrategy(RandomPress()), "random")]
    if method == "snapkv":
        return [LayerHeadCompressionConfig(None, None, PressCompressionStrategy(SnapKVPress()), "snapkv")]
    # kvp: one trained agent per (layer, kv_head)
    data = OmegaConf.load(composite)
    shared = {"n_sinks": 4, "n_running_window": 16}  # same as Apple's demo notebook
    cfgs = []
    for a in data.assignments:
        with training.set_default_dtype(DTYPE), DEVICE:
            press = config.instantiate(OmegaConf.merge(a.press, shared))
        press = press.to(DEVICE).eval()
        cfgs.append(LayerHeadCompressionConfig(a.layer, a.head, PressCompressionStrategy(press), f"kvp_L{a.layer}_H{a.head}"))
    print(f"loaded {len(cfgs)} KVP agents")
    return cfgs


def score(pred, refs, task):
    refs = list(refs) if isinstance(refs, (list, tuple)) else [refs]
    p = pred.lower()
    hits = [str(r).lower() in p for r in refs]
    if task.startswith("qa"):
        return 100.0 * max(hits)            # RULER string_match_part
    return 100.0 * sum(hits) / len(hits)    # RULER string_match_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["full", "streaming", "snapkv", "kvp", "random"])
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    ap.add_argument("--per_task", type=int, default=20)
    ap.add_argument("--composite", default="config_composites/ruler845_v1.yaml")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    ratios = [1.0] if args.method == "full" else args.ratios

    used = {int(p.name.split("_")[-1]) for p in Path("data/gqa_safetensors").glob("**/sample_*")}
    ds = load_dataset("simonjegou/ruler", data_dir="4096", split="test")
    picked, count = [], {}
    for i, row in enumerate(ds):
        t = row["task"]
        if i in used or count.get(t, 0) >= args.per_task:
            continue
        count[t] = count.get(t, 0) + 1
        picked.append(i)
    print(f"{len(used)} training ids excluded; evaluating {len(picked)} rows over {len(count)} tasks")

    tok = qwen2_5_tokenizer(path=str(MODEL_DIR / "vocab.json"), merges_file=str(MODEL_DIR / "merges.txt"))
    ckpt = config.instantiate(OmegaConf.create({
        "_component_": "torchtune.training.FullModelHFCheckpointer",
        "checkpoint_dir": str(MODEL_DIR),
        "checkpoint_files": [f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)],
        "output_dir": "/tmp", "model_type": "QWEN2"})).load_checkpoint()
    with training.set_default_dtype(DTYPE), DEVICE:
        model = config.instantiate(OmegaConf.create({"_component_": "torchtune.models.qwen2_5.qwen2_5_7b_instruct"}))
    model.load_state_dict(ckpt[training.MODEL_KEY], strict=True)
    model.eval()
    del ckpt

    cfgs = build_configs(args.method, args.composite)

    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    out_file = out / f"ruler_{args.method}.jsonl"
    done = set()
    if out_file.exists():
        for line in open(out_file):
            r = json.loads(line); done.add((r["id"], r["ratio"]))

    with torch.no_grad(), open(out_file, "a") as fout:
        for n, i in enumerate(picked):
            row = ds[i]
            msgs = [Message(role="user", content=row["context"] + row["question"]),
                    Message(role="assistant", content=row["answer_prefix"], eot=False)]
            ids, _ = tok.tokenize_messages(msgs, add_eos=False)
            prompt = torch.tensor([ids], device=DEVICE)
            plen = prompt.shape[1]
            max_new = int(row.get("max_new_tokens") or 128)
            for ratio in ratios:
                if (i, ratio) in done:
                    continue
                budget = plen if args.method == "full" else max(1, int(ratio * plen))
                t0 = time.time()
                with KVCompressor(model=model, compression_strategies=cfgs) as comp:
                    with DEVICE:
                        model.setup_caches(batch_size=1, dtype=DTYPE, decoder_max_seq_len=plen + max_new)
                    gen = generate_with_compression(
                        model, prompt, None if args.method == "full" else comp, budget, max_new,
                        stop_tokens=tok.stop_tokens)
                pred = tok.decode(gen[0, plen:].tolist(), skip_special_tokens=True)
                s = score(pred, row["answer"], row["task"])
                rec = {"id": i, "task": row["task"], "method": args.method, "ratio": ratio,
                       "prompt_len": plen, "budget": budget, "score": s, "pred": pred,
                       "answer": row["answer"], "sec": round(time.time() - t0, 2)}
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                print(f"[{n+1}/{len(picked)}] {row['task']:<16} r={ratio:<5} score={s:5.1f} "
                      f"len={plen} {rec['sec']}s | {pred[:60]!r}", flush=True)


if __name__ == "__main__":
    main()
