"""RULER-4K eval with a DIFFERENT keep ratio per KV head (allocation file from make_allocation.py).

Same protocol as eval_ruler.py (same held-out rows, prompt format, compress once after prefill,
SnapKV / StreamingLLM ranking inside each head, RULER string-match score); the only difference is
that head (layer, kv_head) keeps int(keep_ratio[layer, kv_head] x prompt_len) tokens instead of a
single ratio for all heads. A ratio >= 1.0 means that head is not compressed at all.

Usage (from the ml-learning-to-evict repo root, with eval_ruler.py also in scripts/):
  uv run python scripts/eval_ruler_alloc.py --alloc allocations/profile-mean_snapkv_t0.25.json --per_task 20
Sanity check: the uniform allocation must reproduce eval_ruler.py (snapkv @ 0.25 = 64.4, @ 0.5 = 77.9).

Output: <out_dir>/ruler_<alloc name>.jsonl, one line per example, resumable.
  method = "<scorer>:<kind>", ratio = target average keep ratio, so summarize_ruler.py works as is.
  Also logs kept tokens and KV-cache bytes actually used (bf16 K+V, 128-dim heads).
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from torchtune import config, training
from torchtune.data import Message
from torchtune.models.qwen2_5 import qwen2_5_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ruler import DEVICE, DTYPE, MODEL_DIR, SnapKVPress, StreamingLLMPress, score  # noqa: E402

from kvcompression.hooks.compressor import KVCompressor, LayerHeadCompressionConfig
from kvcompression.kv_cache.compression_strategy_protocol import (
    CompressionStrategy, DummyCompressionStrategy, PressCompressionStrategy)
from kvcompression.utils.generation import generate_with_compression

HEAD_DIM = 128
BYTES_PER_TOKEN_PER_HEAD = 2 * HEAD_DIM * 2  # K + V, bf16


class FixedRatioStrategy(CompressionStrategy):
    """Wraps a strategy and replaces the global target size with this head's own keep ratio."""

    def __init__(self, inner: CompressionStrategy, keep_ratio: float):
        self.inner, self.keep_ratio = inner, keep_ratio

    def __repr__(self):
        return f"FixedRatio({self.keep_ratio:.3f}, {self.inner!r})"

    def get_mask(self, k_cache_active, v_cache_active, target_compressed_size, q_assoc_cache_active=None, **kwargs):
        S = k_cache_active.shape[2]
        if self.keep_ratio >= 1.0:
            return torch.ones(k_cache_active.shape[:-1], dtype=torch.bool, device=k_cache_active.device)
        keep = max(1, int(self.keep_ratio * S))
        return self.inner.get_mask(k_cache_active=k_cache_active, v_cache_active=v_cache_active,
                                   target_compressed_size=keep, q_assoc_cache_active=q_assoc_cache_active, **kwargs)


def build_configs(alloc, scorer):
    press = SnapKVPress() if scorer == "snapkv" else StreamingLLMPress()
    inner = PressCompressionStrategy(press)
    cfgs = []
    for h in alloc["heads"]:
        strat = FixedRatioStrategy(inner, float(h["keep_ratio"]))
        cfgs.append(LayerHeadCompressionConfig(int(h["layer"]), int(h["kv_head"]), strat,
                                               f"{scorer}_L{h['layer']}_H{h['kv_head']}_r{h['keep_ratio']:.3f}"))
    assert len(cfgs) == 112, f"allocation must cover all 112 KV heads, got {len(cfgs)}"
    return cfgs


def kept_tokens(alloc, plen):
    return sum(plen if h["keep_ratio"] >= 1.0 else max(1, int(h["keep_ratio"] * plen)) for h in alloc["heads"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", required=True, help="JSON from make_allocation.py")
    ap.add_argument("--scorer", default=None, choices=["snapkv", "streaming"], help="default: the allocation's scorer")
    ap.add_argument("--per_task", type=int, default=20)
    ap.add_argument("--out_dir", default="results_alloc")
    args = ap.parse_args()

    alloc = json.load(open(args.alloc))
    scorer = args.scorer or alloc.get("scorer", "snapkv")
    method = f"{scorer}:{alloc['kind']}"
    target = float(alloc["target_avg_keep_ratio"])

    # identical row selection to eval_ruler.py
    used = {int(p.name.split("_")[-1]) for p in Path("data/gqa_safetensors").glob("**/sample_*")}
    ds = load_dataset("simonjegou/ruler", data_dir="4096", split="test")
    picked, count = [], {}
    for i, row in enumerate(ds):
        t = row["task"]
        if i in used or count.get(t, 0) >= args.per_task:
            continue
        count[t] = count.get(t, 0) + 1
        picked.append(i)
    print(f"{alloc['name']}: {len(picked)} rows, scorer={scorer}, target avg keep={target}, "
          f"actual avg keep={alloc['actual_avg_keep_ratio']:.4f}, heads uncompressed={sum(h['keep_ratio'] >= 1 for h in alloc['heads'])}")

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

    cfgs = build_configs(alloc, scorer)

    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    out_file = out / f"ruler_{alloc['name']}.jsonl"
    done = set()
    if out_file.exists():
        done = {json.loads(l)["id"] for l in open(out_file)}

    with torch.no_grad(), open(out_file, "a") as fout:
        for n, i in enumerate(picked):
            if i in done:
                continue
            row = ds[i]
            msgs = [Message(role="user", content=row["context"] + row["question"]),
                    Message(role="assistant", content=row["answer_prefix"], eot=False)]
            ids, _ = tok.tokenize_messages(msgs, add_eos=False)
            prompt = torch.tensor([ids], device=DEVICE)
            plen = prompt.shape[1]
            max_new = int(row.get("max_new_tokens") or 128)
            t0 = time.time()
            with KVCompressor(model=model, compression_strategies=cfgs) as comp:
                with DEVICE:
                    model.setup_caches(batch_size=1, dtype=DTYPE, decoder_max_seq_len=plen + max_new)
                # the global target is ignored by FixedRatioStrategy; plen just has to be a valid size
                gen = generate_with_compression(model, prompt, comp, plen, max_new, stop_tokens=tok.stop_tokens)
            pred = tok.decode(gen[0, plen:].tolist(), skip_special_tokens=True)
            s = score(pred, row["answer"], row["task"])
            kept = kept_tokens(alloc, plen)
            rec = {"id": i, "task": row["task"], "method": method, "alloc": alloc["name"], "ratio": target,
                   "prompt_len": plen, "kept_tokens_all_heads": kept, "kept_frac": kept / (112 * plen),
                   "kv_bytes": kept * BYTES_PER_TOKEN_PER_HEAD, "kv_bytes_full": 112 * plen * BYTES_PER_TOKEN_PER_HEAD,
                   "score": s, "pred": pred, "answer": row["answer"], "sec": round(time.time() - t0, 2)}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(f"[{n+1}/{len(picked)}] {row['task']:<16} score={s:5.1f} kept={kept / (112 * plen):.3f} "
                  f"{rec['sec']}s | {pred[:60]!r}", flush=True)


if __name__ == "__main__":
    main()
