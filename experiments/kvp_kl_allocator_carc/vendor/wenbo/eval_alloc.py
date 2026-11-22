"""RULER-4K evaluation of per-head token COUNTS, with the frozen KVP agents choosing the tokens.

Same protocol as the team's eval_ruler_alloc.py (prompt format, compress once after prefill, greedy decode,
RULER string-match score). Differences:
  - prompts come from our splits file (no dependence on trace folders), so test = the team's 260 IDs;
  - an allocation gives an exact token count per head (not a ratio), so totals are exact.

Allocation file (JSON): {"name": str, "ratio": r, "per_prompt": {"<id>": [112 ints, head = 4*layer + kv]}}
Built-in uniform: --uniform r  ->  every head keeps int(r * prompt_len), as in the team's uniform runs.
Output: <out_dir>/ruler_<name>.jsonl, one line per prompt, resumable.

Usage: uv run python $ALLOC/eval_alloc.py --splits $DATA/splits.json --split test --alloc A.json --out_dir $DATA/eval
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label import DEVICE, DTYPE, KVP_SHARED, MODEL_DIR, build_prompt, load_model, ruler_score  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torchtune import config, training  # noqa: E402
from torchtune.models.qwen2_5 import qwen2_5_tokenizer  # noqa: E402
from kvcompression.hooks.compressor import KVCompressor, LayerHeadCompressionConfig  # noqa: E402
from kvcompression.kv_cache.compression_strategy_protocol import CompressionStrategy, PressCompressionStrategy  # noqa: E402
from kvcompression.utils.generation import generate_with_compression  # noqa: E402


class FixedCountStrategy(CompressionStrategy):
    def __init__(self, inner, count):
        self.inner, self.count = inner, int(count)

    def get_mask(self, k_cache_active, v_cache_active, target_compressed_size, q_assoc_cache_active=None, **kw):
        S = k_cache_active.shape[2]
        if self.count >= S:
            return torch.ones(k_cache_active.shape[:-1], dtype=torch.bool, device=k_cache_active.device)
        return self.inner.get_mask(k_cache_active=k_cache_active, v_cache_active=v_cache_active,
                                   target_compressed_size=max(1, self.count),
                                   q_assoc_cache_active=q_assoc_cache_active, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids", default=None, help="optional JSON list of IDs (subset of the split)")
    ap.add_argument("--alloc", default=None)
    ap.add_argument("--uniform", type=float, default=None)
    ap.add_argument("--team_static", default=None, help="team allocation JSON (keep_ratio per head), e.g. shareplan")
    ap.add_argument("--composite", default="config_composites/ruler845_b128.yaml")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    assert sum(x is not None for x in (args.alloc, args.uniform, args.team_static)) == 1

    splits = json.load(open(args.splits))
    ids = splits[args.split]
    if args.ids:
        sub = set(json.load(open(args.ids)))
        ids = [i for i in ids if i in sub]
    if args.alloc:
        alloc = json.load(open(args.alloc))
        name, ratio = alloc["name"], alloc["ratio"]
        missing = [i for i in ids if str(i) not in alloc["per_prompt"]]
        assert not missing, f"{len(missing)} prompts missing from allocation"
    elif args.team_static:
        ts = json.load(open(args.team_static))
        team_ratio = {(int(h["layer"]), int(h["kv_head"])): float(h["keep_ratio"]) for h in ts["heads"]}
        name, ratio, alloc = f"team_{ts['name']}__kvp", float(ts["target_avg_keep_ratio"]), None
    else:
        name, ratio, alloc = f"uniform_kvp_r{args.uniform}", args.uniform, None

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"ruler_{name}.jsonl"
    done = {json.loads(l)["id"] for l in open(out_file)} if out_file.exists() else set()

    ds = load_dataset("simonjegou/ruler", data_dir="4096", split="test")
    tok = qwen2_5_tokenizer(path=str(MODEL_DIR / "vocab.json"), merges_file=str(MODEL_DIR / "merges.txt"))
    model = load_model()
    data = OmegaConf.load(args.composite)
    kvp = {}
    for a in data.assignments:
        with training.set_default_dtype(DTYPE), DEVICE:
            p = config.instantiate(OmegaConf.merge(a.press, KVP_SHARED))
        kvp[(int(a.layer), int(a.head))] = PressCompressionStrategy(p.to(DEVICE).eval())

    with torch.no_grad(), open(out_file, "a") as fout:
        for n, i in enumerate(ids):
            if i in done:
                continue
            t0 = time.time()
            row = ds[i]
            prompt = build_prompt(tok, row)
            P = prompt.shape[1]
            if alloc:
                counts = [int(c) for c in alloc["per_prompt"][str(i)]]
            elif args.team_static:   # same rule as the team's FixedRatioStrategy
                counts = [P if team_ratio[(L, h)] >= 1.0 else max(1, int(team_ratio[(L, h)] * P))
                          for L in range(28) for h in range(4)]
            else:
                counts = [int(ratio * P)] * 112
            assert len(counts) == 112
            cfgs = [LayerHeadCompressionConfig(L, h, FixedCountStrategy(kvp[(L, h)], counts[4 * L + h]), f"L{L}H{h}")
                    for L in range(28) for h in range(4)]
            max_new = int(row.get("max_new_tokens") or 128)
            with KVCompressor(model=model, compression_strategies=cfgs) as comp:
                with DEVICE:
                    model.setup_caches(batch_size=1, dtype=DTYPE, decoder_max_seq_len=P + max_new)
                gen = generate_with_compression(model, prompt, comp, P, max_new, stop_tokens=tok.stop_tokens)
            pred = tok.decode(gen[0, P:].tolist(), skip_special_tokens=True)
            s = ruler_score(pred, row["answer"], row["task"])
            kept = sum(min(c, P) for c in counts)
            rec = {"id": i, "task": row["task"], "method": name, "ratio": ratio, "prompt_len": P,
                   "kept_frac": kept / (112 * P), "score": s, "pred": pred, "sec": round(time.time() - t0, 2)}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(f"[{n+1}/{len(ids)}] {row['task']:<16} score={s:5.1f} kept={rec['kept_frac']:.4f} {rec['sec']}s",
                  flush=True)


if __name__ == "__main__":
    main()
