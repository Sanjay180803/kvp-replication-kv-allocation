#!/usr/bin/env python
"""Policy overhead of the learned allocation (instructor ask: "report policy overhead").

Per prompt, on the same GPU, wall-clock (CUDA-synchronized) time of:
  prefill        plain full forward over the prompt (sdpa, no capture)       = the cost we compare to
  decode/token   greedy decoding, per generated token (32 tokens, minus prefill)
  features       head_features(): 8 numbers per KV head from the SnapKV scores
                 (the SnapKV scores themselves are computed by SnapKV in any case, so not counted)
  cost model     MLP forward -> predicted log-cost for 112 heads x 5 levels
  planner        shrinkage blend + greedy knapsack -> keep ratio per head (CPU, pandas)
plus the memory cost of the plan itself (112 keep counts) vs the KV cache it controls.

Usage: slurm/time_overhead.sh   (writes rl_runs/plan/overhead.json)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_alloc as ra          # noqa: E402
import kv_plan as kp           # noqa: E402
import profile_heads as ph     # noqa: E402

BYTES_PER_TOKEN_PER_HEAD = 2 * 128 * 2     # K + V, bf16, head_dim 128


def timed(f, reps=1):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        out = f()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) / reps * 1000.0     # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="rl_runs/plan")
    ap.add_argument("--model", default=os.path.expandvars("/scratch2/$USER/kvp/models/Qwen2.5-7B-Instruct"))
    ap.add_argument("--data_dir", default=os.path.expanduser("~/ml-learning-to-evict/data/gqa_safetensors"))
    ap.add_argument("--prompt_style", default="torchtune")
    ap.add_argument("--n_per_task", type=int, default=2)
    ap.add_argument("--targets", default="0.05+0.1")
    ap.add_argument("--n_decode", type=int, default=32)
    args = ap.parse_args()
    run = Path(args.run_dir)
    targets = [float(t) for t in args.targets.split("+")]

    model, tok = ra.load_model(args.model)
    dev = model.device
    ds = ra.load_ruler()
    rows, cnt = [], {}
    for r in ra.eval_rows(ds, args.data_dir):
        if cnt.get(r["task"], 0) < args.n_per_task:
            cnt[r["task"]] = cnt.get(r["task"], 0) + 1
            rows.append(r)

    cm = kp.CostModel().to(dev)
    cm.load_state_dict(torch.load(run / "cost_model.pt", map_location=dev)); cm.eval()
    sh = [r["costs"][:, :kp.NL - 1] / max(float(r["costs"][:, 0].sum()), kp.EPS) for r in kp.load(run, "train")]
    prior_lc = np.log(np.mean(sh, 0) + 1e-8)
    alpha = {t: json.load(open(run / f"alpha_t{t}.json"))["alpha"] for t in targets}
    print(f"{len(rows)} prompts, alpha {alpha}, GPU {torch.cuda.get_device_name()}", flush=True)

    rec = []
    with torch.no_grad():
        for n, row in enumerate(rows):
            ids = tok(ph.build_prompt(tok, row, args.prompt_style), return_tensors="pt",
                      add_special_tokens=False).input_ids.to(dev)
            P = ids.shape[1]
            try:
                ep = ra.Episode(model, tok, row, args.prompt_style, ids=ids)      # fills SnapKV scores
            except AssertionError:
                continue                                                         # continuation too short
            S = torch.stack([ph.State.snap_scores[L] for L in range(ep.nL)])
            ph.State.mode = "off"

            _, t_pre = timed(lambda: model(ids, use_cache=True, logits_to_keep=1))
            _, t_gen = timed(lambda: model.generate(ids, attention_mask=torch.ones_like(ids),
                                                    max_new_tokens=args.n_decode, min_new_tokens=args.n_decode,
                                                    do_sample=False, temperature=None, top_p=None, top_k=None,
                                                    pad_token_id=tok.pad_token_id or tok.eos_token_id))
            t_tok = max(t_gen - t_pre, 0.0) / args.n_decode

            feats, t_feat = timed(lambda: ra.head_features(S, P), reps=5)
            lc, t_cm = timed(lambda: cm(feats.float()[None])[0].cpu().numpy(), reps=5)
            r = dict(task=row["task"], P=P, prefill_ms=t_pre, decode_ms_per_token=t_tok,
                     features_ms=t_feat, cost_model_ms=t_cm)
            for t in targets:
                alloc, t_plan = timed(lambda: kp.plan(np.exp(alpha[t] * lc + (1 - alpha[t]) * prior_lc), t), reps=3)
                kept = sum(P if a >= 1 else max(1, int(a * P)) for a in alloc)
                r[f"planner_ms_t{t}"] = t_plan
                r[f"kv_bytes_t{t}"] = kept * BYTES_PER_TOKEN_PER_HEAD
            rec.append(r)
            print(f"[{n + 1}/{len(rows)}] {row['task']:<16} P={P} prefill {t_pre:.0f}ms decode {t_tok:.1f}ms/tok "
                  f"feat {t_feat:.2f}ms model {t_cm:.2f}ms plan " +
                  " ".join(f"{r[f'planner_ms_t{t}']:.0f}ms" for t in targets), flush=True)

    rec = rec[1:] if len(rec) > 1 else rec       # drop the first prompt (CUDA warm-up)
    m = {k: float(np.mean([r[k] for r in rec])) for k in rec[0] if k != "task"}
    plan_bytes = 112 * 4                          # one int32 keep count per KV head
    print("\n=== policy overhead (mean over %d prompts, avg prompt %.0f tokens) ===" % (len(rec), m["P"]))
    print(f"prefill (reference)        {m['prefill_ms']:8.1f} ms")
    print(f"decode                     {m['decode_ms_per_token']:8.1f} ms / token")
    print(f"features                   {m['features_ms']:8.2f} ms")
    print(f"cost model                 {m['cost_model_ms']:8.2f} ms")
    for t in targets:
        tot = m["features_ms"] + m["cost_model_ms"] + m[f"planner_ms_t{t}"]
        print(f"planner @ {t:<5}            {m[f'planner_ms_t{t}']:8.1f} ms  -> total {tot:.1f} ms = "
              f"{100 * tot / m['prefill_ms']:.1f}% of prefill, = {tot / m['decode_ms_per_token']:.1f} decode tokens; "
              f"plan metadata {plan_bytes} B = {100 * plan_bytes / m[f'kv_bytes_t{t}']:.4f}% of the kept KV "
              f"({m[f'kv_bytes_t{t}'] / 1e6:.1f} MB)")
    json.dump(dict(mean=m, per_prompt=rec, plan_metadata_bytes=plan_bytes, gpu=torch.cuda.get_device_name()),
              open(run / "overhead.json", "w"), indent=1)
    print(f"saved {run}/overhead.json")


if __name__ == "__main__":
    main()
