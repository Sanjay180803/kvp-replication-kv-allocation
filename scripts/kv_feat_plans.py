#!/usr/bin/env python
"""Per-prompt allocations for (a) the FEATURE-SHUFFLE control and (b) LONGER CONTEXTS.

Uses the trained cost model, share prior and tuned alpha from kv_plan.py (rl_runs/plan); nothing is
retrained. For every RULER eval row at --length L it computes the prompt's head features once, then
writes per-prompt allocation JSONs for eval_ruler_alloc.py:

  blendplan    alpha*model(own features) + (1-alpha)*prior     (our method)
  shareplan    the static prior alone (alpha = 0)               (fixed learned split)
  --shuffle adds two controls, same planner but WRONG features:
  shufsame     features of the closest-length OTHER prompt of the SAME task
  shufcross    features of the closest-length prompt of a DIFFERENT task
If input conditioning matters, blendplan > shufsame/shufcross (and shufcross worst).

Rows: L = 4096 -> exactly the 260 eval rows used everywhere else (training rows excluded);
      L > 4096 -> first --per_task rows of each task in RULER-L (KVP traces were only 4K).
Output: <run_dir>/allocs_L<L>/<kind>_snapkv_t<T>.json  (both budgets)
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

H, NL, EPS = kp.H, kp.NL, kp.EPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="rl_runs/plan")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--per_task", type=int, default=20)
    ap.add_argument("--targets", default="0.05+0.1")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--model", default=os.path.expandvars("/scratch2/$USER/kvp/models/Qwen2.5-7B-Instruct"))
    ap.add_argument("--data_dir", default=os.path.expanduser("~/ml-learning-to-evict/data/gqa_safetensors"))
    ap.add_argument("--prompt_style", default="torchtune")
    args = ap.parse_args()
    run = Path(args.run_dir)
    targets = [float(t) for t in args.targets.split("+")]
    out = run / f"allocs_L{args.length}"; out.mkdir(exist_ok=True)
    logf = open(out / "feat_plans.log", "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    log(f"=== {time.strftime('%Y-%m-%d %H:%M')} {vars(args)}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm = kp.CostModel().to(dev)
    cm.load_state_dict(torch.load(run / "cost_model.pt", map_location=dev)); cm.eval()
    sh = [r["costs"][:, :NL - 1] / max(float(r["costs"][:, 0].sum()), EPS) for r in kp.load(run, "train")]
    prior_lc = np.log(np.mean(sh, 0) + 1e-8)
    alpha = {t: json.load(open(run / f"alpha_t{t}.json"))["alpha"] for t in targets}

    model, tok = ra.load_model(args.model)
    from datasets import load_dataset
    ds = load_dataset("simonjegou/ruler", data_dir=str(args.length), split="test")
    if args.length == 4096:
        rows = ra.eval_rows(ds, args.data_dir, per_task=args.per_task)
    else:
        rows, cnt = [], {}
        for i, task in enumerate(ds["task"]):
            if cnt.get(task, 0) < args.per_task:
                cnt[task] = cnt.get(task, 0) + 1
                rows.append(dict(ds[i], prompt_id=i))
    log(f"RULER-{args.length}: {len(rows)} rows, alpha {alpha}")

    feats, plen, tasks = {}, {}, {}
    t0 = time.time()
    for n, row in enumerate(rows):
        pid = str(row["prompt_id"]); tasks[pid] = row["task"]
        try:
            ep = ra.Episode(model, tok, row, args.prompt_style)
            feats[pid], plen[pid] = ep.feats.float().cpu(), ep.P
        except AssertionError:
            feats[pid], plen[pid] = None, None
        if n % 20 == 0:
            log(f"[feat] {n + 1}/{len(rows)} ({(time.time() - t0) / (n + 1):.1f}s/prompt)")
        torch.cuda.empty_cache()
    ok = [p for p in feats if feats[p] is not None]
    log(f"features for {len(ok)}/{len(rows)} prompts (others fall back to shareplan); "
        f"prompt length min/mean/max {min(plen[p] for p in ok)}/{np.mean([plen[p] for p in ok]):.0f}/{max(plen[p] for p in ok)}")

    def lc(f):
        with torch.no_grad():
            return cm(f.to(dev)[None])[0].cpu().numpy()

    def partner(pid, same_task):
        cand = [q for q in ok if q != pid and (tasks[q] == tasks[pid]) == same_task]
        return min(cand, key=lambda q: abs(plen[q] - plen[pid])) if cand and plen[pid] else None

    kinds = ["blendplan", "shareplan"] + (["shufsame", "shufcross"] if args.shuffle else [])
    src = {"blendplan": {p: p for p in ok}}
    if args.shuffle:
        src["shufsame"] = {p: partner(p, True) for p in ok}
        src["shufcross"] = {p: partner(p, False) for p in ok}
    heads = [{"layer": h // 4, "kv_head": h % 4} for h in range(H)]
    for t in targets:
        share = kp.plan(np.exp(prior_lc), t)
        plans = {k: {} for k in kinds}
        for pid in feats:
            plans["shareplan"][pid] = share
            for k in kinds[:1] + kinds[2:]:
                q = src[k].get(pid)
                plans[k][pid] = share if q is None else kp.plan(np.exp(alpha[t] * lc(feats[q]) + (1 - alpha[t]) * prior_lc), t)
        for k in kinds:
            name = f"{k}_snapkv_t{t}"
            json.dump({"name": name, "kind": k, "scorer": "snapkv", "target_avg_keep_ratio": t, "actual_avg_keep_ratio": t,
                       "alpha": 0.0 if k == "shareplan" else alpha[t], "length": args.length, "heads": heads,
                       "per_prompt": {p: [float(x) for x in a] for p, a in plans[k].items()}},
                      open(out / f"{name}.json", "w"))
        for k in kinds[2:]:
            d = np.mean([np.abs(plans[k][p] - plans["blendplan"][p]).sum() / 2 for p in ok])
            log(f"[t={t}] {k}: mean memory moved vs own-feature plan = {d:.3f} head-units ({100 * d / (t * H):.1f}% of budget)")
    log(f"wrote {len(kinds) * len(targets)} files to {out}")


if __name__ == "__main__":
    main()
