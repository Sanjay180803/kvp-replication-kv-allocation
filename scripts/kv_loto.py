#!/usr/bin/env python
"""Generalization test for the per-input allocator: LEAVE-ONE-TASK-OUT (LOTO).

For each of the 13 RULER tasks T, the cost model and the static prior are learned WITHOUT any
prompt of task T, and then used only on task T's prompts. So every allocation here comes from a
model that never saw that kind of input. Compared against the static plan built the same way
(shareplan without T), this tests whether input-conditioning helps on unseen task types.

Outputs (run dir, default rl_runs/plan):
  loto.log                         per-task + overall true KL on val prompts (5 per task)
  loto_val_t{T}.json               the same numbers as JSON
  allocs_loto/loto-blendplan_snapkv_t{T}.json   per-prompt allocations for the 260 RULER rows
  allocs_loto/loto-shareplan_snapkv_t{T}.json   per-prompt (per held-out task) static plan
  -> RULER with slurm/eval_alloc.sh, ALLOC_DIR=<run>/allocs_loto

alpha per budget is read from alpha_t{T}.json (tuned earlier on all tasks; one scalar, small leak,
stated in the report). Needs the cost tables from kv_plan.py --step collect (train + val).
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_alloc as ra          # noqa: E402
import kv_plan as kp           # noqa: E402

LV, NL, H, EPS = kp.LV, kp.NL, kp.H, kp.EPS


def train_cm(tr, va, dev, epochs, lr, seed):
    torch.manual_seed(seed)
    Xt, Yt = kp.to_xy(tr, dev)
    Xv, Yv = kp.to_xy(va, dev)
    cm = kp.CostModel().to(dev)
    opt = torch.optim.AdamW(cm.parameters(), lr=lr, weight_decay=1e-4)
    best, best_sd = 1e9, None
    for ep in range(1, epochs + 1):
        cm.train()
        perm = torch.randperm(len(Xt), device=dev)
        for j in range(0, len(perm), 16):
            b = perm[j:j + 16]
            loss = F.mse_loss(cm(Xt[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 0:
            cm.eval()
            with torch.no_grad():
                vl = F.mse_loss(cm(Xv), Yv).item()
            if vl < best:
                best, best_sd = vl, {k: v.clone() for k, v in cm.state_dict().items()}
    cm.load_state_dict(best_sd); cm.eval()
    return cm, best


def share_prior(recs):
    sh = [r["costs"][:, :NL - 1] / max(float(r["costs"][:, 0].sum()), EPS) for r in recs]
    return np.log(np.mean(sh, 0) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="rl_runs/plan")
    ap.add_argument("--targets", default="0.05+0.1")
    ap.add_argument("--model", default=os.path.expandvars("/scratch2/$USER/kvp/models/Qwen2.5-7B-Instruct"))
    ap.add_argument("--data_dir", default=os.path.expanduser("~/ml-learning-to-evict/data/gqa_safetensors"))
    ap.add_argument("--prompt_style", default="torchtune")
    ap.add_argument("--min_k", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_val_per_task", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_val", action="store_true", help="only write the RULER allocations")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    run = Path(args.run_dir)
    targets = [float(t) for t in args.targets.split("+")]
    logf = open(run / "loto.log", "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    log(f"=== loto {time.strftime('%Y-%m-%d %H:%M')} {vars(args)}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_all, va_all = kp.load(run, "train"), kp.load(run, "val")
    tasks = sorted({r["task"] for r in tr_all})
    alpha = {t: json.load(open(run / f"alpha_t{t}.json"))["alpha"] for t in targets}
    log(f"{len(tr_all)} train / {len(va_all)} val cost tables, {len(tasks)} tasks, alpha {alpha}")

    # 1) one cost model + one static prior per held-out task, learned without that task
    cms, priors, seen_cm = {}, {}, kp.CostModel().to(dev)
    seen_cm.load_state_dict(torch.load(run / "cost_model.pt", map_location=dev)); seen_cm.eval()
    seen_prior = share_prior(tr_all)
    for T in tasks:
        tr = [r for r in tr_all if r["task"] != T]
        va = [r for r in va_all if r["task"] != T]
        cms[T], mse = train_cm(tr, va, dev, args.epochs, args.lr, args.seed)
        priors[T] = share_prior(tr)
        log(f"[fit] held out {T:<16} train {len(tr)} prompts, val MSE (other tasks) {mse:.3f}")

    def lc(cm, feats):
        with torch.no_grad():
            return cm(feats.to(dev).float()[None])[0].cpu().numpy()

    def blend(cm, prior_lc, feats, a, t):
        return kp.plan(np.exp(a * lc(cm, feats) + (1 - a) * prior_lc), t)

    model, tok = ra.load_model(args.model)
    ds = ra.load_ruler()

    # 2) true KL on val prompts: every task scored only by the model that did not see it
    if not args.skip_val:
        names = ["loto_share", "loto_blend", "seen_share", "seen_blend"]
        res = {t: {k: [] for k in names} for t in targets}
        per_task = {t: {} for t in targets}
        for row in ra.rows_by_offset(ds, 405, args.n_val_per_task):
            ep = ra.make_episode(model, tok, row, args)
            if ep is None:
                continue
            T = row["task"]
            for t in targets:
                cand = np.stack([kp.plan(np.exp(priors[T]), t), blend(cms[T], priors[T], ep.feats, alpha[t], t),
                                 kp.plan(np.exp(seen_prior), t), blend(seen_cm, seen_prior, ep.feats, alpha[t], t)])
                r = -np.log(ep.kl(model, cand, args.batch_size) + EPS)
                for k, v in zip(names, r):
                    res[t][k].append(float(v))
                per_task[t].setdefault(T, []).append(float(r[1] - r[0]))
            log(f"[val] {row['prompt_id']} {T:<16} done")
        for t in targets:
            R = {k: np.array(v) for k, v in res[t].items()}
            d = R["loto_blend"] - R["loto_share"]
            g = R["seen_blend"] - R["loto_blend"]
            log(f"[val] target {t}: n={len(d)} mean -log KL (higher is better)  " +
                "  ".join(f"{k} {v.mean():.3f}" for k, v in R.items()))
            log(f"[val] target {t}: LOTO blend - LOTO share = {d.mean():+.3f} +- {d.std(ddof=1) / np.sqrt(len(d)):.3f} (SE); "
                f"generalization gap (seen blend - LOTO blend) = {g.mean():+.3f} +- {g.std(ddof=1) / np.sqrt(len(g)):.3f}")
            for T, v in sorted(per_task[t].items()):
                log(f"[val]   {t} {T:<16} blend-share {np.mean(v):+.3f} (n={len(v)})")
            json.dump(dict(target=t, alpha=alpha[t], mean={k: float(v.mean()) for k, v in R.items()},
                           blend_minus_share=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))),
                           per_task={T: float(np.mean(v)) for T, v in per_task[t].items()}),
                      open(run / f"loto_val_t{t}.json", "w"), indent=1)

    # 3) per-prompt allocations for the 260 RULER eval rows
    out = run / "allocs_loto"; out.mkdir(exist_ok=True)
    heads = [{"layer": h // 4, "kv_head": h % 4} for h in range(H)]
    pp = {t: {"blend": {}, "share": {}} for t in targets}
    rows = ra.eval_rows(ds, args.data_dir)
    for n, row in enumerate(rows):
        T, pid = row["task"], str(row["prompt_id"])
        try:
            ep = ra.Episode(model, tok, row, args.prompt_style)
            feats = ep.feats
        except AssertionError:
            feats = None
        for t in targets:
            share = kp.plan(np.exp(priors[T]), t)
            pp[t]["share"][pid] = [float(x) for x in share]
            a = share if feats is None else blend(cms[T], priors[T], feats, alpha[t], t)
            pp[t]["blend"][pid] = [float(x) for x in a]
        if n % 20 == 0:
            log(f"[dump] {n + 1}/{len(rows)}")
    for t in targets:
        for kind in ("blend", "share"):
            name = f"loto-{kind}plan_snapkv_t{t}"
            json.dump({"name": name, "kind": f"loto-{kind}plan", "scorer": "snapkv", "target_avg_keep_ratio": t,
                       "actual_avg_keep_ratio": t, "alpha": alpha[t] if kind == "blend" else 0.0,
                       "heads": heads, "per_prompt": pp[t][kind]}, open(out / f"{name}.json", "w"))
    log(f"[dump] wrote {len(targets) * 2} allocation files to {out}")


if __name__ == "__main__":
    main()
