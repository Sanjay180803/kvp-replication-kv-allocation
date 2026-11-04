#!/usr/bin/env python
"""Model-based allocation: learn each head's per-prompt cost curve, then PLAN with a greedy knapsack.

Why: factorized policies (RLOO, AWR, DPO, MBPO in kv_offline_rl.py) pick each head's level on its
own and only then squeeze into the budget, which breaks the coupling between heads. The per-prompt
oracle (rl_alloc.py --mode oracle) works because it knows, for THIS prompt, how much KL each head
costs at each budget level, and then fills the budget greedily. Here we learn that cost model from
the prompt's features and keep the oracle's planner: model-based RL with planning [EE599 Wk 5].

  collect   per-prompt single-head costs: KL when ONLY head h is shrunk to level l (all others
            full), for all 112 heads x levels < 1 -> labels [112, L-1]          (same as oracle)
  train     cost model: shared MLP over heads, input = head features + head embedding,
            output = log-cost at every level (monotone by construction)
  evaluate  on val prompts, TRUE joint KL for uniform / static / planned-from-model / oracle
            (planned from the true costs) -> how much of the oracle gap the model closes
  tune      shrinkage: plan on  alpha*model + (1-alpha)*prior  (in log-cost space), where the
            prior is the share-normalized average train cost table (a static allocation). alpha is
            picked per budget on val TRUE KL; alpha=0 is the static plan, alpha=1 the raw model.
  dump      per-prompt allocations for the 260 RULER eval rows -> eval_ruler_alloc.py
            (writes modelplan, avgplan, and with a tuned alpha also blendplan + shareplan)

Usage: slurm/kv_plan.sh (STEP=collect|train|evaluate|dump, TARGET=0.05)
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_alloc as ra          # noqa: E402
import make_allocation as ma   # noqa: E402

LV = np.asarray(ra.LEVELS)
NL = len(ra.LEVELS)
H = 112
EPS = 1e-6


def head_costs(model, ep, batch_size):
    """[H, NL] KL with only head h shrunk to level l (level 1.0 = 0)."""
    items = [(h, j) for h in range(H) for j in range(NL - 1)]
    ratios = np.ones((len(items), H))
    for b, (h, j) in enumerate(items):
        ratios[b, h] = LV[j]
    kl = ep.kl(model, ratios, batch_size)
    c = np.zeros((H, NL))
    for (h, j), v in zip(items, kl):
        c[h, j] = max(float(v), 0.0)
    return c


def plan(costs, target):
    """Greedy knapsack on a cost table [H, NL] (cummin'd) -> keep ratio per head (same as oracle)."""
    costs = np.asarray(costs, dtype=float)
    if costs.shape[1] == NL - 1:                                  # model output: no column for level 1.0
        costs = np.concatenate([costs, np.zeros((H, 1))], 1)
    df = pd.DataFrame(costs, index=pd.MultiIndex.from_tuples([(h // 4, h % 4) for h in range(H)]), columns=list(LV))
    df[1.0] = 0.0
    df = df.cummin(axis=1)
    alloc, _ = ma.greedy(df, target)
    return np.array([alloc[(h // 4, h % 4)] for h in range(H)])


class CostModel(nn.Module):
    """log-cost at levels 0..NL-2 for every head; monotone non-increasing in the level."""

    def __init__(self, n_feat=ra.N_FEAT, hid=128, emb=32):
        super().__init__()
        self.emb = nn.Embedding(H, emb)
        self.net = nn.Sequential(nn.Linear(n_feat + emb, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(),
                                 nn.Linear(hid, NL - 1))

    def forward(self, feats):                                     # [B,H,F] -> [B,H,NL-1] log-costs
        x = torch.cat([feats, self.emb.weight[None].expand(feats.shape[0], -1, -1)], -1)
        o = self.net(x)
        top = o[..., :1]                                          # cost at the smallest level
        steps = F.softplus(o[..., 1:])                            # decreases as the level grows
        return torch.cat([top, top - steps.cumsum(-1)], -1)


def load(run, split):
    recs = []
    for f in sorted(Path(run).glob(f"costs_{split}_shard*.pt")):
        recs += torch.load(f, weights_only=False)
    assert recs, f"no {split} cost data in {run}"
    return recs


def to_xy(recs, dev):
    X = torch.stack([r["feats"].float() for r in recs]).to(dev)
    Y = torch.stack([torch.log(torch.as_tensor(r["costs"][:, :NL - 1]).float() + EPS) for r in recs]).to(dev)
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["collect", "train", "evaluate", "tune", "dump"])
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--init_alloc", required=True)
    ap.add_argument("--run_dir", default="rl_runs/plan")
    ap.add_argument("--model", default=os.path.expandvars("/scratch2/$USER/kvp/models/Qwen2.5-7B-Instruct"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--min_k", type=int, default=3)
    ap.add_argument("--prompt_style", default="torchtune", choices=["torchtune", "hf"])
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--n_per_task", type=int, default=24)
    ap.add_argument("--n_val_per_task", type=int, default=5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data_dir", default=os.path.expanduser("~/ml-learning-to-evict/data/gqa_safetensors"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", default="0+0.1+0.2+0.3+0.5+0.7+1", help="'+'-separated grid for --step tune")
    ap.add_argument("--alpha", type=float, default=None, help="dump: override the tuned alpha")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    run = Path(args.run_dir); run.mkdir(parents=True, exist_ok=True)
    logf = open(run / f"{args.step}.log", "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    log(f"=== {args.step} {time.strftime('%Y-%m-%d %H:%M')} {vars(args)}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    budget = args.target * H
    static_lv = ra.alloc_to_levels(args.init_alloc, H)
    prior = LV[static_lv] + 1e-3 * np.arange(H)[::-1] / H

    if args.step == "train":
        tr, va = load(run, "train"), load(run, "val")
        Xt, Yt = to_xy(tr, dev); Xv, Yv = to_xy(va, dev)
        cm = CostModel().to(dev)
        opt = torch.optim.AdamW(cm.parameters(), lr=args.lr, weight_decay=1e-4)
        best = 1e9
        for ep in range(1, args.epochs + 1):
            perm = torch.randperm(len(Xt), device=dev)
            for j in range(0, len(perm), 16):
                b = perm[j:j + 16]
                loss = F.mse_loss(cm(Xt[b]), Yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
            if ep % 20 == 0:
                with torch.no_grad():
                    pv = cm(Xv)
                    vl = F.mse_loss(pv, Yv).item()
                    # ranking of heads by cost at the 5% level, within each prompt (what the planner uses)
                    li = int(np.searchsorted(LV, args.target - 1e-9))
                    sp = np.nanmean([np.corrcoef(pv[i, :, li].argsort().argsort().cpu(),
                                                 Yv[i, :, li].argsort().argsort().cpu())[0, 1] for i in range(len(Xv))])
                log(f"[train] epoch {ep} train {loss.item():.3f} val log-cost MSE {vl:.3f} "
                    f"val head-ranking Spearman {sp:.3f}")
                if vl < best:
                    best = vl; torch.save(cm.state_dict(), run / f"cost_model.pt")
        log(f"[train] best val MSE {best:.3f} -> {run}/cost_model.pt")
        return

    model, tok = ra.load_model(args.model)
    ds = ra.load_ruler()

    if args.step == "collect":
        rows = ra.rows_by_offset(ds, 410 if args.split == "train" else 405,
                                 args.n_per_task if args.split == "train" else args.n_val_per_task)
        rows = rows[args.shard::args.n_shards]
        out, t0 = [], time.time()
        for n, row in enumerate(rows):
            ep = ra.make_episode(model, tok, row, args)
            if ep is None:
                continue
            c = head_costs(model, ep, args.batch_size)
            out.append(dict(pid=int(row["prompt_id"]), task=row["task"], P=ep.P, K=ep.K,
                            feats=ep.feats.float().cpu().half(), costs=c.astype(np.float32)))
            if n % 5 == 0:
                log(f"[collect] {n + 1}/{len(rows)} {row['task']:<16} K={ep.K} "
                    f"({(time.time() - t0) / (n + 1):.1f}s/prompt)")
                torch.save(out, run / f"costs_{args.split}_shard{args.shard}.pt")
        torch.save(out, run / f"costs_{args.split}_shard{args.shard}.pt")
        log(f"[collect] saved {len(out)} prompts")
        return

    cm = CostModel().to(dev)
    cm.load_state_dict(torch.load(run / "cost_model.pt", map_location=dev)); cm.eval()

    def model_lc(feats):
        with torch.no_grad():
            return cm(feats.to(dev).float()[None])[0].cpu().numpy()     # [H, NL-1] log-costs

    def model_plan(feats):
        return plan(np.exp(model_lc(feats)), args.target)

    # prior = static cost table: each train prompt's costs divided by its own total at the lowest level
    # ("share", the normalization that made profile-share robust), averaged over prompts, in log space.
    # The planner is scale-free per prompt, so mixing log-costs of different scales only changes the
    # RELATIVE head ordering: alpha=0 -> static plan, alpha=1 -> the model's plan.
    def share_prior():
        sh = [r["costs"][:, :NL - 1] / max(float(r["costs"][:, 0].sum()), EPS) for r in load(run, "train")]
        return np.log(np.mean(sh, 0) + 1e-8)

    def blend_plan(feats, alpha, prior_lc):
        return plan(np.exp(alpha * model_lc(feats) + (1 - alpha) * prior_lc), args.target)

    if args.step == "tune":
        prior_lc = share_prior()
        alphas = [float(a) for a in args.alphas.replace(",", "+").split("+")]
        va = {r["pid"]: r for r in load(run, "val")}
        rows = ra.rows_by_offset(ds, 405, args.n_val_per_task)
        names = ["uniform", "static"] + [f"a={a:g}" for a in alphas] + ["oracle_plan"]
        res = {k: [] for k in names}
        for row in rows:
            ep = ra.make_episode(model, tok, row, args)
            if ep is None or int(row["prompt_id"]) not in va:
                continue
            cand = np.stack([np.full(H, args.target), LV[ra.project(static_lv, budget, prior)]]
                            + [blend_plan(ep.feats, a, prior_lc) for a in alphas]
                            + [plan(va[int(row["prompt_id"])]["costs"], args.target)])
            r = -np.log(ep.kl(model, cand, args.batch_size) + EPS)
            for k, v in zip(names, r):
                res[k].append(float(v))
            log(f"[tune] {row['prompt_id']} {row['task']:<16} " + "  ".join(f"{k} {v:.2f}" for k, v in zip(names, r)))
        R = {k: np.array(v) for k, v in res.items()}
        a0 = R[f"a={alphas[0]:g}"]
        log(f"[tune] n={len(a0)} prompts, target {args.target}. value = mean -log KL (higher is better); "
            f"diff = paired difference vs alpha={alphas[0]:g} (static plan) +- standard error")
        for k in names:
            d = R[k] - a0
            log(f"[tune]   {k:<12} {R[k].mean():7.3f}   diff {d.mean():+.3f} +- {d.std(ddof=1) / np.sqrt(len(d)):.3f}")
        best = max(alphas, key=lambda a: R[f"a={a:g}"].mean())
        log(f"[tune] best alpha at target {args.target}: {best:g}")
        json.dump(dict(target=args.target, alpha=best, mean={k: float(v.mean()) for k, v in R.items()}),
                  open(run / f"alpha_t{args.target}.json", "w"), indent=1)
        return

    if args.step == "evaluate":
        va = {r["pid"]: r for r in load(run, "val")}
        avg_alloc = plan(np.mean([r["costs"] for r in load(run, "train")], 0), args.target)
        rows = ra.rows_by_offset(ds, 405, args.n_val_per_task)
        res = {k: [] for k in ["uniform", "static", "avg_plan", "model_plan", "oracle_plan"]}
        for row in rows:
            ep = ra.make_episode(model, tok, row, args)
            if ep is None or int(row["prompt_id"]) not in va:
                continue
            true_c = va[int(row["prompt_id"])]["costs"]
            cand = np.stack([np.full(H, args.target), LV[ra.project(static_lv, budget, prior)], avg_alloc,
                             model_plan(ep.feats), plan(true_c, args.target)])
            r = -np.log(ep.kl(model, cand, args.batch_size) + EPS)
            for k, v in zip(res, r):
                res[k].append(float(v))
            log(f"[eval] {row['prompt_id']} {row['task']:<16} " + "  ".join(f"{k} {v:.3f}" for k, v in zip(res, r)))
        summ = {k: dict(true_r=float(np.mean(v)), mean_kl=float(np.mean(np.exp(-np.array(v)))), n=len(v))
                for k, v in res.items()}
        base = max(summ["static"]["true_r"], summ["avg_plan"]["true_r"])     # best static baseline
        gap = summ["oracle_plan"]["true_r"] - base
        closed = (summ["model_plan"]["true_r"] - base) / gap if gap > 0 else float("nan")
        log("[eval summary] " + json.dumps(summ) + f"  share of (best static)->oracle gap closed: {closed:.2f}")
        json.dump(dict(summ, gap_closed=closed), open(run / f"eval_t{args.target}.json", "w"), indent=1)
        return

    if args.step == "dump":
        rows = ra.eval_rows(ds, args.data_dir)
        pp = {}
        for n, row in enumerate(rows):
            try:
                ep = ra.Episode(model, tok, row, args.prompt_style)
                alloc = model_plan(ep.feats)
            except AssertionError:
                alloc = LV[ra.project(static_lv, budget, prior)]
            pp[str(row["prompt_id"])] = [float(x) for x in alloc]
            if n % 20 == 0:
                log(f"[dump] {n + 1}/{len(rows)}")
        heads = [{"layer": h // 4, "kv_head": h % 4} for h in range(H)]
        name = f"modelplan_snapkv_t{args.target}"
        json.dump({"name": name, "kind": "modelplan", "scorer": "snapkv", "target_avg_keep_ratio": args.target,
                   "actual_avg_keep_ratio": args.target, "heads": heads, "per_prompt": pp},
                  open(run / f"{name}.json", "w"))
        # the same planner on the AVERAGE training cost table = a static allocation from the same data
        avg = plan(np.mean([r["costs"] for r in load(run, "train")], 0), args.target)
        json.dump({"name": f"avgplan_snapkv_t{args.target}", "kind": "avgplan", "scorer": "snapkv",
                   "target_avg_keep_ratio": args.target, "actual_avg_keep_ratio": float(avg.mean()),
                   "heads": [dict(h, keep_ratio=float(r)) for h, r in zip(heads, avg)]},
                  open(run / f"avgplan_snapkv_t{args.target}.json", "w"))
        log(f"[dump] wrote {run}/{name}.json and avgplan_snapkv_t{args.target}.json")

        # shrinkage version (needs --step tune first, or --alpha); written to its own folder so
        # slurm/eval_alloc.sh can run ALLOC_DIR=<that folder> without picking up other JSONs
        af = run / f"alpha_t{args.target}.json"
        alpha = args.alpha if args.alpha is not None else (json.load(open(af))["alpha"] if af.exists() else None)
        if alpha is None:
            log("[dump] no tuned alpha -> skipping blendplan (run --step tune first)")
            return
        prior_lc = share_prior()
        out = run / "allocs_blend"; out.mkdir(exist_ok=True)
        pp = {}
        for row in rows:
            try:
                ep = ra.Episode(model, tok, row, args.prompt_style)
                a = blend_plan(ep.feats, alpha, prior_lc)
            except AssertionError:
                a = LV[ra.project(static_lv, budget, prior)]
            pp[str(row["prompt_id"])] = [float(x) for x in a]
        bname = f"blendplan_snapkv_t{args.target}"
        json.dump({"name": bname, "kind": "blendplan", "scorer": "snapkv", "target_avg_keep_ratio": args.target,
                   "actual_avg_keep_ratio": args.target, "alpha": alpha, "heads": heads, "per_prompt": pp},
                  open(out / f"{bname}.json", "w"))
        sp = plan(np.exp(prior_lc), args.target)
        json.dump({"name": f"shareplan_snapkv_t{args.target}", "kind": "shareplan", "scorer": "snapkv",
                   "target_avg_keep_ratio": args.target, "actual_avg_keep_ratio": float(sp.mean()),
                   "heads": [dict(h, keep_ratio=float(r)) for h, r in zip(heads, sp)]},
                  open(out / f"shareplan_snapkv_t{args.target}.json", "w"))
        log(f"[dump] alpha={alpha:g}: wrote {out}/{bname}.json and shareplan_snapkv_t{args.target}.json")


if __name__ == "__main__":
    main()
