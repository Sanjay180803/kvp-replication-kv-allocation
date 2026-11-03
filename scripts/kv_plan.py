#!/usr/bin/env python
"""Model-based allocation: learn each head's per-prompt cost curve, then PLAN with a greedy knapsack.

Factorized policies (RLOO, AWR, DPO, MBPO) pick each head's level on its own and only then squeeze
into the budget, which breaks the coupling between heads. The per-prompt oracle works because it
knows, for THIS prompt, how much KL each head costs at each budget level, then fills the budget
greedily. Here we learn that cost model from the prompt's features and keep the oracle's planner:
model-based RL with planning [EE599 Wk 5].

  collect   per-prompt single-head costs: KL when ONLY head h is shrunk to level l (others full)
  train     cost model: shared MLP over heads (features + head embedding) -> monotone log-costs
  evaluate  val prompts, TRUE joint KL: uniform / static / avg_plan / model_plan / oracle_plan
  dump      per-prompt allocations for the 260 RULER eval rows (+ avg_plan) -> eval_ruler_alloc.py
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
    if costs.shape[1] == NL - 1:
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

    def forward(self, feats):
        x = torch.cat([feats, self.emb.weight[None].expand(feats.shape[0], -1, -1)], -1)
        o = self.net(x)
        top = o[..., :1]
        steps = F.softplus(o[..., 1:])
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
    ap.add_argument("--step", required=True, choices=["collect", "train", "evaluate", "dump"])
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
                    li = int(np.searchsorted(LV, args.target - 1e-9))
                    sp = np.nanmean([np.corrcoef(pv[i, :, li].argsort().argsort().cpu(),
                                                 Yv[i, :, li].argsort().argsort().cpu())[0, 1] for i in range(len(Xv))])
                log(f"[train] epoch {ep} train {loss.item():.3f} val log-cost MSE {vl:.3f} "
                    f"val head-ranking Spearman {sp:.3f}")
                if vl < best:
                    best = vl; torch.save(cm.state_dict(), run / "cost_model.pt")
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

    def model_plan(feats):
        with torch.no_grad():
            lc = cm(feats.to(dev).float()[None])[0].cpu().numpy()
        return plan(np.exp(lc), args.target)

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
        base = max(summ["static"]["true_r"], summ["avg_plan"]["true_r"])
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
        avg = plan(np.mean([r["costs"] for r in load(run, "train")], 0), args.target)
        json.dump({"name": f"avgplan_snapkv_t{args.target}", "kind": "avgplan", "scorer": "snapkv",
                   "target_avg_keep_ratio": args.target, "actual_avg_keep_ratio": float(avg.mean()),
                   "heads": [dict(h, keep_ratio=float(r)) for h, r in zip(heads, avg)]},
                  open(run / f"avgplan_snapkv_t{args.target}.json", "w"))
        log(f"[dump] wrote {run}/{name}.json and avgplan_snapkv_t{args.target}.json")


if __name__ == "__main__":
    main()
