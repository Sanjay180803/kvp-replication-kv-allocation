#!/usr/bin/env python
"""RL policy that splits a fixed KV-cache budget across the 112 KV heads of Qwen2.5-7B-Instruct,
PER INPUT. Built on profile_heads.py (same model hook, SnapKV scores and batched KL machinery).

Episode (one prompt = one contextual-bandit step)
  1. Full-cache reference pass on the prompt (profile_heads.reference_pass): greedy continuation
     c_1..c_K (K <= 32), post-RoPE prompt K/V and SnapKV scores for every layer/head.
  2. State: per-head features from the SnapKV score distribution (share of attention mass kept at
     each budget level, entropy, prompt length) + a learned embedding of the head's identity.
  3. Action: one budget level per head from LEVELS (fraction of prompt tokens that head keeps;
     1.0 = uncompressed). A deterministic, policy-independent projection then enforces the total
     budget: average keep ratio over the 112 heads == --target (downgrade the largest heads first,
     least-important first by the static prior; fill leftover budget most-important first).
  4. Reward: -log(KL) between full-cache and compressed next-token distributions over the K
     teacher-forced continuation tokens (matched prefix), measured against an evict-nothing control
     item in the same GPU batch (profile_heads.kl_vs_control), so bf16 noise cancels.
  5. RLOO: G sampled allocations per prompt in one batched pass; advantage = reward minus the
     mean of the other G-1 samples, normalized per prompt. Loss = -A * log pi(a) - entropy bonus.

Modes
  train   : train the policy; evaluates on validation prompts every --eval_every steps against
            uniform and the static allocation (--init_alloc); saves best checkpoint.
  oracle  : per-prompt greedy upper bound on validation prompts (each prompt's own single-head
            costs at every level -> greedy split), next to uniform / static. No training.
  dump    : write per-prompt allocations for the 260-row RULER eval set (same rows as
            eval_ruler.py) from a trained checkpoint, for scripts/eval_ruler_alloc.py.

Data (RULER-4K, task t = rows 500t..500t+499; first 400 rows of each block hold KVP training and
eval rows): train = 500t+410.., val = 500t+405..409, profiling used 500t+400..404.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_heads as ph  # noqa: E402  (registers the custom attention)

LEVELS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
N_FEAT = len(LEVELS) - 1 + 3   # mass kept at each level < 1, entropy, sqrt(max mass), log P
N_TASKS, BLOCK = 13, 500


def load_ruler():
    from datasets import load_dataset
    ds = load_dataset("simonjegou/ruler", data_dir="4096", split="test")
    assert len(ds) == N_TASKS * BLOCK
    return ds


def rows_by_offset(ds, first, n_per_task):
    out = []
    for t in range(N_TASKS):
        for j in range(n_per_task):
            i = BLOCK * t + first + j
            assert 400 <= i % BLOCK < BLOCK
            out.append(dict(ds[i], prompt_id=i))
    return out


def eval_rows(ds, data_dir, per_task=20):
    """Exactly the 260 rows eval_ruler.py / eval_ruler_alloc.py use."""
    used = {int(p.name.split("_")[-1]) for p in Path(data_dir).glob("**/sample_*")}
    assert used, f"no training sample folders under {data_dir}; pass --data_dir"
    picked, count = [], {}
    for i, task in enumerate(ds["task"]):
        if i in used or count.get(task, 0) >= per_task:
            continue
        count[task] = count.get(task, 0) + 1
        picked.append(i)
    return [dict(ds[i], prompt_id=i) for i in picked]


class Episode:
    """Reference pass + everything needed to score many allocations for one prompt."""

    def __init__(self, model, tok, row, prompt_style, ids=None):
        if ids is None:
            ids = tok(ph.build_prompt(tok, row, prompt_style), return_tensors="pt",
                      add_special_tokens=False).input_ids.to(model.device)
        self.row, self.P = row, ids.shape[1]
        self.cont, self.K, _, self.tf_match = ph.reference_pass(model, tok, ids)
        nL = model.config.num_hidden_layers
        S = torch.stack([ph.State.snap_scores[L] for L in range(nL)])          # [nL, nkv, P], window = inf
        self.nL, self.nkv = S.shape[0], S.shape[1]
        order = S.argsort(dim=-1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(-1, order, torch.arange(self.P, device=S.device).expand_as(order))
        self.rank = rank                                                       # 0 = most important
        self.feats = head_features(S, self.P)                                  # [nL*nkv, F]

    def evict_mask(self, ratios):
        """ratios [B, nL*nkv] keep ratios -> bool [B, nL, nkv, P], True = evicted."""
        r = torch.as_tensor(ratios, dtype=torch.float32, device=self.rank.device).view(-1, self.nL, self.nkv)
        n_keep = torch.where(r >= 1.0, torch.full_like(r, self.P),
                             torch.clamp((r * self.P).floor(), min=1)).long()  # same rule as eval
        return self.rank[None] >= n_keep[..., None]

    @torch.no_grad()
    def kl(self, model, ratios, batch_size):
        """KL (mean over the K continuation positions) for each allocation in `ratios`."""
        out = []
        ratios = np.asarray(ratios)
        step = max(1, batch_size - 1)
        for i in range(0, len(ratios), step):
            ev = self.evict_mask(ratios[i:i + step])
            klm, _, _ = ph.kl_vs_control(model, self.cont[:self.K], self.P, ev)
            out.append(klm.float().cpu())
        return torch.cat(out).numpy()


def make_episode(model, tok, row, args):
    """Episode or None (answers shorter than --min_k tokens give too few KL positions)."""
    try:
        ep = Episode(model, tok, row, args.prompt_style)
    except AssertionError:
        return None
    return ep if ep.K >= args.min_k else None


def head_features(S, P, window=ph.WINDOW):
    """Per-head features from SnapKV scores S [nL, nkv, P] (window tokens are +inf, excluded)."""
    nL, nkv, _ = S.shape
    n = max(1, P - window)
    s = S[..., :n].float().clamp_min(0)
    mass = s / s.sum(-1, keepdim=True).clamp_min(1e-12)
    cum = mass.sort(dim=-1, descending=True).values.cumsum(-1)
    feats = []
    for r in LEVELS[:-1]:                     # share of non-window attention mass kept at level r
        k = int(r * P) - window
        feats.append(torch.zeros_like(cum[..., 0]) if k <= 0 else cum[..., min(k, n) - 1])
    ent = -(mass * (mass + 1e-12).log()).sum(-1) / math.log(n)
    feats += [ent, mass.max(-1).values.sqrt(), torch.full_like(ent, math.log(P) / 10)]
    return torch.stack(feats, -1).view(nL * nkv, -1)


def project(a, budget, prior):
    """a: int level indices [n_heads]. Downgrade until sum(LEVELS[a]) <= budget (largest level
    first, least important by `prior` first), then fill leftover (most important first)."""
    lv = np.asarray(LEVELS)
    a = a.copy()
    while lv[a].sum() > budget + 1e-9:
        cand = np.where(a > 0)[0]
        top = cand[a[cand] == a[cand].max()]
        h = top[np.argmin(prior[top])]
        a[h] -= 1
    up = np.argsort(-prior, kind="stable")
    changed = True
    while changed:
        changed = False
        for h in up:
            if a[h] + 1 < len(lv) and lv[a].sum() - lv[a[h]] + lv[a[h] + 1] <= budget + 1e-9:
                a[h] += 1
                changed = True
                break
    return a


def alloc_to_levels(alloc_json, n_heads):
    """Map a make_allocation.py JSON to level indices (largest level <= the head's keep ratio)."""
    heads = sorted(json.load(open(alloc_json))["heads"], key=lambda h: (h["layer"], h["kv_head"]))
    assert len(heads) == n_heads
    lv = np.asarray(LEVELS)
    return np.array([int(np.searchsorted(lv, h["keep_ratio"] + 1e-9) - 1) for h in heads]).clip(0)


class Policy(nn.Module):
    def __init__(self, n_heads, n_feat, init_levels=None, init_scale=2.0, hidden=128, emb=32):
        super().__init__()
        self.emb = nn.Embedding(n_heads, emb)
        self.mlp = nn.Sequential(nn.Linear(n_feat + emb, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, len(LEVELS)))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        bias = torch.zeros(n_heads, len(LEVELS))
        if init_levels is not None:               # start as the static allocation
            bias[torch.arange(n_heads), torch.as_tensor(init_levels)] = init_scale
        self.bias = nn.Parameter(bias)

    def forward(self, feats):
        ids = torch.arange(feats.shape[0], device=feats.device)
        return self.bias + self.mlp(torch.cat([feats, self.emb(ids)], -1))


def load_model(path, dtype="bfloat16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype),
                                                 attn_implementation=ph.ATTN_NAME, device_map="cuda")
    return model.eval(), tok


def evaluate(model, tok, policy, rows, args, budget, prior, static_lv, log):
    """Mean log-KL on `rows` for uniform, static and the policy's greedy (argmax + projection)."""
    n_heads = len(prior)
    res = {"uniform": [], "static": [], "policy": []}
    for row in rows:
        ep = make_episode(model, tok, row, args)
        if ep is None:
            continue
        with torch.no_grad():
            greedy = policy(ep.feats).argmax(-1).cpu().numpy()
        lv = np.asarray(LEVELS)
        cand = np.stack([np.full(n_heads, args.target),
                         lv[project(static_lv, budget, prior)],
                         lv[project(greedy, budget, prior)]])
        kl = ep.kl(model, cand, args.batch_size)
        for k, v in zip(res, kl):
            res[k].append(float(v))
    out = {k: dict(mean_kl=float(np.mean(v)), mean_logkl=float(np.mean(np.log(np.array(v) + 1e-6))), n=len(v))
           for k, v in res.items()}
    log(f"[eval] " + "  ".join(f"{k}: KL {d['mean_kl']:.4g} logKL {d['mean_logkl']:.3f}" for k, d in out.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "oracle", "dump"], default="train")
    ap.add_argument("--model", default=os.path.expandvars("/scratch2/$USER/kvp/models/Qwen2.5-7B-Instruct"))
    ap.add_argument("--target", type=float, default=0.05, help="average keep ratio over the 112 heads")
    ap.add_argument("--init_alloc", default=None, help="static allocation JSON (start point + projection prior)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--n_train_per_task", type=int, default=90)
    ap.add_argument("--n_val_per_task", type=int, default=5)
    ap.add_argument("--group", type=int, default=8, help="RLOO samples per prompt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent_coef", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=16, help="allocations per KL forward (incl. control)")
    ap.add_argument("--min_k", type=int, default=3, help="skip prompts whose answer is shorter")
    ap.add_argument("--prompt_style", default="torchtune", choices=["torchtune", "hf"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data_dir", default=os.path.expanduser("~/ml-learning-to-evict/data/gqa_safetensors"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out_dir or f"rl_runs/t{args.target}"); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / f"{args.mode}.log", "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    log(f"=== {args.mode} {time.strftime('%Y-%m-%d %H:%M')} args={vars(args)}")
    model, tok = load_model(args.model)
    n_heads = model.config.num_hidden_layers * model.config.num_key_value_heads
    budget = args.target * n_heads
    assert LEVELS[0] * n_heads <= budget, "target below the smallest level"
    static_lv = alloc_to_levels(args.init_alloc, n_heads) if args.init_alloc else \
        np.full(n_heads, int(np.searchsorted(LEVELS, args.target + 1e-9) - 1))
    prior = np.asarray(LEVELS)[static_lv] + 1e-3 * np.arange(n_heads)[::-1] / n_heads   # importance order
    ds = load_ruler()
    val = rows_by_offset(ds, 405, args.n_val_per_task)
    dev = model.device

    if args.mode == "oracle":
        import pandas as pd
        import make_allocation as ma
        res = {"uniform": [], "static": [], "oracle": []}
        lv = np.asarray(LEVELS)
        for row in val:
            ep = make_episode(model, tok, row, args)
            if ep is None:
                continue
            items = [(h, j) for h in range(n_heads) for j in range(len(LEVELS) - 1)]
            ratios = np.full((len(items), n_heads), 1.0)
            for b, (h, j) in enumerate(items):
                ratios[b, h] = LEVELS[j]
            kl = ep.kl(model, ratios, args.batch_size)
            c = pd.DataFrame(1.0, index=pd.MultiIndex.from_tuples([(h // 4, h % 4) for h in range(n_heads)]),
                             columns=LEVELS)
            for (h, j), v in zip(items, kl):
                c.iloc[h, j] = v
            c[1.0] = 0.0
            c = c.cummin(axis=1)
            alloc, _ = ma.greedy(c, args.target)
            orc = np.array([alloc[(h // 4, h % 4)] for h in range(n_heads)])
            cand = np.stack([np.full(n_heads, args.target), lv[project(static_lv, budget, prior)], orc])
            k3 = ep.kl(model, cand, args.batch_size)
            for k, v in zip(res, k3):
                res[k].append(float(v))
            log(f"[oracle] prompt {row['prompt_id']} {row['task']:<16} K={ep.K} "
                f"uniform {k3[0]:.4g} static {k3[1]:.4g} oracle {k3[2]:.4g} (full heads: {int((orc >= 1).sum())})")
        summ = {k: dict(mean_kl=float(np.mean(v)), mean_logkl=float(np.mean(np.log(np.array(v) + 1e-6))), n=len(v))
                for k, v in res.items()}
        log("[oracle summary] " + json.dumps(summ))
        json.dump(summ, open(out / "oracle_summary.json", "w"), indent=1)
        return

    policy = Policy(n_heads, N_FEAT, init_levels=static_lv if args.init_alloc else None).to(dev)
    if args.ckpt:
        policy.load_state_dict(torch.load(args.ckpt, map_location=dev))

    if args.mode == "dump":
        lv = np.asarray(LEVELS)
        rows = eval_rows(ds, args.data_dir)
        per_prompt, static_pp = {}, {}
        for n, row in enumerate(rows):
            try:
                ep = Episode(model, tok, row, args.prompt_style)
                with torch.no_grad():
                    greedy = policy(ep.feats).argmax(-1).cpu().numpy()
            except AssertionError:          # 1-token answer: no continuation to read; use static
                greedy = static_lv
            per_prompt[str(row["prompt_id"])] = [float(x) for x in lv[project(greedy, budget, prior)]]
            static_pp[str(row["prompt_id"])] = [float(x) for x in lv[project(static_lv, budget, prior)]]
            if n % 20 == 0:
                log(f"[dump] {n + 1}/{len(rows)}")
        for name, pp in [("rl", per_prompt), ("static-proj", static_pp)]:
            heads = [{"layer": h // 4, "kv_head": h % 4} for h in range(n_heads)]
            json.dump({"name": f"{name}_snapkv_t{args.target}", "kind": name, "scorer": "snapkv",
                       "target_avg_keep_ratio": args.target, "actual_avg_keep_ratio": args.target,
                       "heads": heads, "per_prompt": pp},
                      open(out / f"{name}_snapkv_t{args.target}.json", "w"))
        log(f"[dump] wrote {out}/rl_snapkv_t{args.target}.json and static-proj_snapkv_t{args.target}.json")
        return

    train = rows_by_offset(ds, 410, args.n_train_per_task)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    lv = np.asarray(LEVELS)
    best, step = float("inf"), 0
    hist = open(out / "train_steps.jsonl", "a")
    ev = evaluate(model, tok, policy, val, args, budget, prior, static_lv, log)
    json.dump({"step": 0, **ev}, open(out / "eval_step0.json", "w"), indent=1)
    best = ev["policy"]["mean_logkl"]                     # the starting policy counts as a candidate
    torch.save(policy.state_dict(), out / "best.pt")
    for epoch in range(args.epochs):
        random.shuffle(train)
        for row in train:
            t0 = time.time()
            ep = make_episode(model, tok, row, args)
            if ep is None:
                continue
            logits = policy(ep.feats) / args.temperature                       # [n_heads, L]
            dist = torch.distributions.Categorical(logits=logits)
            acts = dist.sample((args.group,))                                  # [G, n_heads]
            logp = dist.log_prob(acts).sum(-1)                                 # [G]
            proj = np.stack([project(a, budget, prior) for a in acts.cpu().numpy()])
            overshoot = np.maximum(lv[acts.cpu().numpy()].sum(-1) - budget, 0) / budget
            kl = ep.kl(model, lv[proj], args.batch_size)
            r = torch.tensor(-np.log(kl + 1e-6) - 0.5 * overshoot, dtype=torch.float32, device=dev)
            G = args.group
            base = (r.sum() - r) / (G - 1)
            adv = r - base
            adv = adv / (adv.std() + 1e-6)
            loss = -(adv.detach() * logp).mean() - args.ent_coef * dist.entropy().sum(-1).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
            rec = dict(step=step, prompt_id=row["prompt_id"], task=row["task"], P=ep.P, K=ep.K,
                       kl_mean=float(kl.mean()), kl_best=float(kl.min()), overshoot=float(overshoot.mean()),
                       full_heads=float((lv[proj] >= 1).sum(-1).mean()), sec=round(time.time() - t0, 2))
            hist.write(json.dumps(rec) + "\n"); hist.flush()
            if step % 10 == 0:
                log(f"[train] step {step} {row['task']:<16} KL mean {rec['kl_mean']:.4g} best {rec['kl_best']:.4g} "
                    f"overshoot {rec['overshoot']:.2f} full-heads {rec['full_heads']:.1f} {rec['sec']}s")
            if step % args.eval_every == 0:
                ev = evaluate(model, tok, policy, val, args, budget, prior, static_lv, log)
                json.dump({"step": step, **ev}, open(out / f"eval_step{step}.json", "w"), indent=1)
                torch.save(policy.state_dict(), out / "last.pt")
                if ev["policy"]["mean_logkl"] < best:
                    best = ev["policy"]["mean_logkl"]
                    torch.save(policy.state_dict(), out / "best.pt")
                    log(f"[ckpt] new best policy logKL {best:.3f}")
    ev = evaluate(model, tok, policy, val, args, budget, prior, static_lv, log)
    json.dump({"step": step, **ev}, open(out / "eval_final.json", "w"), indent=1)
    torch.save(policy.state_dict(), out / "last.pt")
    if ev["policy"]["mean_logkl"] < best:
        torch.save(policy.state_dict(), out / "best.pt")


if __name__ == "__main__":
    main()
