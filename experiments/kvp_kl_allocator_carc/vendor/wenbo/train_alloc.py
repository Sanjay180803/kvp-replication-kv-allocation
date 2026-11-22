"""Train and apply the input-aware allocator on frozen KVP rankings (design: allocator_design_proposal.md).

Subcommands
  refs     best-merge reference (per prompt, true future attention) for a split -> refs_<split>.pt
  train    --mode fixed : one sharpness t_h and weight w_h per head (224 numbers), same for every input.
                          The split still moves with the input's KVP scores.
           --mode input : the same plus an input-dependent correction from a small network that reads all
                          112 heads' score summaries together (zero at initialisation).
           RL: contextual bandit, Gaussian policy over (log w_h, log t_h), RLOO with K samples per prompt,
           reward = KVP-style all-budgets attention reward (alloc_lib). Best checkpoint by VAL reward.
  dump     write eval_alloc.py JSON allocations for a split from: a checkpoint, the static split (fitted on
           TRAIN), raw-score merge (t = w = 1), or the best merge (needs labels; upper bound only).
  score    attention reward of saved allocations on a labelled split (no language model).

Leakage rules enforced here: training and model selection read only TRAIN and VAL labels (minus the held-out
task in leave-one-task-out mode); feature normalisation statistics come from TRAIN; `dump` for the test split
loads only KVP scores (need_fut=False), except for the explicit best-merge upper bound.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

import alloc_lib as al
from alloc_lib import H, NPRI, N_FEAT, TRAIN_RATIOS

EVAL_RATIOS = (0.05, 0.10, 0.25)


def split_ids(splits, name, loto_task=None):
    if loto_task is None:
        return splits[name]
    return splits["loto"][loto_task][name]


# ----------------------------------------------------------------------------- model
class Allocator(nn.Module):
    def __init__(self, mode, d=64, layers=2, head_emb=16, init_logstd=-1.2):
        super().__init__()
        self.mode = mode
        self.base = nn.Parameter(torch.zeros(H, 2))               # (log w_h, log t_h)
        self.logstd = nn.Parameter(torch.full((2,), init_logstd))
        self.register_buffer("fmu", torch.zeros(H, N_FEAT))
        self.register_buffer("fsd", torch.ones(H, N_FEAT))
        if mode == "input":
            self.emb = nn.Embedding(H, head_emb)
            self.inp = nn.Linear(N_FEAT + head_emb, d)
            enc = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.mix = nn.TransformerEncoder(enc, layers)
            self.out = nn.Linear(d, 2)
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, feat):                                       # feat [b, H, F] -> mu [b, H, 2]
        mu = self.base.expand(feat.shape[0], H, 2)
        if self.mode == "input":
            z = (feat - self.fmu) / self.fsd
            e = self.emb.weight.expand(feat.shape[0], H, -1)
            h = self.mix(self.inp(torch.cat([z, e], -1)))
            mu = mu + self.out(h)
        return mu


# ----------------------------------------------------------------------------- reward
def reward(ps, idx, ref_ev, logw, logt, ratios=TRAIN_RATIOS):
    """logw/logt [..., b, H] -> (reward [..., b], evicted share per budget [..., b, R])."""
    logv = al.values(ps.score[idx], ps.mask[idx], logw, logt)
    bud = ps.budgets(idx, ratios).expand(*logv.shape[:-2], len(ratios))
    n = al.counts_from_values(logv, bud)
    kept = al.kept_mass(ps.cumfut[idx], n)                         # [..., b, R]
    tot = ps.futtot[idx].sum(-1)[:, None]
    ev = tot - kept
    return -ev.sum(-1) / ref_ev[idx], ev / tot


@torch.no_grad()
def evaluate(model, ps, ref_ev, bs=16, ratios=TRAIN_RATIOS):
    model.eval()
    rs, evs = [], []
    for s in range(0, len(ps), bs):
        idx = torch.arange(s, min(s + bs, len(ps)), device=ps.device)
        mu = model(ps.feat[idx])
        r, ev = reward(ps, idx, ref_ev, mu[..., 0], mu[..., 1], ratios)
        rs.append(r); evs.append(ev)
    model.train()
    return torch.cat(rs).mean().item(), torch.cat(evs).mean(0).tolist()


# ----------------------------------------------------------------------------- refs
@torch.no_grad()
def make_refs(ps, ratios):
    ev = torch.zeros(len(ps), len(ratios), device=ps.device)
    counts = []
    for n in range(len(ps)):
        L = int(ps.P[n]) - NPRI
        c = al.best_merge(ps.cumfut[n], L, al.budgets_free(int(ps.P[n]), ratios))
        counts.append(c.cpu())
        kept = ps.cumfut[n].gather(1, c.T).sum(0)
        ev[n] = ps.futtot[n].sum() - kept
    return ev, counts


def load_refs(args, split, ps):
    path = Path(args.work) / f"refs_{split}.pt"
    if path.exists():
        d = torch.load(path)
        if d["ids"] == ps.ids:
            return d["ev"].to(ps.device)
    ev, _ = make_refs(ps, TRAIN_RATIOS)
    torch.save({"ids": ps.ids, "ev": ev.cpu(), "ratios": TRAIN_RATIOS}, path)
    return ev


# ----------------------------------------------------------------------------- train
def train(args):
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    splits = json.load(open(args.splits))
    tr_ids = split_ids(splits, "train", args.loto)
    va_ids = split_ids(splits, "val", args.loto)
    hw = torch.load(args.head_weights) if args.head_weights else None
    tr = al.PromptSet(Path(args.labels) / "train", tr_ids, dev, head_weights=hw)
    va = al.PromptSet(Path(args.labels) / "val", va_ids, dev, head_weights=hw)
    tag = (f"_loto-{args.loto}" if args.loto else "") + ("_hw" if args.head_weights else "")
    Path(args.work).mkdir(parents=True, exist_ok=True)
    ref_tr = load_refs(args, "train" + tag, tr).sum(-1)
    ref_va = load_refs(args, "val" + tag, va).sum(-1)

    model = Allocator(args.mode).to(dev)
    model.fmu.copy_(tr.feat.mean(0)); model.fsd.copy_(tr.feat.std(0).clamp_min(1e-3))
    if args.init:                                                  # e.g. start input-aware from the fixed model
        sd = torch.load(args.init, map_location=dev)["model"]
        model.base.data.copy_(sd["base"])
    opt = torch.optim.AdamW([{"params": [model.base], "lr": args.lr_base},
                             {"params": [p for n, p in model.named_parameters() if n != "base"], "lr": args.lr}],
                            weight_decay=0.0)
    out = Path(args.work) / f"{args.mode}{tag}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "w")
    best, best_step, bad = -1e9, 0, 0
    v0, ev0 = evaluate(model, va, ref_va)
    print(f"step 0 val_reward {v0:.4f} evicted@grid {[round(x, 4) for x in ev0]}", flush=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(tr), (args.batch,), device=dev)
        mu = model(tr.feat[idx])                                      # [b, H, 2]
        std = model.logstd.exp()
        a = mu.detach()[None] + std.detach() * torch.randn(args.K, *mu.shape, device=dev)
        with torch.no_grad():
            R, _ = reward(tr, idx, ref_tr, a[..., 0], a[..., 1])    # [K, b]
        adv = R - (R.sum(0, keepdim=True) - R) / (args.K - 1)      # leave-one-out baseline
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logp = torch.distributions.Normal(mu[None], std).log_prob(a).sum((-1, -2))
        loss = -(adv * logp).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            model.logstd.clamp_(math.log(args.min_std), 0.0)
        if step % args.eval_every == 0 or step == args.steps:
            v, ev = evaluate(model, va, ref_va)
            rec = {"step": step, "train_R_sampled": R.mean().item(), "val_reward": v, "val_evicted_grid": ev,
                   "std": model.logstd.exp().tolist(), "sec": round(time.time() - t0)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"step {step} train(sampled) {rec['train_R_sampled']:.4f} val {v:.4f} std {rec['std']} "
                  f"{rec['sec']}s", flush=True)
            if v > best + 1e-4:
                best, best_step, bad = v, step, 0
                torch.save({"model": model.state_dict(), "mode": args.mode, "step": step, "val": v,
                            "loto": args.loto, "args": vars(args)}, out / "best.pt")
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"early stop at {step} (best {best:.4f} @ {best_step})", flush=True)
                    break
    json.dump({"best_val_reward": best, "best_step": best_step, "start_val_reward": v0}, open(out / "summary.json", "w"))
    print(f"BEST val {best:.4f} at step {best_step} (start {v0:.4f}) -> {out}/best.pt", flush=True)


# ----------------------------------------------------------------------------- dump
@torch.no_grad()
def dump(args):
    dev = torch.device(args.device)
    splits = json.load(open(args.splits))
    ids = split_ids(splits, args.split, args.loto) if args.split != "test" or not args.loto else splits["loto"][args.loto]["test"]
    need_fut = args.source == "best"                    # only the labelled upper bound reads future attention
    hw = torch.load(args.head_weights) if args.head_weights else None
    ps = al.PromptSet(Path(args.labels) / args.split, ids, dev, need_fut=need_fut, head_weights=hw)
    ratios = tuple(args.ratios)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    per = {r: {} for r in ratios}
    if args.source == "ckpt":
        ck = torch.load(args.ckpt, map_location=dev)
        model = Allocator(ck["mode"]).to(dev); model.load_state_dict(ck["model"]); model.eval()
        src = list(range(len(ps)))
        if args.shuffle_feat:            # control: features of a similar-length prompt from another task
            order = sorted(range(len(ps)), key=lambda n: int(ps.P[n]))
            task = splits["task_of"]
            for k, n in enumerate(order):
                for d in (1, -1, 2, -2, 3, -3):
                    m = order[(k + d) % len(order)]
                    if task[str(ps.ids[m])] != task[str(ps.ids[n])]:
                        src[n] = m
                        break
        for n in range(len(ps)):
            idx = torch.tensor([n], device=dev)
            mu = model(ps.feat[torch.tensor([src[n]], device=dev)])
            logv = al.values(ps.score[idx], ps.mask[idx], mu[..., 0], mu[..., 1])
            c = al.counts_from_values(logv, ps.budgets(idx, ratios))[0]
            for k, r in enumerate(ratios):
                per[r][ps.ids[n]] = c[k].cpu()
    elif args.source == "raw":
        for n in range(len(ps)):
            idx = torch.tensor([n], device=dev)
            z = torch.zeros(1, H, device=dev)
            logv = al.values(ps.score[idx], ps.mask[idx], z, z)
            c = al.counts_from_values(logv, ps.budgets(idx, ratios))[0]
            for k, r in enumerate(ratios):
                per[r][ps.ids[n]] = c[k].cpu()
    elif args.source == "static":
        tr = al.PromptSet(Path(args.labels) / "train", split_ids(splits, "train", args.loto), dev, head_weights=hw)
        fr = al.static_split(tr, range(len(tr)), ratios)
        for n in range(len(ps)):
            c = al.counts_from_fractions(fr.cpu(), int(ps.P[n]), ratios)
            for k, r in enumerate(ratios):
                per[r][ps.ids[n]] = c[k]
    elif args.source == "best":
        for n in range(len(ps)):
            c = al.best_merge(ps.cumfut[n], int(ps.P[n]) - NPRI, al.budgets_free(int(ps.P[n]), ratios))
            for k, r in enumerate(ratios):
                per[r][ps.ids[n]] = c[k].cpu()
    for r in ratios:
        name = f"{args.name}_r{r}"
        al.save_allocation(out / f"{name}.json", name, r, per[r])
        print("wrote", out / f"{name}.json", flush=True)


@torch.no_grad()
def score(args):
    """Attention reward (evicted share per budget) of allocation JSONs on a labelled split."""
    dev = torch.device(args.device)
    splits = json.load(open(args.splits))
    for f in args.allocs:
        a = json.load(open(f))
        ids = [int(i) for i in a["per_prompt"]]
        hw = torch.load(args.head_weights) if args.head_weights else None
        ps = al.PromptSet(Path(args.labels) / args.split, ids, dev, head_weights=hw)
        ev = []
        for n, i in enumerate(ids):
            c = torch.tensor(a["per_prompt"][str(i)], device=dev) - NPRI
            kept = ps.cumfut[n].gather(1, c[:, None]).sum()
            ev.append(float(1 - kept / ps.futtot[n].sum()))
        print(f"{a['name']}: mean evicted share {sum(ev) / len(ev):.4f} over {len(ev)} prompts", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "dump", "score"])
    ap.add_argument("--splits", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--loto", default=None, help="held-out task name (leave-one-task-out)")
    ap.add_argument("--head_weights", default=None, help="tensor [112] from head_weights.py")
    # train
    ap.add_argument("--mode", choices=["fixed", "input"], default="fixed")
    ap.add_argument("--init", default=None)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_base", type=float, default=3e-2)
    ap.add_argument("--min_std", type=float, default=0.03)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    # dump / score
    ap.add_argument("--split", default="test")
    ap.add_argument("--source", choices=["ckpt", "raw", "static", "best"], default="ckpt")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--name", default="alloc")
    ap.add_argument("--ratios", type=float, nargs="+", default=list(EVAL_RATIOS))
    ap.add_argument("--out_dir", default="allocs")
    ap.add_argument("--allocs", nargs="*", default=[])
    ap.add_argument("--shuffle_feat", action="store_true")
    args = ap.parse_args()
    {"train": train, "dump": dump, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
