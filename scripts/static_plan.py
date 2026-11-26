"""Static split (wstatic), planned per prompt for that prompt's own length and budget.

  fit   average the share-normalised, head-weighted kept-attention curves over the TRAIN prompts
        -> static_curves.pt ([112, grid+1], about 1 MB). Needs train labels (future attention).
  plan  for each test prompt: plan the best split of its free slots on the averaged curves, for its own
        budget share, then write eval_alloc.py JSON. Reads only the prompt length from the labels.

Same idea as alloc_lib.static_split, but the budget share comes from each prompt's own length instead of
the train average, so it stays correct at 8K / 16K (the always-kept 20 tokens per head are a smaller share).
"""
import argparse
import json
from pathlib import Path

import torch

import alloc_lib as al
from alloc_lib import H, NPRI


def fit(a):
    sp = json.load(open(a.splits))
    hw = torch.load(a.head_weights)
    grid = a.grid
    g = torch.linspace(0, 1, grid + 1)
    acc = torch.zeros(H, grid + 1)
    ids = sp["train"]
    for i in ids:
        r = torch.load(Path(a.labels) / f"{i}.pt")
        f = r["fut"].float()[:, NPRI:] * hw[:, None]
        L = f.shape[1]
        cum = torch.cat([torch.zeros(H, 1), f.cumsum(1)], 1)          # cum[:, k] = attention kept with k slots
        acc += cum[:, (g * L).round().long()] / f.sum().clamp_min(1e-9)
    acc /= len(ids)
    torch.save({"acc": acc, "grid": grid, "n_train": len(ids)}, a.out)
    print("wrote", a.out, "from", len(ids), "train prompts", flush=True)


def plan(a):
    cv = torch.load(a.curves)
    acc, grid = cv["acc"].double(), cv["grid"]
    ids = json.load(open(a.ids))
    ids = list(ids.keys()) if isinstance(ids, dict) else list(ids)
    ratios = tuple(a.ratios)
    per = {r: {} for r in ratios}
    for i in ids:
        P = torch.load(Path(a.labels) / f"{i}.pt")["score"].shape[1]
        L = P - NPRI
        steps = torch.tensor([round(int(b) / L * grid) for b in al.budgets_free(P, ratios)])
        fr = al.best_merge(acc, grid, steps).float() / grid                 # [R, H] fractions of free slots
        c = al.counts_from_fractions(fr, P, ratios)
        for k, r in enumerate(ratios):
            per[r][i] = c[k]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for r in ratios:
        name = f"{a.name}_r{r}"
        al.save_allocation(out / f"{name}.json", name, r, per[r])
        print("wrote", out / f"{name}.json", len(per[r]), "prompts", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--splits", required=True)
    f.add_argument("--labels", required=True, help="folder with the train <id>.pt labels")
    f.add_argument("--head_weights", required=True)
    f.add_argument("--grid", type=int, default=2000)
    f.add_argument("--out", default="static_curves.pt")
    p = sub.add_parser("plan")
    p.add_argument("--curves", required=True)
    p.add_argument("--labels", required=True, help="folder with the test <id>.pt labels (only the length is read)")
    p.add_argument("--ids", required=True, help="JSON list of prompt IDs (or a dict keyed by ID)")
    p.add_argument("--ratios", type=float, nargs="+", default=[0.05, 0.1])
    p.add_argument("--name", default="wstatic")
    p.add_argument("--out_dir", default="allocs")
    a = ap.parse_args()
    fit(a) if a.cmd == "fit" else plan(a)


if __name__ == "__main__":
    main()
