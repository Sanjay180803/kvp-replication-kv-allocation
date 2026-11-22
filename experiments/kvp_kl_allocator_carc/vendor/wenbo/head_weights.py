"""Per-head importance weights for the attention reward (fix for the failed equal-weight go/no-go test).

alpha_h = (average share of output damage caused by head h when shrunk alone) /
          (average share of head h's future attention lost at the same keep level)
- Output damage: the team's single-head KL profiling (results/profiling_lowr/head_kl.csv): 65 prompts from
  RULER rows 400-404 (outside our train / val / test), keep levels 0.02-0.25, SnapKV ranking, KL to the
  full cache. Per prompt and level, KL is divided by the sum over heads ("share"), as in the team's shareplan.
- Attention lost: our KVP labels on a subset of TRAIN prompts, keeping the same fraction of tokens.
The weights are rescaled to mean 1. Output: torch tensor [112] saved to --out.
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

import torch

H, NPRI = 112, 20
LEVELS = (0.02, 0.05, 0.10, 0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiling_csv", nargs="+", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n_train", type=int, default=260)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude_task", default=None, help="leave-one-task-out: drop this task everywhere")
    a = ap.parse_args()

    kl = defaultdict(lambda: torch.zeros(H))
    rows = [r for f in a.profiling_csv for r in csv.DictReader(open(f))]
    seen = set()
    for r in rows:
        key = (r["prompt_id"], r["keep_ratio"], r["layer"], r["kv_head"], r["scorer"])
        if key in seen:
            continue
        seen.add(key)
        if r["scorer"] != "snapkv" or float(r["keep_ratio"]) not in LEVELS or r["task"] == a.exclude_task:
            continue
        kl[(r["prompt_id"], float(r["keep_ratio"]))][4 * int(r["layer"]) + int(r["kv_head"])] = float(r["kl_mean"])
    share = torch.zeros(len(LEVELS), H); cnt = torch.zeros(len(LEVELS))
    for (pid, lv), v in kl.items():
        if v.sum() > 0:
            share[LEVELS.index(lv)] += v / v.sum(); cnt[LEVELS.index(lv)] += 1
    share /= cnt[:, None]
    prof_ids = sorted({int(p) for p, _ in kl})
    sp = json.load(open(a.splits))
    assert not (set(prof_ids) & set(sp["train"] + sp["val"] + sp["test"])), "profiling prompts overlap our splits"

    pool = sp["loto"][a.exclude_task]["train"] if a.exclude_task else sp["train"]
    ids = pool[:: max(1, len(pool) // a.n_train)][: a.n_train]
    lost = torch.zeros(len(LEVELS), H)
    for i in ids:
        d = torch.load(Path(a.labels) / "train" / f"{i}.pt")
        P = d["fut"].shape[1]
        f = d["fut"].float()[:, NPRI:]
        cf = torch.cat([torch.zeros(H, 1), f.cumsum(1)], 1)
        tot = f.sum(1).clamp_min(1e-9)
        for k, lv in enumerate(LEVELS):
            n = max(0, int(lv * P) - NPRI)
            lost[k] += (tot - cf[:, n]) / tot
    lost /= len(ids)
    lost_share = lost / lost.sum(1, keepdim=True)
    alpha = share.mean(0) / lost_share.mean(0).clamp_min(1e-9)
    alpha = alpha / alpha.mean()
    torch.save(alpha, a.out)
    top = alpha.argsort(descending=True)[:8].tolist()
    print("profiling prompts:", len(prof_ids), "train prompts for attention:", len(ids))
    print("alpha: min %.3g median %.3g max %.3g" % (alpha.min(), alpha.median(), alpha.max()))
    print("top heads (layer,kv):", [(h // 4, h % 4, round(float(alpha[h]), 1)) for h in top])


if __name__ == "__main__":
    main()
