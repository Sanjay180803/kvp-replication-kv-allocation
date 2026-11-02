"""Build per-KV-head budget allocations (keep ratio per head) under a fixed total memory budget.

Two kinds:
  uniform      every head keeps target x prompt_len tokens (same as eval_ruler.py)
  profile      greedy allocation from the head-profiling results: start every head at the lowest
               profiled keep ratio, then repeatedly upgrade the head with the largest KL reduction
               per extra token, until the average keep ratio over the 112 heads reaches the target.

Total memory is the same for both at a given target: sum over heads of keep_ratio == target x 112.
Levels are the keep ratios present in head_kl.csv plus 1.0 (KL = 0, nothing evicted).
Targets at or below the lowest profiled ratio need lower ratios profiled first, e.g.
  python scripts/profile_heads.py --n_per_task 5 --ratios 0.02,0.05,0.1,0.25,0.5

Usage:
  python scripts/make_allocation.py --profile results/profiling/head_kl.csv --targets 0.25 0.5
  python scripts/make_allocation.py --targets 0.25 0.5 --stat share     # robust to outlier prompts
Writes allocations/<kind>_<scorer>_t<target>.json
"""
import argparse, json
from pathlib import Path

import pandas as pd

N_LAYERS, N_KV = 28, 4


def head_costs(df, scorer, stat):
    """Per-head cost at each profiled keep ratio. stat: mean KL | median KL | per-prompt share."""
    d = df[df.scorer == scorer].copy()
    if stat == "share":  # each prompt's KL normalized by that prompt's total over all heads at the
        # LOWEST profiled ratio, so one prompt can't dominate and costs stay comparable across ratios
        lo = d.keep_ratio.min()
        tot = d[d.keep_ratio == lo].groupby("prompt_id").kl_mean.sum()
        d["v"] = d.kl_mean / d.prompt_id.map(tot).where(lambda x: x > 0, 1.0)
    else:
        d["v"] = d.kl_mean
    agg = "median" if stat == "median" else "mean"
    c = d.groupby(["layer", "kv_head", "keep_ratio"]).v.agg(agg).unstack("keep_ratio")
    c[1.0] = 0.0
    c = c[sorted(c.columns)]
    # enforce non-increasing cost as the budget grows (removes small noise inversions)
    c = c.cummin(axis=1)
    return c


def greedy(costs, target):
    levels = list(costs.columns)
    heads = list(costs.index)
    budget = target * len(heads)
    cur = {h: 0 for h in heads}  # index into levels
    spent = levels[0] * len(heads)
    if spent > budget + 1e-9:
        raise ValueError(f"target {target} is below the lowest profiled keep ratio {levels[0]}")
    while True:
        # look at jumps to ANY higher level, not just the next one: some heads (e.g. (4,1), (22,0))
        # only stop hurting at full cache, so a one-level-at-a-time greedy would never upgrade them
        best, best_j, best_gain = None, None, 0.0
        for h in heads:
            i = cur[h]
            for j in range(i + 1, len(levels)):
                step = levels[j] - levels[i]
                if spent + step > budget + 1e-9:
                    break
                gain = (costs.loc[h, levels[i]] - costs.loc[h, levels[j]]) / step
                if gain > best_gain:
                    best, best_j, best_gain = h, j, gain
        if best is None:
            break
        spent += levels[best_j] - levels[cur[best]]
        cur[best] = best_j
    alloc = {h: levels[cur[h]] for h in heads}
    grid_cost = sum(costs.loc[h, alloc[h]] for h in heads)
    # spread any leftover budget evenly below 1.0 so total memory matches the uniform allocation
    left = budget - sum(alloc.values())
    room = [h for h in heads if alloc[h] < 1.0]
    while left > 1e-9 and room:
        add = left / len(room)
        nxt = []
        for h in room:
            inc = min(add, 1.0 - alloc[h])
            alloc[h] += inc
            left -= inc
            if alloc[h] < 1.0 - 1e-9:
                nxt.append(h)
        room = nxt
    return alloc, grid_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="results/profiling/head_kl.csv")
    ap.add_argument("--scorer", default="snapkv")
    ap.add_argument("--stat", default="mean", choices=["mean", "median", "share"])
    ap.add_argument("--targets", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--out_dir", default="allocations")
    a = ap.parse_args()

    df = pd.read_csv(a.profile)
    costs = head_costs(df, a.scorer, a.stat)
    assert len(costs) == N_LAYERS * N_KV, f"expected 112 heads, got {len(costs)}"
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    for t in a.targets:
        uni_cost = sum(costs.loc[h, t] for h in costs.index) if t in costs.columns else float("nan")
        allocs = {"uniform": ({h: t for h in costs.index}, uni_cost)}
        try:
            allocs[f"profile-{a.stat}"] = greedy(costs, t)
        except ValueError as e:
            print("skip profile allocation:", e)
        for kind, (alloc, pred) in allocs.items():
            rows = [{"layer": int(l), "kv_head": int(h), "keep_ratio": round(float(r), 6)} for (l, h), r in sorted(alloc.items())]
            avg = sum(r["keep_ratio"] for r in rows) / len(rows)
            name = f"{kind}_{a.scorer}_t{t}"
            json.dump({"name": name, "kind": kind, "scorer": a.scorer, "stat": a.stat, "target_avg_keep_ratio": t,
                       "actual_avg_keep_ratio": avg, "heads": rows}, open(out / f"{name}.json", "w"), indent=1)
            full = sum(r["keep_ratio"] >= 1.0 for r in rows)
            full_heads = [(r["layer"], r["kv_head"]) for r in rows if r["keep_ratio"] >= 1.0]
            print(f"{name}: avg keep {avg:.3f}, predicted total {a.stat} cost {pred:.4g}, "
                  f"{full} heads at full cache: {full_heads}")


if __name__ == "__main__":
    main()
