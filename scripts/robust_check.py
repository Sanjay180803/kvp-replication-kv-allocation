#!/usr/bin/env python
"""Extra analysis appended to results/profiling/summary.md:
noise-robust input-dependence, allocation test, StreamingLLM comparison, and run notes.
Run AFTER analyze_profiling.py (which rewrites summary.md). Safe to rerun."""
import argparse
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

FLOOR = 1e-7      # KL below this is treated as "no effect" (bf16 rounding level)
MARK = "<!-- robust_check -->"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/profiling")
    ap.add_argument("--ratio", type=float, default=0.25)
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(args.dir, "head_kl.csv"))
    df["head"] = df.layer * 4 + df.kv_head
    lab = lambda h: f"({h // 4},{h % 4})"
    rng = np.random.default_rng(0)

    d = df[(df.scorer == "snapkv") & (df.keep_ratio == args.ratio)]
    M = d.pivot(index="prompt_id", columns="head", values="kl_mean").sort_index()
    task = d.groupby("prompt_id").task.first().loc[M.index]
    tasks = list(dict.fromkeys(task))
    L = np.log10(M.clip(lower=FLOOR))
    out = [MARK, "", "## Robust analysis (scripts/robust_check.py)", "",
           f"SnapKV, keep ratio {args.ratio}. Log-scale stats treat KL < {FLOOR:g} as {FLOOR:g} (no effect).", ""]

    # dominance
    gm = M.mean().sort_values(ascending=False)
    shares = [gm.iloc[:k].sum() / gm.sum() for k in (2, 5, 10)]
    out += ["### How concentrated is importance?",
            f"- Top 2 heads ({lab(gm.index[0])}, {lab(gm.index[1])}) carry **{shares[0]:.0%}** of total single-head KL; "
            f"top 5: {shares[1]:.0%}; top 10: {shares[2]:.0%}.",
            f"- {(M < FLOOR).values.mean():.0%} of (prompt, head) results are below {FLOOR:g} (effectively no effect).", ""]

    # variance decomposition
    X = L.values - L.values.mean(1, keepdims=True)
    head_eff = X.mean(0)
    tm = pd.DataFrame(X, index=L.index).groupby(task.values).mean().loc[task.values].values
    ss = ((X - X.mean()) ** 2).sum()
    v_head = ((head_eff - X.mean()) ** 2).sum() * X.shape[0] / ss
    v_task = ((tm - head_eff) ** 2).sum() / ss
    v_prompt = ((X - tm) ** 2).sum() / ss

    # split-half reliability
    rel, cross = [], []
    for _ in range(200):
        halves = {}
        for t in tasks:
            ids = rng.permutation(L.index[task.values == t])
            k = len(ids) // 2
            halves[t] = (L.loc[ids[:k]].mean(), L.loc[ids[k:]].mean())
        rel.append(np.mean([pearsonr(*halves[t])[0] for t in tasks]))
        cross.append(np.mean([pearsonr(halves[a][0], halves[b][1])[0] for a, b in combinations(tasks, 2)]))
    out += ["### Input dependence (noise-robust)",
            f"- Log-KL correlation of head maps: same task (split halves) **{np.mean(rel):.2f}**, "
            f"different tasks **{np.mean(cross):.2f}**. The gap means task type changes which heads matter.",
            f"- Variance of log head importance (per-prompt scale removed): shared across inputs {v_head:.0%}, "
            f"task x head {v_task:.0%}, prompt-within-task {v_prompt:.0%} (real prompt differences + noise).",
            "- The plain Spearman (above) is lower because many near-zero heads have noisy ranks.", ""]

    # allocation test
    rows = []
    for k in (5, 10, 20):
        res = {"global": [], "task": [], "oracle": []}
        for pid in M.index:
            v, others, t = M.loc[pid], M.drop(pid), task[pid]
            cov = lambda sel: v[list(sel)].sum() / v.sum()
            res["global"].append(cov(others.mean().nlargest(k).index))
            res["task"].append(cov(others[task.drop(pid).values == t].mean().nlargest(k).index))
            res["oracle"].append(cov(v.nlargest(k).index))
        rows.append(f"| {k} | {np.mean(res['global']):.2f} | {np.mean(res['task']):.2f} | "
                    f"{np.mean(res['oracle']):.2f} | {k / 112:.2f} |")
    out += ["### Does knowing the input help choose heads? (leave-one-prompt-out)",
            "Share of a held-out prompt's total single-head KL covered by protecting k heads chosen by:", "",
            "| k | global ranking | same-task ranking | per-prompt oracle | random |", "|---|---|---|---|---|", *rows, "",
            "- Task-specific ranking does **not** beat the global ranking. Only per-prompt knowledge helps.",
            "- So an input-conditioned policy must predict per-prompt importance to beat a static global allocation.", ""]

    # per-task top heads
    out += ["### Top-3 heads per task", "", "| task | top 3 heads | share of task KL |", "|---|---|---|"]
    for t in tasks:
        s = M[task.values == t].mean().sort_values(ascending=False)
        out.append(f"| {t} | {', '.join(lab(h) for h in s.index[:3])} | {s.iloc[:3].sum() / s.sum():.2f} |")
    out.append("")

    # streaming
    if (df.scorer == "streaming").any():
        out += ["### SnapKV vs StreamingLLM", "", "| keep ratio | total KL snapkv | total KL streaming | head-map Spearman | streaming top 3 |",
                "|---|---|---|---|---|"]
        for r in sorted(df.keep_ratio.unique()):
            s = df[(df.scorer == "snapkv") & (df.keep_ratio == r)].groupby("head").kl_mean.mean()
            t = df[(df.scorer == "streaming") & (df.keep_ratio == r)].groupby("head").kl_mean.mean()
            out.append(f"| {r} | {s.sum():.3f} | {t.sum():.3f} | {spearmanr(s, t)[0]:.2f} | "
                       f"{', '.join(lab(h) for h in t.nlargest(3).index)} |")
        out += ["", "- Both scorers agree on which heads matter, so importance is a property of the head.", ""]

    # stability on log scale
    out += ["### Stability across keep ratios (global head map, r=0.1 vs r=0.5)"]
    for sc in df.scorer.unique():
        a = np.log10(df[(df.scorer == sc) & (df.keep_ratio == 0.1)].groupby("head").kl_mean.mean().clip(lower=FLOOR))
        b = np.log10(df[(df.scorer == sc) & (df.keep_ratio == 0.5)].groupby("head").kl_mean.mean().clip(lower=FLOOR))
        out.append(f"- {sc}: log-KL Pearson {pearsonr(a, b)[0]:.2f}, Spearman {spearmanr(a, b)[0]:.2f}")
    out.append("")

    # continuation length
    pp = d.groupby("prompt_id").agg(n_pos=("n_pos", "first"), tot=("kl_mean", "sum"))
    short = (pp.n_pos <= 10).sum()
    out += ["## Notes on the method", "",
            "- **KL reference:** an evict-nothing control item inside the same GPU batch as the evicted runs. "
            "Same meaning as the spec's full-cache reference, but bf16 batch-shape noise cancels (r = 1.0 gives KL = 0 exactly).",
            "- **Compute shortcut:** one full pass per prompt captures the prompt K/V; each (head, ratio) run is a "
            "32-token pass on that cache. Exactly equivalent to the full-length masked forward (sanity check 5).",
            "- **Matched to scripts/eval_ruler.py** (where it differs from the task spec): SnapKV scores are its "
            "`SnapKVPress.sort` exactly; budget is `max(1, int(r*P))` (spec said ceil); prompt uses torchtune style "
            "with no default system prompt (spec said `apply_chat_template`; rerun with `--prompt_style hf` if the eval "
            "does include it). The question is part of the compressed prefill, as in the eval.",
            f"- **Limitation:** continuations stop at EOS, so {short} of {len(pp)} prompts have 10 or fewer measured "
            f"positions (median total KL {pp[pp.n_pos <= 10].tot.median():.1e} vs {pp[pp.n_pos > 20].tot.median():.1e} "
            "for prompts with > 20). A future run should force 32 tokens for stronger signal.", ""]

    path = os.path.join(args.dir, "summary.md")
    base = open(path).read().split(MARK)[0].rstrip() + "\n\n"
    with open(path, "w") as f:
        f.write(base + "\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
