#!/usr/bin/env python
"""Turn results/profiling/head_kl.csv into heatmaps and summary.md."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

EPS = 1e-8


def head_grid(df, n_layers, n_kv):
    g = df.groupby(["layer", "kv_head"])["kl_mean"].mean()
    m = np.full((n_layers, n_kv), np.nan)
    for (L, h), v in g.items():
        m[L, h] = v
    return m


def plot_grid(ax, m, title, vmin, vmax):
    im = ax.imshow(np.clip(m, vmin, None), aspect="auto", cmap="magma", norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("KV head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(m.shape[1]))
    ax.set_yticks(range(0, m.shape[0], 3))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/profiling")
    ap.add_argument("--scorer", default="snapkv")
    ap.add_argument("--ref_ratio", type=float, default=0.25)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.dir, "head_kl.csv"))
    df = df[df.scorer == args.scorer]
    nL, nkv = df.layer.max() + 1, df.kv_head.max() + 1
    ratios = sorted(float(r) for r in df.keep_ratio.unique())
    tasks = list(dict.fromkeys(df.sort_values("prompt_id").task))
    vmax = df.kl_mean.max()
    vmin = max(df.kl_mean[df.kl_mean > 0].quantile(0.01), EPS)

    # 1. heatmap per ratio
    for r in ratios:
        fig, ax = plt.subplots(figsize=(3.2, 7))
        im = plot_grid(ax, head_grid(df[df.keep_ratio == r], nL, nkv), f"mean KL, keep r={r}", vmin, vmax)
        fig.colorbar(im, ax=ax, label="KL (nats)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.dir, f"heatmap_r{r}.png"), dpi=150)
        plt.close(fig)

    # 2. per-task panels at ref ratio
    sub = df[df.keep_ratio == args.ref_ratio]
    cols = 7
    fig, axes = plt.subplots(2, cols, figsize=(2.0 * cols, 10), gridspec_kw=dict(hspace=0.3))
    for i, (ax, t) in enumerate(zip(axes.flat, tasks)):
        im = plot_grid(ax, head_grid(sub[sub.task == t], nL, nkv), t, vmin, vmax)
        if i % cols:
            ax.set_ylabel("")
    for ax in list(axes.flat)[len(tasks):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, label="KL (nats)", shrink=0.6)
    fig.savefig(os.path.join(args.dir, f"heatmap_by_task_r{args.ref_ratio}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. rankings and input dependence
    overall = sub.groupby(["layer", "kv_head"])["kl_mean"].mean().sort_values(ascending=False)
    vec = {t: head_grid(sub[sub.task == t], nL, nkv).ravel() for t in tasks}
    C = pd.DataFrame(np.eye(len(tasks)), index=tasks, columns=tasks)
    for i, a in enumerate(tasks):
        for b in tasks[i + 1:]:
            C.loc[a, b] = C.loc[b, a] = spearmanr(vec[a], vec[b]).statistic
    off = C.values[~np.eye(len(tasks), dtype=bool)]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C.values, cmap="viridis", vmin=min(0, off.min()), vmax=1)
    ax.set_xticks(range(len(tasks)), tasks, rotation=90, fontsize=7)
    ax.set_yticks(range(len(tasks)), tasks, fontsize=7)
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            ax.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="w")
    ax.set_title(f"Spearman of 112-head importance between tasks (r={args.ref_ratio})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.dir, f"task_spearman_r{args.ref_ratio}.png"), dpi=150)
    plt.close(fig)

    # within-task (prompt-to-prompt) correlation, as a noise floor for the cross-task numbers
    pv = {pid: head_grid(g, nL, nkv).ravel() for pid, g in sub.groupby("prompt_id")}
    ptask = sub.groupby("prompt_id").task.first()
    within, across = [], []
    pids = list(pv)
    for i, a in enumerate(pids):
        for b in pids[i + 1:]:
            rho = spearmanr(pv[a], pv[b]).statistic
            (within if ptask[a] == ptask[b] else across).append(rho)

    # 4. stability across ratios
    stab = {}
    if 0.1 in ratios and 0.5 in ratios:
        v1 = head_grid(df[df.keep_ratio == 0.1], nL, nkv).ravel()
        v5 = head_grid(df[df.keep_ratio == 0.5], nL, nkv).ravel()
        stab["mean over prompts"] = spearmanr(v1, v5).statistic
        per = [spearmanr(head_grid(g[g.keep_ratio == 0.1], nL, nkv).ravel(),
                         head_grid(g[g.keep_ratio == 0.5], nL, nkv).ravel()).statistic
               for _, g in df.groupby("prompt_id")]
        stab["per prompt (mean)"] = float(np.mean(per))

    meta_p = os.path.join(args.dir, "run_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}

    def fmt(rows):
        out = ["| rank | layer | kv_head | mean KL | top1 agree |", "|---|---|---|---|---|"]
        agree = sub.groupby(["layer", "kv_head"])["top1_agree"].mean()
        for i, ((L, h), v) in enumerate(rows.items(), 1):
            out.append(f"| {i} | {L} | {h} | {v:.3e} | {agree[(L, h)]:.3f} |")
        return "\n".join(out)

    md = [f"# Per-KV-head sensitivity profile ({args.scorer})", "",
          f"{sub.prompt_id.nunique()} prompts, {len(tasks)} tasks, keep ratios {ratios}. "
          "KL(p_full || p_compressed) in nats, mean over 32 teacher-forced continuation positions.", "",
          f"## Top-10 most sensitive heads (r={args.ref_ratio})", fmt(overall.head(10)), "",
          f"## Bottom-10 least sensitive heads (r={args.ref_ratio})", fmt(overall.tail(10)[::-1]), "",
          "## Input dependence", "",
          f"Average pairwise Spearman between tasks' 112-head importance vectors: **{off.mean():.3f}** "
          f"(min {off.min():.3f}, max {off.max():.3f}).", "",
          f"Prompt-level noise floor: same-task prompt pairs average {np.mean(within):.3f}, "
          f"cross-task prompt pairs average {np.mean(across):.3f}. "
          "If cross-task is clearly below same-task, head importance depends on the input type.", "",
          f"![task correlation](task_spearman_r{args.ref_ratio}.png)", "",
          C.round(2).to_markdown(), "",
          "## Stability across keep ratios (Spearman, r=0.1 vs r=0.5)", "",
          *[f"- {k}: {v:.3f}" for k, v in stab.items()], "",
          "## Run", "",
          f"- GPU: {meta.get('gpu', '?')}",
          f"- Runtime: {meta.get('runtime_s', '?')} s (last invocation)",
          f"- Model: {meta.get('model', '?')}, torch {meta.get('torch', '?')}, transformers {meta.get('transformers', '?')}", ""]
    with open(os.path.join(args.dir, "summary.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
