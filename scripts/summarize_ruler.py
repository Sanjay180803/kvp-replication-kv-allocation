"""Print RULER results: average score per method x keep-ratio, plus a per-task table.

Usage: uv run python scripts/summarize_ruler.py [results_dir]
"""
import json, sys
from collections import defaultdict
from pathlib import Path

rows = []
for f in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "results").glob("ruler_*.jsonl")):
    rows += [json.loads(l) for l in open(f)]

by, by_task = defaultdict(list), defaultdict(list)
for r in rows:
    k = (r["method"], r["ratio"])
    by[k].append(r["score"])
    by_task[(k, r["task"])].append(r["score"])

tasks = sorted({r["task"] for r in rows})
keys = sorted(by, key=lambda k: (k[1], k[0]))
print(f"{'method':<10}{'keep':>6}{'n':>6}{'avg':>8}")
for k in keys:
    v = by[k]
    print(f"{k[0]:<10}{k[1]:>6}{len(v):>6}{sum(v)/len(v):>8.1f}")

print("\nper task:")
print(f"{'task':<18}" + "".join(f"{k[0][:6]}@{k[1]:<5}".rjust(13) for k in keys))
for t in tasks:
    cells = []
    for k in keys:
        v = by_task.get((k, t), [])
        cells.append((f"{sum(v)/len(v):.1f}" if v else "-").rjust(13))
    print(f"{t:<18}" + "".join(cells))
