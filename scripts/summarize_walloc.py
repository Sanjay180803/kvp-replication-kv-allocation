import json, glob, os, collections
os.chdir(os.path.expanduser("~/ml-learning-to-evict"))
def summarize(path, label):
    rows = collections.defaultdict(list); per_task = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted(glob.glob(f"{path}/ruler_*.jsonl")):
        for l in open(f):
            r = json.loads(l); m = os.path.basename(f)[6:-6]
            rows[m].append(r["score"]); per_task[m][r["task"]].append(r["score"])
    if not rows:
        print(f"{label}: nothing yet"); return None
    print(f"\n=== {label} ===")
    for m in sorted(rows):
        print(f"{m:<45} n={len(rows[m]):>3}  mean={sum(rows[m])/len(rows[m]):5.1f}")
    return per_task
for L in (4096, 8192, 16384):
    summarize(f"results_walloc_L{L}", f"RULER {L}")
pt = summarize("results_walloc_LLB", "LongBench 4K")
meta = glob.glob(f"/scratch2/{os.environ['USER']}/kvp/alloc_labels_LLB/test/meta_shard*.jsonl")
full = [json.loads(l) for f in meta for l in open(f)]
if full:
    print(f"{'full cache (from labels)':<45} n={len(full):>3}  mean={sum(r['full_score'] for r in full)/len(full):5.1f}")
if pt:
    for r in full:
        pt.setdefault("FULL_CACHE", collections.defaultdict(list))[r["task"]].append(r["full_score"])
    tasks = sorted({t for m in pt for t in pt[m]})
    print("\nLongBench per task:", " | ".join(tasks))
    for m in sorted(pt):
        print(f"{m:<45}", "  ".join(f"{sum(pt[m][t])/len(pt[m][t]):5.1f}" if pt[m][t] else "  -  " for t in tasks))
