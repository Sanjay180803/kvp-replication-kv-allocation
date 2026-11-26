import json, os, random
os.chdir(os.path.expanduser("~/ml-learning-to-evict"))
SETS = [("4K", "results_walloc_L4096"), ("8K", "results_walloc_L8192"), ("16K", "results_walloc_L16384"), ("LB", "results_walloc_LLB")]
def load(d, m):
    f = f"{d}/ruler_{m}.jsonl"
    if not os.path.exists(f): return None
    out = {}
    for l in open(f):
        r = json.loads(l); out[r["id"]] = r["score"]
    return out
def seeds(d, base, r, prefix):            # prefix 'w' (Wenbo) or 'k' (allocator 3)
    names = [f"{prefix}{base}_s0_r{r}", f"{prefix}{base}_s1_r{r}", f"{prefix}{base}_s2_r{r}" if prefix == "k" else f"w{base}_r{r}"]
    return [load(d, n) for n in names]
def avg(runs):
    runs = [x for x in runs if x]
    if not runs: return None
    ids = set.intersection(*(set(x) for x in runs))
    return {i: sum(x[i] for x in runs) / len(runs) for i in ids}
def mean(x): return sum(x.values()) / len(x) if x else float("nan")
def ci(a, b, B=10000):
    if not a or not b: return "   n/a"
    ids = sorted(set(a) & set(b)); d = [a[i] - b[i] for i in ids]; n = len(d); rnd = random.Random(0)
    bs = sorted(sum(d[rnd.randrange(n)] for _ in range(n)) / n for _ in range(B))
    m, lo, hi = sum(d) / n, bs[int(.025 * B)], bs[int(.975 * B)]
    return f"{m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'*' if lo > 0 or hi < 0 else ' '}"
for r in ("0.05", "0.1"):
    print(f"\n######## budget {r} ########")
    for name, d in SETS:
        wf, wi = seeds(d, "fixed", r, "w"), seeds(d, "input", r, "w")
        kf, ki = seeds(d, "fixed", r, "k"), seeds(d, "input", r, "k")
        ws = load(d, f"wstatic_r{r}"); sp = load(d, f"team_shareplan_snapkv_t{r}__kvp"); ps = load(d, f"team_profile-share_snapkv_t{r}__kvp")
        fmt = lambda xs: " ".join(f"{mean(x):5.1f}" if x else "  -  " for x in xs)
        print(f"\n[{name}] wstatic {mean(ws):5.1f} | shareplan {mean(sp):5.1f} | profile-share {mean(ps):5.1f}")
        print(f"  wfixed s0 s1 s2: {fmt(wf)}  -> mean {mean(avg(wf)):5.1f}")
        print(f"  winput s0 s1 s2: {fmt(wi)}  -> mean {mean(avg(wi)):5.1f}")
        if any(kf): print(f"  kfixed s0 s1 s2: {fmt(kf)}  -> mean {mean(avg(kf)):5.1f}")
        if any(ki): print(f"  kinput s0 s1 s2: {fmt(ki)}  -> mean {mean(avg(ki)):5.1f}")
        for s in range(3):
            print(f"  winput-wfixed s{s}:        {ci(wi[s], wf[s])}")
        print(f"  winput-wfixed 3-seed avg: {ci(avg(wi), avg(wf))}")
        print(f"  wfixed(avg) - wstatic:    {ci(avg(wf), ws)}")
        print(f"  wfixed(avg) - shareplan:  {ci(avg(wf), sp)}")
        print(f"  winput(avg) - wstatic:    {ci(avg(wi), ws)}")
        if any(kf):
            print(f"  kinput-kfixed 3-seed avg: {ci(avg(ki), avg(kf))}")
            print(f"  kfixed(avg) - wstatic:    {ci(avg(kf), ws)}")
            print(f"  kfixed(avg) - wfixed(avg):{ci(avg(kf), avg(wf))}")
