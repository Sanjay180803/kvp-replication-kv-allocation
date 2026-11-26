import json, os, random
os.chdir(os.path.expanduser("~/ml-learning-to-evict"))
SETS = [("4K", "results_walloc_L4096"), ("8K", "results_walloc_L8192"), ("16K", "results_walloc_L16384"), ("LB", "results_walloc_LLB")]
def load(d, m):
    f = f"{d}/ruler_{m}.jsonl"
    if not os.path.exists(f): return None
    return {json.loads(l)["id"]: json.loads(l)["score"] for l in open(f)}
def avg(runs):
    runs = [x for x in runs if x]
    if not runs: return None
    ids = set.intersection(*(set(x) for x in runs))
    return {i: sum(x[i] for x in runs) / len(runs) for i in ids}
def mean(x): return f"{sum(x.values())/len(x):5.1f}" if x else "  -  "
def ci(a, b, B=10000):
    if not a or not b: return "      n/a"
    ids = sorted(set(a) & set(b)); d = [a[i] - b[i] for i in ids]; n = len(d); rnd = random.Random(0)
    bs = sorted(sum(d[rnd.randrange(n)] for _ in range(n)) / n for _ in range(B))
    m, lo, hi = sum(d) / n, bs[int(.025 * B)], bs[int(.975 * B)]
    return f"{m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'*' if lo > 0 or hi < 0 else ' '}"
def methods(d, r):
    M = {"uniform": load(d, f"uniform_kvp_r{r}"), "raw": load(d, f"raw_r{r}"),
         "profile-share": load(d, f"team_profile-share_snapkv_t{r}__kvp"), "shareplan": load(d, f"team_shareplan_snapkv_t{r}__kvp"),
         "wstatic": load(d, f"wstatic_r{r}"), "kstatic_exact": load(d, f"kstatic_exact_r{r}"), "kstatic_approx": load(d, f"kstatic_r{r}")}
    M["wfixed"] = avg([load(d, f"wfixed_s0_r{r}"), load(d, f"wfixed_s1_r{r}"), load(d, f"wfixed_r{r}")])
    M["winput"] = avg([load(d, f"winput_s0_r{r}"), load(d, f"winput_s1_r{r}"), load(d, f"winput_r{r}")])
    M["kfixed"] = avg([load(d, f"kfixed_s{s}_r{r}") for s in range(3)])
    M["kinput"] = avg([load(d, f"kinput_s{s}_r{r}") for s in range(3)])
    return M
ORDER = ["uniform", "raw", "profile-share", "shareplan", "wstatic", "kstatic_approx", "kstatic_exact", "wfixed", "winput", "kfixed", "kinput"]
PAIRS = [("kfixed", "kstatic_exact"), ("kinput", "kstatic_exact"), ("kstatic_exact", "wstatic"), ("kfixed", "wstatic"),
         ("kfixed", "wfixed"), ("kinput", "kfixed"), ("wfixed", "wstatic"), ("winput", "wfixed"),
         ("kstatic_approx", "kstatic_exact"), ("wfixed", "raw")]
for r in ("0.05", "0.1"):
    allM = {n: methods(d, r) for n, d in SETS}
    print(f"\n######## budget {r}  (fixed/input = 3-seed mean) ########")
    print(f"{'method':<16}" + "".join(f"{n:>8}" for n, _ in SETS))
    for m in ORDER:
        print(f"{m:<16}" + "".join(f"{mean(allM[n][m]):>8}" for n, _ in SETS))
    print()
    for a, b in PAIRS:
        print(f"{a+' - '+b:<32}" + "  ".join(f"{n}:{ci(allM[n][a], allM[n][b])}" for n, _ in SETS))
