import json, os, sys, random
os.chdir(os.path.expanduser("~/ml-learning-to-evict"))
def load(d, m):
    return {json.loads(l)["id"]: json.loads(l)["score"] for l in open(f"{d}/ruler_{m}.jsonl")}
def ci(d, a, b, B=10000, seed=0):
    A, Bm = load(d, a), load(d, b); ids = sorted(set(A) & set(Bm))
    diff = [A[i] - Bm[i] for i in ids]; n = len(diff); rnd = random.Random(seed)
    boots = sorted(sum(diff[rnd.randrange(n)] for _ in range(n)) / n for _ in range(B))
    m = sum(diff) / n; lo, hi = boots[int(.025 * B)], boots[int(.975 * B)]
    star = "*" if lo > 0 or hi < 0 else " "
    print(f"{d[16:]:<6} {a:<40} - {b:<40} {m:+5.1f} [{lo:+5.1f}, {hi:+5.1f}] {star} n={n}")
SP, PS = "team_shareplan_snapkv_t{r}__kvp", "team_profile-share_snapkv_t{r}__kvp"
for r in ("0.05", "0.1"):
    for L in ("L4096", "L8192", "L16384"):
        ci(f"results_walloc_{L}", f"wfixed_r{r}", f"raw_r{r}")
        ci(f"results_walloc_{L}", f"wfixed_r{r}", f"uniform_kvp_r{r}")
        ci(f"results_walloc_{L}", f"wfixed_r{r}", SP.format(r=r))
    d = "results_walloc_LLB"
    for b in (f"winput_r{r}", SP.format(r=r), PS.format(r=r), f"uniform_kvp_r{r}", f"raw_r{r}"):
        ci(d, f"wfixed_r{r}", b)
    ci(d, f"winput_r{r}", SP.format(r=r))
