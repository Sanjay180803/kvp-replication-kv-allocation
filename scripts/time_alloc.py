"""Overhead of the final allocator's decision step, on real KVP scores from our test labels.
Steps timed per prompt (GPU, CUDA-synchronized): per-head calibration (sharpness-scaled softmax x head weight)
+ one global top-B over all 112 x P scores + per-head counts. KVP scores themselves are computed for eviction anyway."""
import glob, os, time, torch
torch.manual_seed(0); dev = "cuda"
w = torch.rand(112, device=dev) + 0.5; s = torch.rand(112, device=dev) + 0.5      # stand-in calibration params
def step(S, B):
    p = torch.softmax(S * s[:, None], dim=-1) * w[:, None]
    idx = torch.topk(p.flatten(), B).indices
    return torch.bincount(idx // S.shape[1], minlength=112)
print(f"GPU {torch.cuda.get_device_name()}")
for L in (4096, 8192, 16384):
    files = sorted(glob.glob(f"/scratch2/{os.environ['USER']}/kvp/alloc_labels_L{L}/test/*.pt"))[:20]
    for r in (0.05, 0.1):
        ts, P_all = [], []
        for n, f in enumerate(files):
            S = torch.load(f)["score"].float().to(dev); P = S.shape[1]; B = int(r * 112 * P)
            step(S, B); torch.cuda.synchronize()                                     # warm-up
            t = time.perf_counter()
            for _ in range(10): step(S, B)
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t) / 10 * 1000); P_all.append(P)
        print(f"L={L:<6} budget {r:<5} mean prompt {sum(P_all)/len(P_all):6.0f} tok  allocator step {sum(ts)/len(ts):6.2f} ms "
              f"(max {max(ts):.2f})  [{len(ts)} prompts]")
