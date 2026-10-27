"""Diagnostic 2: like diag_rewards.py but adds SnapKV and the oracle ceiling (ranking by true future
attention; the reward is normalized so the oracle scores exactly -1).

Usage: uv run python scripts/diag_rewards2.py <pairs> <n_val_traces> [composite.yaml]
  <pairs>: comma-separated layers (all 4 heads each), e.g. 0,8,16,27
           or layer:head pairs, e.g. 16:1,8:3
  e.g. uv run python scripts/diag_rewards2.py 0,8,16,27 12
       uv run python scripts/diag_rewards2.py 16:1,8:3 44 config_composites/ruler845_b128.yaml
Only lines starting with RESULT are results: grep RESULT <log>
"""
import os, sys, math
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchtune import config, training
from kvcompression.data.activations_dataset import QKVDataset
from kvcompression.rl.rewards.oracle import Oracle

DEV = torch.device("cuda")
DATA = os.environ["KVCOMPRESSION_DATA_ROOT"] + "/gqa_safetensors/simonjegou_ruler/Qwen2.5-7B-Instruct/temperature_0.00"
PAIRS = []
for p in sys.argv[1].split(","):
    if ":" in p:
        l, h = p.split(":"); PAIRS.append((int(l), int(h)))
    else:
        PAIRS += [(int(p), h) for h in range(4)]
N = int(sys.argv[2])
COMPOSITE = sys.argv[3] if len(sys.argv) > 3 else "config_composites/ruler845_v1.yaml"
comp = {(a.layer, a.head): a.press for a in OmegaConf.load(COMPOSITE).assignments}
oracle = Oracle()


def reward(rank, k, q, P, L):
    c = oracle.compute_cost(kv_rankings=rank[None].to(DEV), kv_rankings_length=torch.tensor([P], device=DEV),
        all_keys=k[None].float().to(DEV), all_queries=q[None].float().to(DEV), seq_lengths=torch.tensor([L], device=DEV))
    return -c.cost.squeeze().item()


def stream_rank(P):
    s = torch.arange(P, dtype=torch.float32); s[:4] = float("inf"); return s.argsort(descending=True)


def snapkv_rank(k, q, P, w=64):
    k, q = k[:P].float().to(DEV), q[:, :P].float().to(DEV)
    a = (q[:, P - w:] @ k.T) / math.sqrt(k.shape[-1])
    a = a.masked_fill(torch.arange(P, device=DEV)[None] > torch.arange(P - w, P, device=DEV)[:, None], float("-inf"))
    s = a.softmax(-1).mean((0, 1))
    s = F.avg_pool1d(s[None, None], 5, stride=1, padding=2)[0, 0]; s[P - w:] = float("inf")
    return s.argsort(descending=True).cpu()


def oracle_rank(k, q, P, L):
    k, q = k.float().to(DEV), q.float().to(DEV)
    a = (q[:, P:L] @ k.T) / math.sqrt(k.shape[-1])
    a = a.masked_fill(torch.arange(L, device=DEV)[None] > torch.arange(P, L, device=DEV)[:, None], float("-inf"))
    return a.softmax(-1).amax(0)[:, :P].sum(0).argsort(descending=True).cpu()


cols = ["agent+sw", "stream", "snapkv", "oracle"]
tot = {c: [] for c in cols}
print(f"composite: {COMPOSITE}")
print(f"RESULT {'layer':>5} {'head':>4} " + " ".join(f"{c:>9}" for c in cols))
with torch.no_grad():
    for layer, head in PAIRS:
        ds = QKVDataset(DATA, "bf16", layer, head, "val.txt")
        with training.set_default_dtype(torch.bfloat16), DEV:
            press = config.instantiate(OmegaConf.merge(comp[(layer, head)], {"n_sinks": 4, "n_running_window": 16})).to(DEV).eval()
        acc = {c: [] for c in cols}
        for i in range(min(N, len(ds))):
            s = ds[i]; k, v, q, P = s["keys"], s["values"], s["queries"], int(s["prompt_len"]); L = k.shape[0]
            r = press.sort(None, k[None, None, :P].to(DEV), v[None, None, :P].to(DEV), q[None, :, :P].to(DEV))[0, 0]
            acc["agent+sw"].append(reward(r, k, q, P, L))
            acc["stream"].append(reward(stream_rank(P), k, q, P, L))
            acc["snapkv"].append(reward(snapkv_rank(k, q, P), k, q, P, L))
            acc["oracle"].append(reward(oracle_rank(k, q, P, L), k, q, P, L))
        m = {c: sum(x) / len(x) for c, x in acc.items()}
        for c in cols: tot[c].append(m[c])
        print(f"RESULT {layer:>5} {head:>4} " + " ".join(f"{m[c]:>9.3f}" for c in cols), flush=True)
print("RESULT MEAN " + "  ".join(f"{c}={sum(x)/len(x):.3f}" for c, x in tot.items()))
