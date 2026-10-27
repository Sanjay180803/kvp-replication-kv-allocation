"""Diagnostic 1: Apple's training reward (-normalized future-attention AUC; -1 = perfect) on held-out
val traces, for our trained agents vs simple rankings (StreamingLLM, random), per (layer, head).

Usage: uv run python scripts/diag_rewards.py <layers comma-separated> <n_val_traces>
  e.g. uv run python scripts/diag_rewards.py 0,4,8,12,16,20,24,27 12
Set DISABLE_TORCH_COMPILER=1 to skip torch.compile in the oracle.
"""
import os, sys
import torch
from omegaconf import OmegaConf
from torchtune import config, training

from kvcompression.data.activations_dataset import QKVDataset
from kvcompression.rl.rewards.oracle import Oracle

DEV = torch.device("cuda")
DATA = os.environ["KVCOMPRESSION_DATA_ROOT"] + "/gqa_safetensors/simonjegou_ruler/Qwen2.5-7B-Instruct/temperature_0.00"
LAYERS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,4,8,12,16,20,24,27").split(",")]
N_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 12

comp = {(a.layer, a.head): a.press for a in OmegaConf.load("config_composites/ruler845_v1.yaml").assignments}
oracle = Oracle()


def reward(rank, keys, queries, P, L):
    c = oracle.compute_cost(
        kv_rankings=rank[None].to(DEV), kv_rankings_length=torch.tensor([P], device=DEV),
        all_keys=keys[None].float().to(DEV), all_queries=queries[None].float().to(DEV),
        seq_lengths=torch.tensor([L], device=DEV))
    return -c.cost.squeeze().item()


def streaming_rank(P, sinks=4):
    s = torch.arange(P, dtype=torch.float32); s[:sinks] = float("inf")
    return s.argsort(descending=True)


totals = {}
print(f"{'layer':>5} {'head':>4} {'agent':>8} {'agent+sw':>9} {'stream':>8} {'random':>8}")
with torch.no_grad():
    for layer in LAYERS:
        for head in range(4):
            ds = QKVDataset(DATA, "bf16", layer, head, "val.txt")
            presses = {}
            for name, extra in [("agent", {}), ("agent+sw", {"n_sinks": 4, "n_running_window": 16})]:
                with training.set_default_dtype(torch.bfloat16), DEV:
                    presses[name] = config.instantiate(OmegaConf.merge(comp[(layer, head)], extra)).to(DEV).eval()
            acc = {k: [] for k in ["agent", "agent+sw", "stream", "random"]}
            for i in range(min(N_SAMPLES, len(ds))):
                s = ds[i]
                k, v, q, P = s["keys"], s["values"], s["queries"], int(s["prompt_len"])
                L = k.shape[0]
                for name, press in presses.items():
                    r = press.sort(None, k[None, None, :P].to(DEV), v[None, None, :P].to(DEV), q[None, :, :P].to(DEV))[0, 0]
                    acc[name].append(reward(r, k, q, P, L))
                acc["stream"].append(reward(streaming_rank(P), k, q, P, L))
                acc["random"].append(reward(torch.randperm(P), k, q, P, L))
            m = {kk: sum(vv) / len(vv) for kk, vv in acc.items()}
            for kk, vv in m.items():
                totals.setdefault(kk, []).append(vv)
            print(f"{layer:>5} {head:>4} {m['agent']:>8.3f} {m['agent+sw']:>9.3f} {m['stream']:>8.3f} {m['random']:>8.3f}", flush=True)

print("\nMEAN  " + "  ".join(f"{k}={sum(v)/len(v):.3f}" for k, v in totals.items()))
print("(higher is better; -1 = perfect)")
