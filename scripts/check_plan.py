import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, "scripts")
import kv_plan as kp
recs = kp.load(Path("rl_runs/plan"), "train")[:3]
for r in recs:
    for t in (0.05, 0.1):
        a, b = kp.plan_slow(r["costs"], t), kp.plan(r["costs"], t)
        assert np.allclose(a, b, atol=1e-9), f"MISMATCH prompt {r['pid']} target {t}"
        print(f"prompt {r['pid']} target {t}: fast plan == slow plan")
print("planner check OK")
