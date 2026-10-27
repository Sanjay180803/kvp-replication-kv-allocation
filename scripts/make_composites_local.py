"""Local replacement for Apple's S3-only agent_registry.py / assignment_generator.py.

Scans agents/grouped/<sweep>/layer_*/kv_head_*/<run>/ and writes config_composites/<sweep>.yaml
pointing each (layer, head) at its checkpoint via a direct ckpt_path (no S3 needed).

Usage (from the ml-learning-to-evict repo root):
  uv run python scripts/make_composites_local.py ruler845_v1
"""
import sys
from pathlib import Path
import yaml

sweep = sys.argv[1] if len(sys.argv) > 1 else "ruler845_v1"
root = Path("agents/grouped") / sweep
out, problems = [], []

for layer_dir in sorted(root.glob("layer_*")):
    for head_dir in sorted(layer_dir.glob("kv_head_*")):
        runs = sorted(d for d in head_dir.iterdir() if d.is_dir() and d.name != "figures")
        if len(runs) != 1:
            problems.append(f"{head_dir}: {len(runs)} run folders")
            if not runs:
                continue
        run = runs[-1]
        ckpt = run / "best_ckpt.pth"
        if not ckpt.exists():
            resumes = sorted((run / "checkpoints").glob("*.pth"), key=lambda p: int(p.stem))
            if not resumes:
                problems.append(f"{run}: no checkpoint")
                continue
            ckpt = resumes[-1]
        out.append({
            "layer": int(layer_dir.name.split("_")[-1]),
            "head": int(head_dir.name.split("_")[-1]),
            "press": {
                "_component_": "kvcompression.presses.sampler_agent_press.SamplerAgentPress",
                "ckpt_path": str(ckpt.resolve()),
                "query_selection_mode": "no_agg",
            },
        })

Path("config_composites").mkdir(exist_ok=True)
f = Path("config_composites") / f"{sweep}.yaml"
yaml.dump({"assignments": out}, open(f, "w"), sort_keys=False)
print(f"wrote {f} with {len(out)} agents (full model = 112)")
print("best_ckpt used:", sum("best_ckpt" in a["press"]["ckpt_path"] for a in out))
for p in problems:
    print("WARNING:", p)
