"""Atomic profiling outputs and strict experiment/resume checks (standard library)."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def source_fingerprint(paths):
    return {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def local_model_identity(model):
    """Record tokenizer/config content and weight identities, not a 15GB rehash."""
    root = Path(model)
    if not root.is_dir():
        return None
    return dict(configs=source_fingerprint(sorted(root.glob("*.json"))),
                weights=[dict(name=p.name, bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                         for p in sorted(root.glob("*.safetensors"))])


def prepare_run(out_dir, experiment):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "experiment.json"
    if path.exists():
        if json.loads(path.read_text()) != experiment:
            raise ValueError(f"Experiment differs from {path}; use a new output directory")
    elif (out / "head_kl.csv").exists() or list((out / "prompts").glob("*.json")):
        raise ValueError(f"Unversioned existing results in {out}; use a fresh output directory")
    else:
        atomic_json(path, experiment)
    (out / "prompts").mkdir(exist_ok=True)


def row_key(r):
    return (r["scorer"], float(r["keep_ratio"]), int(r["layer"]), int(r["kv_head"]))


def expected_keys(scorers, ratios, heads):
    return {(s, float(r), int(l), int(h)) for s in scorers for r in ratios for l, h in heads}


def validate_rows(rows, prompt_id, expected):
    keys = [row_key(r) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"Incomplete or duplicate rows for prompt {prompt_id}")
    for r in rows:
        if int(r["prompt_id"]) != int(prompt_id):
            raise ValueError("Wrong prompt ID in result shard")
        if any(not math.isfinite(float(r[k])) for k in ("kl_mean", "kl_max", "top1_agree")):
            raise ValueError(f"Nonfinite measurement for prompt {prompt_id}")
        P = int(r["prompt_len"])
        kept = min(P, max(1, int(float(r["keep_ratio"]) * P)))
        if int(r["n_evicted"]) != P - kept or int(r["n_pos"]) < 1:
            raise ValueError(f"Invalid kept count or continuation length for prompt {prompt_id}")


def load_prompt(out_dir, prompt_id, expected):
    path = Path(out_dir) / "prompts" / f"{prompt_id}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record["status"] == "complete":
        validate_rows(record["rows"], prompt_id, expected)
    elif record["status"] != "skipped":
        raise ValueError(f"Invalid result status in {path}")
    return record


def publish_csv(out_dir, prompt_ids, expected, fields):
    out = Path(out_dir)
    rows, skipped, pending = [], [], []
    for pid in prompt_ids:
        record = load_prompt(out, pid, expected)
        if record is None:
            pending.append(pid)
        elif record["status"] == "skipped":
            skipped.append({"prompt_id": pid, "reason": record["reason"]})
        else:
            rows.extend(record["rows"])
    tmp = out / f"head_kl.csv.{os.getpid()}.tmp"
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out / "head_kl.csv")
    summary = dict(rows=len(rows), completed_prompts=len(prompt_ids) - len(skipped) - len(pending),
                   skipped=skipped, pending=pending, complete=not pending)
    atomic_json(out / "coverage.json", summary)
    return summary
