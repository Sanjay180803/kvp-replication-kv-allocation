"""Check complete atomic profiling outputs; no GPU required."""
import argparse
import json
from pathlib import Path

from profiling_io import expected_keys, load_prompt, publish_csv
from profile_heads import CSV_FIELDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    out = Path(args.dir)
    cfg = json.loads((out / "experiment.json").read_text())
    keys = expected_keys(cfg["scorers"], cfg["ratios"], cfg["heads"])
    coverage = publish_csv(out, cfg["prompt_ids"], keys, CSV_FIELDS)
    if not coverage["complete"] or not coverage["completed_prompts"]:
        raise SystemExit(f"Profile is incomplete or empty: {coverage}")
    print(json.dumps(coverage, indent=2))
    print("SUCCESS: complete result shards, exact kept counts, unique finite rows.")


if __name__ == "__main__":
    main()
