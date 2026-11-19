"""Read-only asset checks. Run with the prepared environment on CARC."""
import argparse
import importlib.metadata as md
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from kvp_ranker import checkpoint_metadata, read_composite
from profiling_io import atomic_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--composite", required=True)
    ap.add_argument("--checkpoint_root", required=True)
    ap.add_argument("--require_gpu", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    model = Path(args.model)
    cfg = json.loads((model / "config.json").read_text())
    if (cfg["num_hidden_layers"], cfg["num_key_value_heads"]) != (28, 4):
        raise ValueError("This experiment expects Qwen2.5-7B-Instruct (28 layers, 4 KV heads)")
    index = model / "model.safetensors.index.json"
    weights = (set(json.loads(index.read_text())["weight_map"].values()) if index.exists()
               else {"model.safetensors"})
    for name in [*sorted(weights), "vocab.json", "merges.txt", "tokenizer_config.json"]:
        with (model / name).open("rb") as f:
            if not f.read(1):
                raise ValueError(f"Empty model/tokenizer file: {name}")
    entries = read_composite(args.composite, args.checkpoint_root)
    # Import the actual external component before spending time loading Qwen.
    from kvcompression.presses.sampler_agent_press import SamplerAgentPress  # noqa: F401
    import torch
    if args.require_gpu:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("An allocated NVIDIA GPU with bf16 support is required")
    report = dict(model=str(model.resolve()), heads=len(entries),
                  kvp=checkpoint_metadata(entries), python=sys.version,
                  versions={p: md.version(p) for p in ("torch", "transformers", "torchtune", "torchao")},
                  gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    if args.out:
        atomic_json(args.out, report)
    print(f"Preflight passed: {len(entries)} readable KVP checkpoints; Qwen weights and tokenizer present.")


if __name__ == "__main__":
    main()
