"""Frozen KVP ranking shared by KL profiling and allocator episodes.

Uses Apple's existing SamplerAgentPress; no agent architecture is copied here.
Qwen2's HF and TorchTune implementations both use split-half RoPE and the
Qwen weight converter does not permute Q/K projections. Do not apply a Llama
interleaving permutation to these captured Qwen keys.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import torch
import yaml

KVP_SHARED = {"n_sinks": 4, "n_running_window": 16}


def read_composite(path, checkpoint_root=None, n_layers=28, n_kv=4):
    """Validate complete, unique head coverage and resolve transferred weights.

    checkpoint_root is the copied ruler845_b128 directory, preserving the
    layer_*/kv_head_*/<run>/best_ckpt.pth hierarchy from the original composite.
    """
    path = Path(path).expanduser().resolve()
    data = yaml.safe_load(path.read_text())
    entries = data.get("assignments", []) if isinstance(data, dict) else []
    expected = {(l, h) for l in range(n_layers) for h in range(n_kv)}
    out = {}
    for entry in entries:
        pair = (int(entry["layer"]), int(entry["head"]))
        if pair not in expected or pair in out:
            raise ValueError(f"Invalid or duplicate KVP head: {pair}")
        cfg = dict(entry["press"])
        if cfg.get("_component_") != "kvcompression.presses.sampler_agent_press.SamplerAgentPress":
            raise ValueError(f"Unexpected KVP component for head {pair}")
        ckpt = Path(cfg["ckpt_path"]).expanduser()
        if checkpoint_root:
            start = next((i for i, p in enumerate(ckpt.parts)
                          if re.fullmatch(r"layer_\d+", p)), None)
            if start is None:
                raise ValueError(f"Cannot remap checkpoint path: {ckpt}")
            ckpt = Path(checkpoint_root).expanduser().joinpath(*ckpt.parts[start:])
        elif not ckpt.is_absolute():
            ckpt = path.parent / ckpt
        ckpt = ckpt.resolve()
        with ckpt.open("rb") as f:
            if not f.read(1):
                raise ValueError(f"Empty checkpoint: {ckpt}")
        cfg.update(KVP_SHARED, ckpt_path=str(ckpt))
        out[pair] = cfg
    if set(out) != expected:
        raise ValueError(f"KVP composite missing heads: {sorted(expected - set(out))}")
    return out


def checkpoint_metadata(entries):
    """Fingerprint config and checkpoint identities without reading all weights."""
    files = []
    for (layer, head), cfg in sorted(entries.items()):
        p = Path(cfg["ckpt_path"])
        st = p.stat()
        files.append(dict(layer=layer, head=head, config=cfg,
                          bytes=st.st_size, mtime_ns=st.st_mtime_ns))
    encoded = json.dumps(files, sort_keys=True).encode()
    return dict(fingerprint=hashlib.sha256(encoded).hexdigest(), checkpoints=files,
                fingerprint_kind="config+path+size+mtime (not a weight-content hash)")


class KVPRanker:
    def __init__(self, presses, n_layers, n_kv, metadata=None):
        expected = {(l, h) for l in range(n_layers) for h in range(n_kv)}
        if set(presses) != expected:
            raise ValueError("KVP presses must cover every layer/KV-head exactly once")
        self.presses = presses
        self.n_layers, self.n_kv = n_layers, n_kv
        self.metadata = metadata or {"fingerprint": "injected-test-ranker"}
        for press in presses.values():
            press.eval().requires_grad_(False)

    @classmethod
    def load(cls, composite, device, dtype, n_layers, n_kv, checkpoint_root=None):
        # Lazy imports keep ordinary SnapKV runs independent of Apple's package.
        from omegaconf import OmegaConf
        from torchtune import config, training

        entries = read_composite(composite, checkpoint_root, n_layers, n_kv)
        device = torch.device(device)
        presses = {}
        for pair, cfg in sorted(entries.items()):
            with training.set_default_dtype(dtype), device:
                press = config.instantiate(OmegaConf.create(cfg))
            # Match the existing evaluation loader's dtype behavior exactly.
            presses[pair] = press.to(device)
        print(f"Loaded {len(presses)} frozen KVP agents", flush=True)
        return cls(presses, n_layers, n_kv, checkpoint_metadata(entries))

    @torch.no_grad()
    def rank_layer(self, layer, keys, values, queries):
        """Prompt-only tensors: K/V [1,nkv,P,D], Q [1,nq,P,D]."""
        if keys.shape != values.shape or keys.ndim != 4 or keys.shape[0] != 1:
            raise ValueError("Expected matching K/V tensors [1,nkv,P,D]")
        _, nkv, P, _ = keys.shape
        if nkv != self.n_kv or queries.shape[0] != 1 or queries.shape[2] != P:
            raise ValueError("KVP query/key head or prompt dimensions disagree")
        nq = queries.shape[1]
        if nq % nkv:
            raise ValueError("Query heads must be divisible by KV heads")
        group = nq // nkv
        orders = []
        for h in range(nkv):
            order = self.presses[(layer, h)].sort(
                None, keys[:, h:h + 1], values[:, h:h + 1],
                queries[:, h * group:(h + 1) * group])[0, 0].long()
            if order.shape != (P,) or not torch.equal(order.sort().values, torch.arange(P, device=order.device)):
                raise ValueError(f"KVP returned an invalid token permutation for {(layer, h)}")
            orders.append(order)
        return torch.stack(orders)


def add_scorer_arguments(parser):
    parser.add_argument("--scorer", choices=["snapkv", "kvp"], default="snapkv")
    parser.add_argument("--composite", help="KVP agent mapping YAML; required with --scorer kvp")
    parser.add_argument("--checkpoint_root", help="Transferred ruler845_b128 directory on this cluster")


def ranker_for_args(args, model):
    if getattr(args, "scorer", "snapkv") != "kvp":
        return None
    if not args.composite:
        raise ValueError("--scorer kvp requires --composite")
    return KVPRanker.load(args.composite, model.device, next(model.parameters()).dtype,
                          model.config.num_hidden_layers, model.config.num_key_value_heads,
                          getattr(args, "checkpoint_root", None))


def episode_kwargs(args):
    return dict(scorer=getattr(args, "scorer", "snapkv"),
                kvp_ranker=getattr(args, "_kvp_ranker", None))
