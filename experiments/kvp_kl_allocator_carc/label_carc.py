"""Reward data for the allocator: one model run per prompt, no traces kept.

For each RULER prompt (same prompt format as the team's eval_ruler*.py):
  1. Full-cache greedy answer with Apple's generate_with_compression (no compressor), the exact path the
     RULER evaluation uses. max_new_tokens = the row's value (default 128).
  2. One no-cache forward pass over prompt + answer with hooks on every attention layer (same hook point as
     Apple's trace extractor), and for each of the 112 KV heads, on the GPU:
       rank  = the frozen KVP agent's ranking of the prompt tokens, computed exactly as at evaluation time
               (SamplerAgentPress.sort with n_sinks=4, n_running_window=16 on the prompt's K/V/Q). Keeping
               b tokens in this head at evaluation time keeps rank[:b].
       score = the agent's raw score of each prompt token (feature input for the allocator)
       fut   = future attention on each prompt token: sum over the answer queries that run AFTER compression
               (generated tokens g_1..g_{m-1}; g_1 itself is produced by the uncompressed prefill), of the max
               over the 7 query heads sharing this KV head, of the softmax attention weight (KVP's definition,
               paper eq. 3.4). Labels only: never used as allocator input.
       win   = attention from the last 64 prompt tokens (question window) to each prompt token, same pooling.
               Only for the question-aware ablation; not used by the main allocator.
     All four are stored in rank order: arrays [112, prompt_len] (head = 4*layer + kv_head).
Output per prompt: <out>/<id>.pt (fp16/int16 tensors), and one JSON line in <out>/meta.jsonl.

Usage (repo root): uv run python $ALLOC/label.py --splits $DATA/splits.json --split train --shard 0 --nshards 4
                   --composite config_composites/ruler845_b128.yaml --out $DATA/labels
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from torchtune import config, training
from torchtune.data import Message
from torchtune.models.qwen2_5 import qwen2_5_tokenizer

from kvcompression.hooks.compressor import KVCompressor, LayerHeadCompressionConfig
from kvcompression.hooks.registrar import register_attention_forward_hook
from kvcompression.kv_cache.compression_strategy_protocol import DummyCompressionStrategy
from kvcompression.utils.generation import generate_with_compression

from ruler_scoring import score as ruler_score

DEVICE, DTYPE = torch.device("cuda"), torch.bfloat16
MODEL_DIR = None  # Set explicitly by --model before model/tokenizer loading.
KVP_SHARED = {"n_sinks": 4, "n_running_window": 16}  # same as the team's eval scripts
N_LAYERS, N_KV, N_Q, WIN = 28, 4, 28, 64
GROUP = N_Q // N_KV


def load_model():
    ckpt = config.instantiate(OmegaConf.create({
        "_component_": "torchtune.training.FullModelHFCheckpointer",
        "checkpoint_dir": str(MODEL_DIR),
        "checkpoint_files": [f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)],
        "output_dir": "/tmp", "model_type": "QWEN2"})).load_checkpoint()
    with training.set_default_dtype(DTYPE), DEVICE:
        model = config.instantiate(OmegaConf.create({"_component_": "torchtune.models.qwen2_5.qwen2_5_7b_instruct"}))
    model.load_state_dict(ckpt[training.MODEL_KEY], strict=True)
    return model.eval()


def load_presses(composite):
    data = OmegaConf.load(composite)
    presses = {}
    for a in data.assignments:
        with training.set_default_dtype(DTYPE), DEVICE:
            p = config.instantiate(OmegaConf.merge(a.press, KVP_SHARED))
        p = p.to(DEVICE).eval()
        p._last_scores = None

        def grab(mod, inp, out, press=p):
            press._last_scores = out["scores"].detach()
        p.agent.register_forward_hook(grab)
        presses[(int(a.layer), int(a.head))] = p
    assert len(presses) == N_LAYERS * N_KV
    return presses


def build_prompt(tok, row):
    msgs = [Message(role="user", content=row["context"] + row["question"]),
            Message(role="assistant", content=row["answer_prefix"], eot=False)]
    ids, _ = tok.tokenize_messages(msgs, add_eos=False)
    return torch.tensor([ids], device=DEVICE)


class Collector:
    """Attention hook: per layer, compute rank / score / fut / win for its 4 KV heads, store on GPU."""

    def __init__(self, presses, plen, n_fut):
        self.presses, self.plen, self.n_fut = presses, plen, n_fut
        P = plen
        self.rank = torch.empty(N_LAYERS * N_KV, P, dtype=torch.int16, device=DEVICE)
        self.score = torch.empty(N_LAYERS * N_KV, P, dtype=torch.float16, device=DEVICE)
        self.fut = torch.empty(N_LAYERS * N_KV, P, dtype=torch.float32, device=DEVICE)
        self.win = torch.empty(N_LAYERS * N_KV, P, dtype=torch.float32, device=DEVICE)

    @staticmethod
    def wrap(q, k, v, mask, dropout_p, is_causal, original_attention_call, attn_module):
        attn_module._kv_q, attn_module._kv_k, attn_module._kv_v = q, k, v
        return original_attention_call(q, k, v, mask=mask, dropout_p=dropout_p, is_causal=is_causal)

    @torch.no_grad()
    def hook(self, module, args, kwargs, outputs, layer_idx):
        q, k, v = module._kv_q[0], module._kv_k[0], module._kv_v[0]  # [H, T, D]
        del module._kv_q, module._kv_k, module._kv_v
        P, D = self.plen, q.shape[-1]
        T = q.shape[1]
        for h in range(N_KV):
            kh = k[h * GROUP] if k.shape[0] == N_Q else k[h]          # [T, D]
            vh = v[h * GROUP] if v.shape[0] == N_Q else v[h]
            qg = q[h * GROUP:(h + 1) * GROUP]                          # [7, T, D]
            idx = layer_idx * N_KV + h
            press = self.presses[(layer_idx, h)]
            # 1) KVP ranking + raw scores, exactly as the compressor calls the press after prefill
            rank = press.sort(hidden_states=None, keys=kh[None, None, :P], values=vh[None, None, :P],
                              queries=qg[None, :, :P], attention_weights=None)[0, 0]
            sc = press._last_scores.reshape(-1)[:P].float()
            self.rank[idx] = rank.to(torch.int16)
            self.score[idx] = sc[rank].to(torch.float16)
            # 2) future attention from the answer queries that run after compression (positions P .. P+n_fut-1)
            kf = kh[:T].float()
            if self.n_fut > 0:
                qa = qg[:, P:P + self.n_fut].float()                   # [7, f, D]
                lo = qa @ kf.T / math.sqrt(D)                          # [7, f, T]
                qpos = torch.arange(P, P + self.n_fut, device=DEVICE)
                lo = lo.masked_fill(torch.arange(T, device=DEVICE)[None, :] > qpos[:, None], float("-inf"))
                fut = lo.softmax(-1)[..., :P].amax(0).sum(0)           # [P]
            else:
                fut = torch.zeros(P, device=DEVICE)
            self.fut[idx] = fut[rank.long()]
            # 3) question-window attention (ablation feature only)
            w = min(WIN, P)
            qw = qg[:, P - w:P].float()
            lw = qw @ kf[:P].T / math.sqrt(D)
            qpos = torch.arange(P - w, P, device=DEVICE)
            lw = lw.masked_fill(torch.arange(P, device=DEVICE)[None, :] > qpos[:, None], float("-inf"))
            self.win[idx] = lw.softmax(-1).amax(0).sum(0)[rank.long()]


def main():
    # Collector, generation and model loading above are adapted from Wenbo's
    # label.py. Changes below are path resolution, provenance and atomic resume.
    import fcntl
    import importlib.metadata as md
    import yaml
    from label_io import (atomic_json, check_splits, prepare_manifest, resolve_assets,
                          save_label, sha256, validate_coverage, validate_label)
    global MODEL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits',required=True)
    ap.add_argument('--model',required=True)
    ap.add_argument('--composite',required=True)
    ap.add_argument('--checkpoint-root',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--mode',choices=['smoke','full'],default='smoke')
    ap.add_argument('--dataset-revision',default=None)
    args=ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Allocated NVIDIA GPU with bf16 support required')
    MODEL_DIR=Path(args.model).resolve()
    splits=check_splits(args.splits)
    selected={s:splits[s][:2] if args.mode=='smoke' else splits[s] for s in ('train','val')}
    resolved,assets=resolve_assets(MODEL_DIR,args.composite,args.checkpoint_root)
    ds=load_dataset('simonjegou/ruler',data_dir='4096',split='test',revision=args.dataset_revision)
    if len(ds)!=6500:raise ValueError('Unexpected RULER dataset size')
    here=Path(__file__).resolve().parent
    from kvcompression.kv_cache import compressible_attention
    provenance=dict(dataset='simonjegou/ruler:4096:test',dataset_revision=args.dataset_revision,
        dataset_fingerprint=ds._fingerprint,splits_sha256=sha256(args.splits),
        split_sha1=splits.get('sha1'),mode=args.mode,selected_ids=selected,seed=0,
        assets=assets,composite_sha256=sha256(args.composite),
        apple_attention_sha256=sha256(compressible_attention.__file__),
        source={n:sha256(here/n) for n in ('label_carc.py','label_io.py','ruler_scoring.py','prepare.py')},
        versions={p:md.version(p) for p in ('torch','torchtune','torchao','datasets')},
        dtype='bfloat16',continuation='full row max_new_tokens; future positions n_generated-1',
        kvp_shared=KVP_SHARED)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with (out/'.labels.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare_manifest(out,provenance)
        composite=out/'resolved_composite.yaml'
        composite.write_text(yaml.safe_dump(resolved,sort_keys=False))
        atomic_json(out/'splits.original.json',splits)
        print('Selected labels:',{s:len(v) for s,v in selected.items()},flush=True)
        print('GPU:',torch.cuda.get_device_name(),'| validated 112 checkpoints',flush=True)
        tok=qwen2_5_tokenizer(path=str(MODEL_DIR/'vocab.json'),merges_file=str(MODEL_DIR/'merges.txt'))
        model=load_model()
        presses=load_presses(str(composite))
        full_cfg=[LayerHeadCompressionConfig(None,None,DummyCompressionStrategy(),'full')]
        print('Loaded Qwen and 112 frozen KVP agents',flush=True)
        for split,ids in selected.items():
            dest=out/split;dest.mkdir(exist_ok=True)
            with (dest/'meta_shard0.jsonl').open('a') as fmeta:
                for n,i in enumerate(ids):
                    target=dest/f'{i}.pt'
                    if target.exists():
                        validate_label(target,i)
                        print(f'Resume {split} {i}: valid label exists',flush=True)
                        continue
                    t0=time.time();row=ds[i]
                    if row['task']!=splits['task_of'][str(i)]:raise ValueError(f'Task mismatch: {i}')
                    with torch.no_grad():
                        prompt=build_prompt(tok,row);P=prompt.shape[1]
                        max_new=int(row.get('max_new_tokens') or 128)
                        with KVCompressor(model=model,compression_strategies=full_cfg) as comp:
                            with DEVICE:
                                model.setup_caches(batch_size=1,dtype=DTYPE,decoder_max_seq_len=P+max_new)
                            gen=generate_with_compression(model,prompt,comp,P,max_new,stop_tokens=tok.stop_tokens)
                        m=gen.shape[1]-P;pred=tok.decode(gen[0,P:].tolist(),skip_special_tokens=True)
                        n_fut=max(m-1,0);seq=gen[:,:P+n_fut];T=seq.shape[1]
                        col=Collector(presses,P,n_fut)
                        for layer in model.layers:layer.attn.cache_enabled=False
                        with register_attention_forward_hook(model=model,hook=col.hook,attention_wrapper=Collector.wrap):
                            mask=torch.tril(torch.ones(T,T,dtype=torch.bool,device=DEVICE))[None]
                            model(seq,input_pos=torch.arange(T,device=DEVICE)[None],mask=mask)
                        save_label(target,dict(id=i,rank=col.rank.cpu(),score=col.score.cpu(),
                            fut=col.fut.cpu(),win=col.win.to(torch.float16).cpu()))
                    rec=dict(id=i,task=row['task'],prompt_len=P,n_gen=m,n_fut=n_fut,pred=pred,
                        answer=list(row['answer']) if not isinstance(row['answer'],str) else [row['answer']],
                        full_score=ruler_score(pred,row['answer'],row['task']),
                        fut_total_mean=float(col.fut.sum(1).mean()),sec=round(time.time()-t0,2))
                    fmeta.write(json.dumps(rec)+'\n');fmeta.flush()
                    print(f'[{split} {n+1}/{len(ids)}] id={i} P={P} gen={m} full={rec["full_score"]} {rec["sec"]}s',flush=True)
        report=validate_coverage(out,selected)
        print(json.dumps(report,indent=2),flush=True)
        print('SUCCESS: allocator labels generated and validated.',flush=True)


if __name__=='__main__':
    main()
