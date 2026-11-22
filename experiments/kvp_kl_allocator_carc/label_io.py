"""Read-only asset resolution and validated atomic allocator-label persistence."""
import hashlib
import json
import os
from pathlib import Path

import torch
import yaml

from prepare import check_splits, sha256


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n')
    tmp.replace(path)


def resolve_assets(model, composite, checkpoint_root):
    model=Path(model).resolve()
    cfg=json.loads((model/'config.json').read_text())
    if (cfg['num_hidden_layers'],cfg['num_key_value_heads']) != (28,4):
        raise ValueError('Expected Qwen2.5-7B-Instruct, 28 layers and 4 KV heads')
    index=json.loads((model/'model.safetensors.index.json').read_text())
    expected={f'model-0000{i}-of-00004.safetensors' for i in range(1,5)}
    if set(index['weight_map'].values()) != expected:
        raise ValueError('This loader expects the existing four-shard Qwen model')
    for name in [*expected,'vocab.json','merges.txt','tokenizer_config.json']:
        with (model/name).open('rb') as f:
            if not f.read(1):raise ValueError(f'Empty asset: {name}')
    weights=[dict(name=n,bytes=(model/n).stat().st_size,mtime_ns=(model/n).stat().st_mtime_ns)
             for n in sorted(expected)]
    data=yaml.safe_load(Path(composite).read_text())
    assignments=[]; identities=[]; seen=set()
    for entry in data['assignments']:
        pair=(int(entry['layer']),int(entry['head']))
        if pair in seen or not (0<=pair[0]<28 and 0<=pair[1]<4):
            raise ValueError(f'Duplicate/invalid KVP head: {pair}')
        seen.add(pair)
        press=dict(entry['press'])
        if press['_component_'] != 'kvcompression.presses.sampler_agent_press.SamplerAgentPress':
            raise ValueError('Unexpected KVP component')
        original=Path(press['ckpt_path']).parts
        start=next((j for j,part in enumerate(original) if part.startswith('layer_')),None)
        if start is None:raise ValueError('Cannot remap KVP checkpoint')
        p=Path(checkpoint_root).joinpath(*original[start:]).resolve()
        with p.open('rb') as f:
            if not f.read(1):raise ValueError(f'Empty checkpoint: {p}')
        press.update(ckpt_path=str(p),n_sinks=4,n_running_window=16)
        assignments.append(dict(layer=pair[0],head=pair[1],press=press))
        st=p.stat()
        identities.append(dict(layer=pair[0],head=pair[1],path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
    if len(seen)!=112:raise ValueError('Need all 112 KVP checkpoints')
    return dict(assignments=assignments),dict(model=str(model),weights=weights,
        tokenizer_configs={p.name:sha256(p) for p in sorted(model.glob('*.json'))},
        checkpoints=identities,identity_kind='config hashes plus weight path/size/mtime, not weight content hashes')


def prepare_manifest(out, manifest):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p=out/'label_manifest.json'
    if p.exists():
        if json.loads(p.read_text())!=manifest:
            raise ValueError('Label provenance differs; use a new LABELS_DIR')
    elif any(out.glob('*/*.pt')) or any(out.glob('*/meta*.jsonl')):
        raise ValueError('Existing labels lack a matching manifest; use a new LABELS_DIR')
    else:atomic_json(p,manifest)


def validate_label(path, pid):
    d=torch.load(path,map_location='cpu',weights_only=True)
    if int(d['id'])!=pid:raise ValueError(f'Wrong label ID: {pid}')
    shape=d['score'].shape
    if len(shape)!=2 or shape[0]!=112 or shape[1]<=20:
        raise ValueError(f'Invalid label shape: {pid}')
    P=shape[1]
    for k in ('rank','fut','win'):
        if d[k].shape!=shape:raise ValueError(f'Wrong {k} shape: {pid}')
    for k in ('fut','win'):
        if not torch.isfinite(d[k]).all() or (d[k]<0).any():
            raise ValueError(f'Invalid {k}: {pid}')
    if not torch.isfinite(d['score'][:,20:]).all():
        raise ValueError(f'Invalid free-token score: {pid}')
    rank=d['rank'].long()
    if not torch.equal(rank.sort(1).values,torch.arange(P).expand(112,-1)):
        raise ValueError(f'Invalid token ranking: {pid}')
    protected=torch.tensor([*range(4),*range(P-16,P)]).sort().values.expand(112,-1)
    if not torch.equal(rank[:,:20].sort(1).values,protected):
        raise ValueError(f'Protected-token mismatch: {pid}')
    return P


def save_label(path, data):
    path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(data,tmp)
    validate_label(tmp,int(data['id']))
    tmp.replace(path)


def validate_coverage(out, selected):
    out=Path(out);counts={};missing={}
    for split,ids in selected.items():
        absent=[]
        for pid in ids:
            path=out/split/f'{pid}.pt'
            if path.exists():validate_label(path,pid)
            else:absent.append(pid)
        counts[split]=len(ids)-len(absent);missing[split]=absent
    report=dict(complete=not any(missing.values()),completed=counts,missing=missing)
    atomic_json(out/'label_coverage.json',report)
    if not report['complete']:raise ValueError(f'Missing labels: {missing}')
    return report
