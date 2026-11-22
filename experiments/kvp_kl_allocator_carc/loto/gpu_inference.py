"""KVP test features via Wenbo's full-answer pass, then held-out RULER evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))

import torch
from datasets import load_dataset
from torchtune.models.qwen2_5 import qwen2_5_tokenizer
from kvcompression.hooks.compressor import KVCompressor,LayerHeadCompressionConfig
from kvcompression.hooks.registrar import register_attention_forward_hook
from kvcompression.kv_cache.compression_strategy_protocol import CompressionStrategy,PressCompressionStrategy,DummyCompressionStrategy
from kvcompression.utils.generation import generate_with_compression
import label_carc as lc
from label_io import resolve_assets
from protocol import checked_protocol,write_json,validate_counts


class FixedCountStrategy(CompressionStrategy):
    def __init__(self,inner,count):self.inner,self.count=inner,int(count)

    def get_mask(self,k_cache_active,v_cache_active,target_compressed_size,q_assoc_cache_active=None,**kw):
        S=k_cache_active.shape[2]
        if self.count>=S:return torch.ones(k_cache_active.shape[:-1],dtype=torch.bool,device=k_cache_active.device)
        return self.inner.get_mask(k_cache_active=k_cache_active,v_cache_active=v_cache_active,
            target_compressed_size=max(1,self.count),q_assoc_cache_active=q_assoc_cache_active,**kw)


def validate_feature(data,pid):
    if set(data)!= {'id','score','rank'} or data['id']!=pid:raise ValueError('Invalid prompt-only feature record')
    score=data['score'];rank=data['rank'].long()
    if score.ndim!=2 or score.shape[0]!=112 or rank.shape!=score.shape:raise ValueError('Feature shape mismatch')
    P=score.shape[1]
    if P<=20 or not torch.isfinite(score[:,20:]).all():raise ValueError('Invalid feature scores')
    if not torch.equal(rank.sort(1).values,torch.arange(P).expand(112,-1)):raise ValueError('Invalid feature rankings')
    protected=torch.tensor([*range(4),*range(P-16,P)]).sort().values.expand(112,-1)
    if not torch.equal(rank[:,:20].sort(1).values,protected):raise ValueError('Protected feature tokens mismatch')
    return P


def setup(args,ids):
    import yaml
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('bf16 CUDA GPU required')
    torch.manual_seed(0);torch.set_num_threads(1)
    lc.MODEL_DIR=Path(args.model).resolve()
    resolved,assets=resolve_assets(args.model,args.composite,args.checkpoint_root)
    ds=load_dataset('simonjegou/ruler',data_dir='4096',split='test',revision=args.dataset_revision)
    if len(ds)!=6500:raise ValueError('Unexpected RULER-4K size')
    train_manifest=json.loads((Path(args.labels)/'label_manifest.json').read_text())
    if assets!=train_manifest['assets'] or ds._fingerprint!=train_manifest['dataset_fingerprint']:
        raise ValueError('Model/checkpoint/dataset identity differs from TRAIN/VAL labels')
    for name in ('label_carc.py','ruler_scoring.py'):
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=train_manifest['source'][name]:
            raise ValueError(f'Label computation source changed since TRAIN/VAL extraction: {name}')
    if torch.__version__!=train_manifest['versions']['torch']:
        raise ValueError('PyTorch differs from the TRAIN/VAL labeling environment')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    provenance=dict(dataset_fingerprint=ds._fingerprint,assets=assets,ids=ids,
        splits_sha256=hashlib.sha256(Path(args.splits).read_bytes()).hexdigest(),
        dataset_revision=args.dataset_revision,torch=torch.__version__,seed=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=out/'inference_config.json'
    if path.exists() and json.loads(path.read_text())!=provenance:raise ValueError('Inference output provenance differs')
    if not path.exists():
        if list(out.glob('test/*.pt')) or list(out.glob('per_prompt/*.json')):raise ValueError('Unversioned inference output')
        write_json(path,provenance)
    composite=out/'resolved_composite.yaml';composite.write_text(yaml.safe_dump(resolved,sort_keys=False))
    tok=qwen2_5_tokenizer(path=str(lc.MODEL_DIR/'vocab.json'),merges_file=str(lc.MODEL_DIR/'merges.txt'))
    model=lc.load_model();presses=lc.load_presses(str(composite))
    return ds,tok,model,presses,provenance


def features(args):
    sp=checked_protocol(args.splits);ids=sp['test']
    if args.smoke:ids=sp['loto'][sp['tasks'][0]]['test'][:2]
    ds,tok,model,presses,provenance=setup(args,ids)
    out=Path(args.out);dest=out/'test';dest.mkdir(exist_ok=True);lengths={}
    for n,pid in enumerate(ids):
        target=dest/f'{pid}.pt'
        if target.exists():
            lengths[str(pid)]=validate_feature(torch.load(target,map_location='cpu',weights_only=True),pid);continue
        if ds[pid]['task']!=sp['task_of'][str(pid)]:raise ValueError('Task ID mismatch')
        t0=time.time()
        with torch.no_grad():
            prompt=lc.build_prompt(tok,ds[pid]);P=prompt.shape[1]
            # Match Wenbo's test-label forward shapes, including the greedy answer.
            # Only prompt KVP scores/ranks are saved for allocation. Future targets
            # and reference answers are never passed to the allocation policy.
            max_new=int(ds[pid].get('max_new_tokens') or 128)
            full_cfg=[LayerHeadCompressionConfig(None,None,DummyCompressionStrategy(),'full')]
            with KVCompressor(model=model,compression_strategies=full_cfg) as comp:
                with lc.DEVICE:model.setup_caches(batch_size=1,dtype=lc.DTYPE,decoder_max_seq_len=P+max_new)
                gen=generate_with_compression(model,prompt,comp,P,max_new,stop_tokens=tok.stop_tokens)
            n_fut=max(gen.shape[1]-P-1,0);seq=gen[:,:P+n_fut];T=seq.shape[1]
            col=lc.Collector(presses,P,n_fut)
            for layer in model.layers:layer.attn.cache_enabled=False
            with register_attention_forward_hook(model=model,hook=col.hook,attention_wrapper=lc.Collector.wrap):
                mask=torch.tril(torch.ones(T,T,dtype=torch.bool,device=lc.DEVICE))[None]
                model(seq,input_pos=torch.arange(T,device=lc.DEVICE)[None],mask=mask)
            data=dict(id=pid,rank=col.rank.cpu(),score=col.score.cpu())
        lengths[str(pid)]=validate_feature(data,pid)
        tmp=target.with_suffix('.pt.tmp');torch.save(data,tmp);tmp.replace(target)
        print(f'Feature {n+1}/{len(ids)} id={pid} P={P} {time.time()-t0:.2f}s',flush=True)
    write_json(out/'features_manifest.json',dict(complete=True,ids=ids,lengths=lengths,
        contains_future_attention=False,extraction='Wenbo full-answer label forward; save score/rank only',provenance=provenance))
    print('SUCCESS: test KVP features complete; no future targets saved.',flush=True)


def evaluate(args):
    sp=checked_protocol(args.splits)
    alloc=json.loads(Path(args.alloc).read_text());ids=sorted(map(int,alloc['per_prompt']))
    expected=sp['loto'][sp['tasks'][0]]['test'][:2] if args.smoke else sp['test']
    fm=json.loads((Path(args.features)/'features_manifest.json').read_text())
    if not fm['complete'] or set(ids)!=set(expected):raise ValueError('Wrong evaluation ID coverage')
    validate_counts(alloc,expected,fm['lengths'])
    ds,tok,model,presses,provenance=setup(args,ids)
    if provenance['assets']!=fm['provenance']['assets'] or provenance['dataset_fingerprint']!=fm['provenance']['dataset_fingerprint']:
        raise ValueError('Feature/evaluation assets differ')
    out=Path(args.out);dest=out/'per_prompt';dest.mkdir(exist_ok=True)
    ahash=hashlib.sha256(Path(args.alloc).read_bytes()).hexdigest();marker=out/'allocation_identity.json'
    if marker.exists() and json.loads(marker.read_text())!={'sha256':ahash}:raise ValueError('Allocation changed')
    write_json(marker,{'sha256':ahash})
    rows=[]
    with torch.no_grad():
        for n,pid in enumerate(ids):
            target=dest/f'{pid}.json'
            if target.exists():
                r=json.loads(target.read_text())
                if r['id']!=pid or r['method']!=alloc['name']:raise ValueError('Cached eval mismatch')
                rows.append(r);continue
            row=ds[pid]
            if row['task']!=sp['task_of'][str(pid)]:raise ValueError('Task ID mismatch')
            t0=time.time();prompt=lc.build_prompt(tok,row);P=prompt.shape[1]
            if P!=fm['lengths'][str(pid)]:raise ValueError('Feature/eval prompt lengths differ')
            counts=alloc['per_prompt'][str(pid)]
            cfgs=[LayerHeadCompressionConfig(L,h,
                FixedCountStrategy(PressCompressionStrategy(presses[(L,h)]),counts[4*L+h]),f'L{L}H{h}')
                for L in range(28) for h in range(4)]
            max_new=int(row.get('max_new_tokens') or 128)
            with KVCompressor(model=model,compression_strategies=cfgs) as comp:
                with lc.DEVICE:model.setup_caches(batch_size=1,dtype=lc.DTYPE,decoder_max_seq_len=P+max_new)
                gen=generate_with_compression(model,prompt,comp,P,max_new,stop_tokens=tok.stop_tokens)
            pred=tok.decode(gen[0,P:].tolist(),skip_special_tokens=True)
            score=lc.ruler_score(pred,row['answer'],row['task'])
            if not math.isfinite(score) or not 0<=score<=100:raise ValueError('Invalid RULER score')
            r=dict(id=pid,task=row['task'],method=alloc['name'],ratio=alloc['ratio'],prompt_len=P,
                kept_frac=sum(counts)/(112*P),score=score,pred=pred,sec=round(time.time()-t0,2))
            write_json(target,r);rows.append(r)
            print(f'Eval {n+1}/{len(ids)} id={pid} score={score:.1f} {r["sec"]}s',flush=True)
    with (out/'results.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    write_json(out/'coverage.json',dict(complete=True,rows=len(rows),ids=ids,
        mean_score=sum(r['score'] for r in rows)/len(rows)))
    print('SUCCESS: held-out RULER evaluation complete.',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['features','eval'])
    for n in ('splits','labels','model','composite','checkpoint-root','out'):ap.add_argument('--'+n,required=True)
    ap.add_argument('--alloc');ap.add_argument('--features');ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--dataset-revision',default=None);a=ap.parse_args()
    if a.stage=='eval' and (not a.alloc or not a.features):ap.error('eval needs --alloc and --features')
    {'features':features,'eval':evaluate}[a.stage](a)


if __name__=='__main__':main()
