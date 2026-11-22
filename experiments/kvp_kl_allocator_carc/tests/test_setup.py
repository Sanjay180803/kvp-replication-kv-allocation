"""CPU verification: same weight formula, strict profile matching, and actual training CLI."""
import csv
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec = importlib.util.spec_from_file_location('kvp_alloc_prepare', ROOT/'prepare.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
torch.set_num_threads(1)


def split_record():
    tasks = [f'task_{i}' for i in range(13)]
    train = [500*t+j for t in range(13) for j in range(410,500)]
    val = [500*t+j for t in range(13) for j in range(405,410)]
    test = [500*t+j for t in range(13) for j in range(20)]
    dropped = train[:6]
    train = [i for i in train if i not in dropped]
    return dict(tasks=tasks, train=train, val=val, test=test,
                task_of={str(i):tasks[i//500] for i in train+val+test},
                test_answer_overlap_with_trainval={'dropped_trainval_ids':dropped})


def test_saved_split_exclusions_and_leakage(tmp_path):
    sp = split_record()
    path = tmp_path/'splits.json'
    path.write_text(json.dumps(sp))
    assert len(prep.check_splits(path)['train']) == 1164
    sp['train'].append(400)
    sp['task_of']['400']='task_0'
    path.write_text(json.dumps(sp))
    with pytest.raises(ValueError,match='Invalid train ID'):
        prep.check_splits(path)


def write_profile(path, sp, scorer='kvp', omit_last=False):
    fields=['prompt_id','task','scorer','keep_ratio','layer','kv_head','kl_mean',
            'prompt_len','n_evicted','n_pos']
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for t in range(13):
            for j in range(400,405):
                for lv in prep.LEVELS:
                    for h in range(112):
                        if omit_last and t==12 and j==404 and lv==.25 and h==111:
                            continue
                        w.writerow(dict(prompt_id=500*t+j,task=sp['tasks'][t],scorer=scorer,
                            keep_ratio=lv,layer=h//4,kv_head=h%4,kl_mean=h+1,
                            prompt_len=1100,n_evicted=1100-int(lv*1100),n_pos=2))


def test_kvp_profile_shares_complete_finite_and_matching(tmp_path):
    sp=split_record();p=tmp_path/'kvp.csv';write_profile(p,sp)
    share,counts=prep.profile_shares(p,sp)
    expected=torch.arange(1,113,dtype=torch.float32)
    expected/=expected.sum()
    torch.testing.assert_close(share,expected.expand(4,-1))
    assert set(counts.values())=={65}
    snap=tmp_path/'snap.csv'
    with snap.open('w') as f:
        f.write('prompt_id,scorer,keep_ratio\n')
        for lv in prep.LEVELS:
            f.write(f'400,snapkv,{lv}\n')
    # Each matching (prompt, ratio) group must include all 112 heads.
    with pytest.raises(ValueError,match='112 heads'):
        prep.profile_shares(p,sp,[snap])
    with snap.open('w') as f:
        f.write('prompt_id,scorer,keep_ratio,layer,kv_head\n')
        for lv in prep.LEVELS:
            for h in range(112):
                f.write(f'400,snapkv,{lv},{h//4},{h%4}\n')
    _,counts=prep.profile_shares(p,sp,[snap]);assert set(counts.values())=={1}
    write_profile(p,sp,omit_last=True)
    with pytest.raises(ValueError,match='65 prompts'):
        prep.profile_shares(p,sp)


def test_weights_equal_wenbo_original_formula(tmp_path):
    labels=tmp_path/'train';labels.mkdir()
    g=torch.Generator().manual_seed(7)
    train_ids=list(range(10))
    selected=train_ids[::max(1,len(train_ids)//3)][:3]
    for i in selected:
        torch.save({'score':torch.randn(112,1100,generator=g),
                    'fut':torch.rand(112,1100,generator=g)},labels/f'{i}.pt')
    share=torch.rand(4,112,generator=g);share/=share.sum(1,keepdim=True)
    alpha,ids=prep.derive_weights(share,tmp_path,train_ids,3)
    lost=torch.zeros(4,112)
    for i in selected:
        d=torch.load(labels/f'{i}.pt',weights_only=True)
        P=d['fut'].shape[1];f=d['fut'].float()[:,20:]
        cf=torch.cat([torch.zeros(112,1),f.cumsum(1)],1)
        tot=f.sum(1).clamp_min(1e-9)
        for k,lv in enumerate(prep.LEVELS):
            lost[k]+=(tot-cf[:,max(0,int(lv*P)-20)])/tot
    lost/=len(selected)
    ls=lost/lost.sum(1,keepdim=True)
    expected=share.mean(0)/ls.mean(0).clamp_min(1e-9)
    expected/=expected.mean()
    assert ids==selected
    torch.testing.assert_close(alpha,expected,rtol=0,atol=0)


def test_original_training_cli_fixed_to_input(tmp_path):
    g=torch.Generator().manual_seed(0)
    splits={'train':[0,1],'val':[2,3],'test':[],'loto':{},'task_of':{}}
    for split in ('train','val'):
        p=tmp_path/'labels'/split;p.mkdir(parents=True)
        for i in splits[split]:
            score=torch.randn(112,1100,generator=g).sort(-1,descending=True).values
            fut=torch.rand(112,1100,generator=g)
            torch.save(dict(score=score.half(),fut=fut),p/f'{i}.pt')
    (tmp_path/'splits.json').write_text(json.dumps(splits))
    torch.save(torch.linspace(.5,1.5,112),tmp_path/'weights.pt')
    common=['--splits',str(tmp_path/'splits.json'),'--labels',str(tmp_path/'labels'),
            '--work',str(tmp_path/'work'),'--device','cpu','--head_weights',str(tmp_path/'weights.pt'),
            '--steps','1','--eval_every','1','--batch','1','--K','2']
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for mode in ('fixed','input'):
        cmd=[sys.executable,str(ROOT/'vendor/wenbo/train_alloc.py'),'train','--mode',mode]+common
        if mode=='input':cmd+=['--init',str(tmp_path/'work/fixed_hw_s0/best.pt')]
        r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=90)
        assert r.returncode==0,r.stdout+r.stderr
    r=subprocess.run([sys.executable,str(ROOT/'finish.py'),'--run',str(tmp_path),'--seed','0'],
                     env=env,capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stdout+r.stderr
    assert set(json.loads((tmp_path/'training_summary.json').read_text()))=={'fixed','input'}


def test_downloaded_exact_split():
    sp=prep.check_splits(ROOT/'inputs/splits.json')
    assert sp['sha1']=='874e55c37a9a7aa7c87b2939e990c30f07b99355'
    assert [len(sp[s]) for s in ('train','val','test')]==[1164,65,260]


def test_label_core_preserves_wenbo_math():
    original=ast.parse((ROOT/'vendor/wenbo/label.py').read_text())
    adapted=ast.parse((ROOT/'label_carc.py').read_text())
    for name in ('load_model','load_presses','build_prompt','Collector'):
        a=next(n for n in original.body if getattr(n,'name',None)==name)
        b=next(n for n in adapted.body if getattr(n,'name',None)==name)
        assert ast.dump(a)==ast.dump(b),name


def test_label_atomic_validation_and_provenance(tmp_path):
    import label_io as li
    p=tmp_path/'labels';p.mkdir()
    li.prepare_manifest(p,{'version':1})
    li.prepare_manifest(p,{'version':1})
    with pytest.raises(ValueError,match='provenance differs'):
        li.prepare_manifest(p,{'version':2})
    P=100
    order=torch.tensor([*range(4),*range(P-16,P),*range(4,P-16)])
    data=dict(id=416,rank=order.repeat(112,1).short(),score=torch.ones(112,P).half(),
              fut=torch.ones(112,P),win=torch.ones(112,P).half())
    dest=p/'train';dest.mkdir()
    li.save_label(dest/'416.pt',data)
    assert li.validate_label(dest/'416.pt',416)==100
    assert li.validate_coverage(p,{'train':[416]})['complete']
    data['rank'][0,0]=data['rank'][0,1]
    with pytest.raises(ValueError,match='Invalid token ranking'):
        li.save_label(dest/'416.pt',data)
    # A failed rewrite leaves the existing valid label intact.
    li.validate_label(dest/'416.pt',416)


def test_label_checkpoint_remapping(tmp_path):
    import label_io as li
    import yaml
    model=tmp_path/'model';model.mkdir()
    (model/'config.json').write_text(json.dumps({'num_hidden_layers':28,'num_key_value_heads':4}))
    names=[f'model-0000{i}-of-00004.safetensors' for i in range(1,5)]
    for n in [*names,'vocab.json','merges.txt','tokenizer_config.json']:(model/n).write_text('{}')
    (model/'model.safetensors.index.json').write_text(json.dumps({'weight_map':dict(enumerate(names))}))
    mapping=ROOT/'inputs/ruler845_b128.yaml'
    config=yaml.safe_load(mapping.read_text())
    ckpts=tmp_path/'ckpts'
    for a in config['assignments']:
        parts=Path(a['press']['ckpt_path']).parts
        k=next(j for j,v in enumerate(parts) if v.startswith('layer_'))
        path=ckpts.joinpath(*parts[k:]);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'x')
    resolved,identity=li.resolve_assets(model,mapping,ckpts)
    assert len(identity['checkpoints'])==112
    assert len(resolved['assignments'])==112
    assert all(str(ckpts) in a['press']['ckpt_path'] for a in resolved['assignments'])


def test_a40_attention_patch_is_targeted_and_idempotent(tmp_path):
    from patch_a40_attention import apply_patch, patch_source
    original = ('def attention(q, k, v, block_mask):\n'
                '    return compiled_flex(\n'
                '        query=q,\n'
                '        key=k,\n'
                '        value=v,\n'
                '        block_mask=block_mask,\n'
                '    )\n')
    calls=[]
    ns={'compiled_flex':lambda **kw:calls.append(kw)}
    updated=patch_source(original)
    exec(updated,ns)
    ns['attention'](1,2,3,4)
    assert calls==[dict(query=1,key=2,value=3,block_mask=4,
                        kernel_options=dict(BLOCK_M=64,BLOCK_N=64,num_stages=1))]
    assert patch_source(updated)==updated
    path=tmp_path/'src/kvcompression/kv_cache/compressible_attention.py'
    path.parent.mkdir(parents=True);path.write_text(original)
    apply_patch(tmp_path);apply_patch(tmp_path)
    assert path.read_text()==updated
    assert path.with_name(path.name+'.before_kvp_alloc_a40').read_text()==original
    with pytest.raises(ValueError,match='Existing kernel_options differ'):
        patch_source(updated.replace("'BLOCK_M': 64", "'BLOCK_M': 128"))
