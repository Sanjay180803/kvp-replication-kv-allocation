"""CPU tests for task exclusion, held-out inference contracts and dependency wiring."""
import ast
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'loto'));sys.path.insert(0,str(ROOT))
import protocol as lp


def profile(path,sp,perturb=None):
    fields=['prompt_id','task','scorer','keep_ratio','layer','kv_head','kl_mean','prompt_len','n_evicted','n_pos']
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for t,task in enumerate(sp['tasks']):
            for j in range(400,405):
                for lv in (.02,.05,.10,.25):
                    for h in range(112):
                        v=(h+1)*(t+1)+(h%7)*t
                        if task==perturb:v=(113-h)**2
                        w.writerow(dict(prompt_id=500*t+j,task=task,scorer='kvp',keep_ratio=lv,
                            layer=h//4,kv_head=h%4,kl_mean=v,prompt_len=1100,n_evicted=1100-int(lv*1100),n_pos=2))


def test_exact_wenbo_folds_and_profile_task_exclusion(tmp_path):
    sp=lp.checked_protocol(ROOT/'inputs/splits.json')
    assert len(sp['loto'])==13
    a=tmp_path/'a.csv';b=tmp_path/'b.csv';task=sp['tasks'][0]
    profile(a,sp);profile(b,sp,task)
    x,rec=lp.fold_shares(a,sp,task);y,_=lp.fold_shares(b,sp,task)
    torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert set(rec['counts'].values())=={60}
    assert len(rec['prompt_ids'])==60
    assert not any(i//500==0 for i in rec['prompt_ids'])
    broken=json.loads((ROOT/'inputs/splits.json').read_text())
    broken['loto'][task]['train'].append(410)
    # Remove the original checksum so structural fold validation is exercised.
    broken.pop('sha1');path=tmp_path/'bad.json';path.write_text(json.dumps(broken))
    with pytest.raises(ValueError,match='Incorrect Wenbo LOTO'):
        lp.checked_protocol(path)


def test_fold_weight_denominator_and_initialization_inputs(tmp_path):
    sp=lp.checked_protocol(ROOT/'inputs/splits.json');task=sp['tasks'][0]
    shutil.copy(ROOT/'inputs/splits.json',tmp_path/'splits.original.json')
    profile(tmp_path/'profile.csv',sp)
    tr=sp['loto'][task]['train'];selected=tr[::max(1,len(tr)//2)][:2]
    folder=tmp_path/'labels/train';folder.mkdir(parents=True)
    for i in selected:
        torch.save(dict(score=torch.ones(112,1100),fut=torch.rand(112,1100)),folder/f'{i}.pt')
    fold,r=lp.prepare_fold(tmp_path,tmp_path/'labels',0,0,2,smoke=True)
    assert r['weight_label_ids']==selected
    assert all(sp['task_of'][str(i)]!=task for i in r['train_ids']+r['val_ids']+r['weight_label_ids'])
    assert all(sp['task_of'][str(i)]==task for i in r['test_ids'])
    assert [len(r[k]) for k in ('train_ids','val_ids','test_ids')]==[4,4,2]
    assert (fold/'head_weights_kvp.pt').exists()


def test_merge_requires_all_13_disjoint_budget_feasible_folds(tmp_path):
    sp=lp.checked_protocol(ROOT/'inputs/splits.json')
    shutil.copy(ROOT/'inputs/splits.json',tmp_path/'splits.original.json')
    lengths={str(i):1100 for i in sp['test']}
    lp.write_json(tmp_path/'features/features_manifest.json',dict(lengths=lengths))
    for index,task in enumerate(sp['tasks']):
        fold=tmp_path/f'folds/{index:02d}_{task}';ids=sp['loto'][task]['test']
        lp.write_json(fold/'fold.json',dict(heldout=task,test_ids=ids,seed=0))
        lp.write_json(fold/'complete.json',dict(complete=True,heldout=task))
        lp.write_json(fold/'allocs/fixed_r0.05.json',dict(ratio=.05,per_prompt={str(i):[55]*112 for i in ids}))
    a=lp.merge_folds(tmp_path,'fixed',.05)
    assert len(a['per_prompt'])==260
    first=str(sp['test'][0]);a['per_prompt'][first][0]+=1
    with pytest.raises(ValueError,match='Budget mismatch'):
        lp.validate_counts(a,sp['test'],lengths)
    (tmp_path/f'folds/00_{sp["tasks"][0]}/complete.json').unlink()
    with pytest.raises(FileNotFoundError):lp.merge_folds(tmp_path,'fixed',.05)


def test_features_forbid_future_labels_and_validate_rankings():
    tree=ast.parse((ROOT/'loto/gpu_inference.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_feature')
    ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'feature_validator','exec'),ns)
    P=100;order=torch.tensor([*range(4),*range(P-16,P),*range(4,P-16)])
    d=dict(id=1,score=torch.ones(112,P),rank=order.repeat(112,1).short())
    assert ns['validate_feature'](d,1)==P
    d['fut']=torch.ones(112,P)
    with pytest.raises(ValueError,match='prompt-only'):ns['validate_feature'](d,1)
    feature_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='features')
    names={getattr(n.func,'id','') for n in ast.walk(feature_fn) if isinstance(n,ast.Call)}
    assert 'generate_with_compression' in names  # Match Wenbo's full-answer extraction pass.


def test_evaluation_head_count_strategy_matches_wenbo():
    original=ast.parse((ROOT/'vendor/wenbo/eval_alloc.py').read_text())
    adapted=ast.parse((ROOT/'loto/gpu_inference.py').read_text())
    a=next(n for n in original.body if isinstance(n,ast.ClassDef) and n.name=='FixedCountStrategy')
    b=next(n for n in adapted.body if isinstance(n,ast.ClassDef) and n.name=='FixedCountStrategy')
    assert ast.dump(a)==ast.dump(b)


def test_dependency_graph_without_submitting_jobs(tmp_path):
    root=tmp_path/'experiment';(root/'loto').mkdir(parents=True)
    for p in (ROOT/'loto').glob('*'):
        if p.is_file():shutil.copy(p,root/'loto'/p.name)
    shutil.copytree(ROOT/'vendor',root/'vendor')
    for n in ('prepare.py','finish.py','label_carc.py','label_io.py','ruler_scoring.py','patch_a40_attention.py'):
        shutil.copy(ROOT/n,root/n)
    sp=lp.checked_protocol(ROOT/'inputs/splits.json')
    data=tmp_path/'data';data.mkdir();shutil.copy(ROOT/'inputs/splits.json',data/'splits.json')
    profile(data/'profile.csv',sp)
    for split in ('train','val'):
        d=data/'labels'/split;d.mkdir(parents=True)
        for i in sp[split]:(d/f'{i}.pt').write_bytes(b'placeholder; preflight checks presence only')
    lp.write_json(data/'labels/label_coverage.json',dict(complete=True,completed={s:len(sp[s]) for s in ('train','val')}))
    shutil.copy(ROOT/'inputs/ruler845_b128.yaml',data/'composite.yaml')
    config=dict(PYTHON_BIN=sys.executable,SPLITS_JSON=str(data/'splits.json'),LABELS_DIR=str(data/'labels'),
        PROFILE_CSV=str(data/'profile.csv'),EXPERIMENT_ROOT=str(tmp_path/'runs'),
        COMPOSITE=str(data/'composite.yaml'),CARC_ACCOUNT='ctorng_1735',CARC_PARTITION='gpu',CARC_GPU_TYPE='a40')
    import shlex
    (root/'config.sh').write_text('\n'.join(f'{k}={shlex.quote(v)}' for k,v in config.items())+'\nMATCH_SNAPKV_CSVS=()\n')
    bindir=tmp_path/'bin';bindir.mkdir();calls=tmp_path/'calls.jsonl'
    fake=bindir/'sbatch'
    fake.write_text('#!'+sys.executable+'\nimport os,json,sys\nfrom pathlib import Path\n'
        'p=Path(os.environ["MOCK_CALLS"])\nn=len(p.read_text().splitlines()) if p.exists() else 0\n'
        'with p.open("a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\nprint(100+n)\n')
    fake.chmod(0o755)
    env=dict(os.environ,PATH=str(bindir)+os.pathsep+os.environ['PATH'],MOCK_CALLS=str(calls),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    r=subprocess.run(['bash',str(root/'loto/submit_pipeline.sh')],env=env,capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stdout+r.stderr
    args=[json.loads(l) for l in calls.read_text().splitlines()]
    assert len(args)==5
    for n in range(1,5):assert f'--dependency=afterok:{99+n}' in args[n]
    assert '--array=0-12%2' in args[2]
    assert '--array=0-5%2' in args[3]
