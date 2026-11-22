"""Pure CPU LOTO protocol checks, reward weights, allocation merging and reports."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import prepare as base

RATIOS=(.05,.10)


def write_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)


def checked_protocol(path):
    sp=base.check_splits(path)
    for task in sp['tasks']:
        expected={s:[i for i in sp[s] if (sp['task_of'][str(i)]==task)==(s=='test')]
                  for s in ('train','val','test')}
        if sp['loto'][task]!=expected:
            raise ValueError(f'Incorrect Wenbo LOTO fold: {task}')
    return sp


def fold_shares(profile,sp,heldout,match_paths=()):
    # First validate all source groups using the ordinary profile validator.
    base.profile_shares(profile,sp,match_paths)
    allowed=None
    if match_paths:
        allowed=set()
        for p in match_paths:
            with Path(p).open() as f:
                for r in csv.DictReader(f):
                    lv=float(r['keep_ratio'])
                    if r['scorer']=='snapkv' and lv in base.LEVELS:
                        allowed.add((int(r['prompt_id']),lv))
    groups=defaultdict(lambda:torch.zeros(112))
    with Path(profile).open() as f:
        for r in csv.DictReader(f):
            lv=float(r['keep_ratio']);pid=int(r['prompt_id'])
            if r['scorer']!='kvp' or lv not in base.LEVELS or r['task']==heldout:continue
            if allowed is not None and (pid,lv) not in allowed:continue
            groups[(pid,lv)][4*int(r['layer'])+int(r['kv_head'])]=float(r['kl_mean'])
    sums=torch.zeros(4,112);counts=torch.zeros(4)
    for (pid,lv),v in groups.items():
        if sp['tasks'][pid//500]==heldout:raise ValueError('Profiling task leakage')
        if v.sum()>0:
            k=base.LEVELS.index(lv);sums[k]+=v/v.sum();counts[k]+=1
    if not (counts>0).all():raise ValueError('Empty LOTO weight level')
    return sums/counts[:,None],dict(counts={str(lv):int(counts[k]) for k,lv in enumerate(base.LEVELS)},
                                  prompt_ids=sorted({pid for pid,_ in groups}))


def prepare_fold(pipeline,labels,index,seed,n_train,match_paths=(),smoke=False):
    p=Path(pipeline);sp=checked_protocol(p/'splits.original.json');task=sp['tasks'][index]
    tr=sp['loto'][task]['train'];va=sp['loto'][task]['val'];te=sp['loto'][task]['test']
    shares,prof=fold_shares(p/'profile.csv',sp,task,match_paths)
    weights,denom_ids=base.derive_weights(shares,labels,tr,n_train)
    if any(sp['task_of'][str(i)]==task for i in denom_ids):raise ValueError('Denominator leakage')
    if smoke:
        tr=tr[::max(1,len(tr)//4)][:4];va=va[::max(1,len(va)//4)][:4];te=te[:2]
    fold=p/('smoke/fold' if smoke else f'folds/{index:02d}_{task}')
    fold.mkdir(parents=True,exist_ok=True)
    if (fold/'fold.json').exists():raise ValueError('Fold exists; use a fresh pipeline')
    derived=dict(train=tr,val=va,test=te,task_of=sp['task_of'],loto={})
    write_json(fold/'splits.json',derived);torch.save(weights,fold/'head_weights_kvp.pt')
    record=dict(heldout=task,index=index,seed=seed,smoke=smoke,
        profiling=prof,weight_label_ids=denom_ids,train_ids=tr,val_ids=va,test_ids=te,
        original_split_sha256=base.sha256(p/'splits.original.json'),profile_sha256=base.sha256(p/'profile.csv'),
        weights_sha256=base.sha256(fold/'head_weights_kvp.pt'),
        match_profiles={str(x):base.sha256(x) for x in match_paths})
    write_json(fold/'fold.json',record)
    return fold,record


def validate_counts(allocation,ids,lengths):
    if set(map(int,allocation['per_prompt']))!=set(ids):raise ValueError('Wrong allocation prompt coverage')
    ratio=float(allocation['ratio'])
    if ratio not in RATIOS:raise ValueError('Unexpected ratio')
    for pid in ids:
        c=allocation['per_prompt'][str(pid)];P=lengths[str(pid)]
        if len(c)!=112 or any(type(v) is not int or not 20<=v<=P for v in c):
            raise ValueError(f'Invalid head counts: {pid}')
        if sum(c)!=112*int(ratio*P):raise ValueError(f'Budget mismatch: {pid}')


def merge_folds(pipeline,policy,ratio):
    p=Path(pipeline);sp=checked_protocol(p/'splits.original.json')
    features=json.loads((p/'features/features_manifest.json').read_text())
    per={};fold_records=[]
    for index,task in enumerate(sp['tasks']):
        fold=p/f'folds/{index:02d}_{task}'
        rec=json.loads((fold/'fold.json').read_text())
        done=json.loads((fold/'complete.json').read_text())
        if not done['complete'] or done['heldout']!=task:raise ValueError('Fold not complete')
        if rec['heldout']!=task or rec['test_ids']!=sp['loto'][task]['test']:
            raise ValueError(f'Wrong fold record: {task}')
        a=json.loads((fold/f'allocs/{policy}_r{ratio}.json').read_text())
        validate_counts(a,rec['test_ids'],features['lengths'])
        if set(per)&set(a['per_prompt']):raise ValueError('Duplicate merged IDs')
        per.update(a['per_prompt']);fold_records.append(rec)
    if len({r['seed'] for r in fold_records})!=1:raise ValueError('Mixed training seeds across folds')
    out=dict(name=f'loto_kvpkl_{policy}_s{fold_records[0]["seed"]}_r{ratio}',ratio=ratio,per_prompt=per)
    validate_counts(out,sp['test'],features['lengths'])
    return out


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('preflight');q.add_argument('--splits',required=True);q.add_argument('--labels',required=True);q.add_argument('--profile',required=True)
    q=sub.add_parser('record');q.add_argument('--out',required=True);q.add_argument('--jobs',nargs=5,required=True);q.add_argument('--seed',type=int,required=True)
    a=ap.parse_args()
    if a.cmd=='preflight':
        sp=checked_protocol(a.splits)
        c=json.loads((Path(a.labels)/'label_coverage.json').read_text())
        if not c['complete'] or c['completed']!={s:len(sp[s]) for s in ('train','val')}:
            raise ValueError('TRAIN/VAL labels not complete')
        for s in ('train','val'):
            for i in sp[s]:
                if not (Path(a.labels)/s/f'{i}.pt').is_file():raise ValueError(f'Missing {s} {i}')
        base.profile_shares(a.profile,sp)
        print('Preflight: exact 13-fold protocol, full KVP profile and TRAIN/VAL labels present.')
    else:
        write_json(Path(a.out)/'jobs.json',dict(seed=a.seed,jobs=dict(zip(['smoke','features','folds','eval','summary'],a.jobs))))


if __name__=='__main__':main()
