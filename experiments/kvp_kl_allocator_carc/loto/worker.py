"""Execute one frozen LOTO pipeline stage. Called by Slurm run.sh."""
import argparse
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT))
from protocol import (RATIOS,checked_protocol,prepare_fold,merge_folds,validate_counts,write_json)


def run(script,*args):
    command=[sys.executable,str(script),*map(str,args)]
    print('COMMAND:', ' '.join(command),flush=True)
    subprocess.run(command,check=True)


def inference_args(a,out):
    args=['--splits',Path(a.pipeline)/'splits.original.json','--labels',a.labels,
          '--model',a.model,'--composite',Path(a.pipeline)/'composite.original.yaml',
          '--checkpoint-root',a.checkpoint_root,'--out',out]
    if a.dataset_revision:args+=['--dataset-revision',a.dataset_revision]
    return args


def train_fold(a,smoke=False):
    fold,rec=prepare_fold(a.pipeline,a.labels,0 if smoke else a.index,a.seed,
        a.n_weight_train,a.match_snapkv,smoke)
    common=['--splits',fold/'splits.json','--labels',a.labels,'--work',fold/'work',
            '--device','cuda','--head_weights',fold/'head_weights_kvp.pt','--seed',a.seed,
            '--batch',1 if smoke else a.batch,'--K',2 if smoke else a.samples]
    extra=['--eval_every',1,'--patience',2] if smoke else []
    run(ROOT/'vendor/wenbo/train_alloc.py','train',*common,'--mode','fixed',
        '--steps',2 if smoke else a.fixed_steps,*extra)
    run(ROOT/'vendor/wenbo/train_alloc.py','train',*common,'--mode','input',
        '--steps',2 if smoke else a.input_steps,'--patience',a.patience,
        '--init',fold/f'work/fixed_hw_s{a.seed}/best.pt',*extra)
    run(ROOT/'finish.py','--run',fold,'--seed',a.seed)
    feature_root=Path(a.pipeline)/('smoke/features' if smoke else 'features')
    feature_manifest=json.loads((feature_root/'features_manifest.json').read_text())
    view=fold/'label_view';view.mkdir()
    for split in ('train','val'):(view/split).symlink_to((Path(a.labels)/split).resolve(),target_is_directory=True)
    (view/'test').symlink_to((feature_root/'test').resolve(),target_is_directory=True)
    for mode in ('static','fixed','input'):
        source=['--source','static'] if mode=='static' else [
            '--source','ckpt','--ckpt',fold/f'work/{mode}_hw_s{a.seed}/best.pt']
        run(ROOT/'vendor/wenbo/train_alloc.py','dump','--splits',fold/'splits.json',
            '--labels',view,'--split','test','--device','cuda','--head_weights',fold/'head_weights_kvp.pt',
            *source,'--name',mode,'--ratios',*RATIOS,'--out_dir',fold/'allocs')
        for ratio in RATIOS:
            alloc=json.loads((fold/f'allocs/{mode}_r{ratio}.json').read_text())
            validate_counts(alloc,rec['test_ids'],feature_manifest['lengths'])
    write_json(fold/'complete.json',dict(complete=True,heldout=rec['heldout'],seed=a.seed,
        test_ids=rec['test_ids'],checkpoint_selection='this fold VAL reward only'))
    print('SUCCESS: fold',rec['heldout'],flush=True)
    return fold


def paired_bootstrap(a,b,n=5000):
    ids=sorted(a)
    if set(a)!=set(b):raise ValueError('Unmatched evaluation IDs')
    delta=[a[i]-b[i] for i in ids];rng=random.Random(0)
    samples=sorted(sum(rng.choices(delta,k=len(delta)))/len(delta) for _ in range(n))
    return dict(mean=statistics.mean(delta),ci95=[samples[int(.025*n)],samples[int(.975*n)]],
                resamples=n,unit='paired prompt, not training seed',seed=0)


def summary(a):
    p=Path(a.pipeline);sp=checked_protocol(p/'splits.original.json');result={};paired={}
    for policy in ('static','fixed','input'):
        for ratio in RATIOS:
            out=p/f'eval/{policy}_r{ratio}'
            c=json.loads((out/'coverage.json').read_text())
            rows=[json.loads(l) for l in (out/'results.jsonl').read_text().splitlines() if l.strip()]
            if not c['complete'] or len(rows)!=260 or len({r['id'] for r in rows})!=260:
                raise ValueError('Incomplete/duplicate evaluation results')
            if {r['id'] for r in rows}!=set(sp['test']):raise ValueError('Wrong final test IDs')
            by={r['id']:r['score'] for r in rows}
            result[f'{policy}_r{ratio}']=dict(mean=statistics.mean(by.values()),n=260,
                tasks={t:statistics.mean(r['score'] for r in rows if r['task']==t) for t in sp['tasks']})
            paired[(policy,ratio)]=by
    report=dict(context=4096,seed=a.seed,protocol='13-task allocator LOTO; KVP frozen across all tasks',
        results=result,input_minus_fixed={str(r):paired_bootstrap(paired[('input',r)],paired[('fixed',r)]) for r in RATIOS},
        input_minus_static={str(r):paired_bootstrap(paired[('input',r)],paired[('static',r)]) for r in RATIOS})
    write_json(p/'summary.json',report)
    lines=['# KVP KL allocator: RULER-4K leave-one-task-out','',
        '13 folds, one seed per fold, 20 held-out test prompts per task.','',
        '| Policy | 5% | 10% |','| --- | ---: | ---: |']
    for m in ('static','fixed','input'):
        lines.append(f'| {m} | {result[f"{m}_r0.05"]["mean"]:.2f} | {result[f"{m}_r0.1"]["mean"]:.2f} |')
    lines+=['','Input minus fixed:']
    for r,d in report['input_minus_fixed'].items():
        lines.append(f'- {r}: {d["mean"]:+.2f}, paired-bootstrap 95% CI [{d["ci95"][0]:+.2f}, {d["ci95"][1]:+.2f}]')
    lines+=['','KVP itself saw the RULER task types during its earlier training. This tests allocator transfer, not whole-system unseen-task training.',
            'Confidence intervals cover paired test-prompt resampling, not training-seed uncertainty.']
    (p/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','features','fold','eval','summary'])
    for name in ('pipeline','labels','model','checkpoint-root'):ap.add_argument('--'+name,required=True)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--index',type=int,default=0)
    ap.add_argument('--fixed-steps',type=int,default=3000);ap.add_argument('--input-steps',type=int,default=10000)
    ap.add_argument('--patience',type=int,default=20);ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--samples',type=int,default=16);ap.add_argument('--n-weight-train',type=int,default=260)
    ap.add_argument('--match-snapkv',nargs='+',default=[]);ap.add_argument('--dataset-revision',default=None)
    a=ap.parse_args();p=Path(a.pipeline)
    if a.stage=='features':run(HERE/'gpu_inference.py','features',*inference_args(a,p/'features'))
    elif a.stage=='fold':
        if not 0<=a.index<13:raise ValueError('Fold index must be 0..12')
        train_fold(a)
    elif a.stage=='smoke':
        fr=p/'smoke/features'
        run(HERE/'gpu_inference.py','features',*inference_args(a,fr),'--smoke')
        fold=train_fold(a,True)
        run(HERE/'gpu_inference.py','eval',*inference_args(a,p/'smoke/eval'),
            '--features',fr,'--alloc',fold/'allocs/input_r0.1.json','--smoke')
    elif a.stage=='eval':
        if not 0<=a.index<6:raise ValueError('Eval index must be 0..5')
        policy,ratio=[(m,r) for m in ('static','fixed','input') for r in RATIOS][a.index]
        alloc=merge_folds(p,policy,ratio);dest=p/f'merged/{policy}_r{ratio}.json';write_json(dest,alloc)
        run(HERE/'gpu_inference.py','eval',*inference_args(a,p/f'eval/{policy}_r{ratio}'),
            '--features',p/'features','--alloc',dest)
    else:summary(a)
    print('SUCCESS: LOTO stage',a.stage,flush=True)


if __name__=='__main__':main()
