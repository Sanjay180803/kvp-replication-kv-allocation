"""Validate Wenbo's exact splits/labels and derive KVP KL reward weights on CPU."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch

H, NPRI = 112, 20
LEVELS = (0.02, 0.05, 0.10, 0.25)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_splits(path):
    sp = json.loads(Path(path).read_text())
    tasks = sp['tasks']
    if len(tasks) != 13 or len(set(tasks)) != 13:
        raise ValueError('Expected 13 RULER tasks')
    for split, lo, hi in [('train', 410, 500), ('val', 405, 410), ('test', 0, 400)]:
        ids = sp[split]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f'Empty/duplicate IDs in {split}')
        for i in ids:
            if not isinstance(i, int) or not 0 <= i < 6500 or not lo <= i % 500 < hi:
                raise ValueError(f'Invalid {split} ID: {i}')
            if sp['task_of'][str(i)] != tasks[i // 500]:
                raise ValueError(f'Wrong task mapping for {i}')
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if set(sp[a]) & set(sp[b]):
            raise ValueError(f'{a}/{b} overlap')
    if len(sp['test']) != 260 or set(Counter(sp['task_of'][str(i)] for i in sp['test']).values()) != {20}:
        raise ValueError('Expected 20 test IDs per task')
    dropped = sp['test_answer_overlap_with_trainval']['dropped_trainval_ids']
    if len(dropped) != len(set(dropped)) or any(not 405 <= i % 500 < 500 for i in dropped):
        raise ValueError('Invalid recorded answer-overlap exclusions')
    for split, offsets in [('train', range(410, 500)), ('val', range(405, 410))]:
        expected = {500*t + j for t in range(13) for j in offsets} - set(dropped)
        if set(sp[split]) != expected:
            raise ValueError(f'{split} differs from task offsets minus recorded exclusions')
    if 'sha1' in sp:
        blob = {k: v for k, v in sp.items() if k != 'sha1'}
        if hashlib.sha1(json.dumps(blob, sort_keys=True).encode()).hexdigest() != sp['sha1']:
            raise ValueError('Wenbo split SHA1 is inconsistent')
    return sp


def profile_shares(path, sp, match_paths=()):
    calibration = {500*t + j for t in range(13) for j in range(400, 405)}
    selected = {(i, lv) for i in calibration for lv in LEVELS}
    if match_paths:
        matching_heads = defaultdict(set)
        for p in match_paths:
            with Path(p).open() as f:
                reader = csv.DictReader(f)
                if not {'layer', 'kv_head'}.issubset(reader.fieldnames or []):
                    raise ValueError('Matching profile groups must cover all 112 heads')
                for r in reader:
                    lv = float(r['keep_ratio'])
                    if r['scorer'] == 'snapkv' and lv in LEVELS:
                        matching_heads[(int(r['prompt_id']), lv)].add((int(r['layer']), int(r['kv_head'])))
        selected = set(matching_heads)
        if not selected or any(i not in calibration for i, _ in selected):
            raise ValueError('Matching profiles must use calibration rows 400-404')
        expected_heads = {(l, h) for l in range(28) for h in range(4)}
        if any(heads != expected_heads for heads in matching_heads.values()):
            raise ValueError('Matching profile groups must cover all 112 heads')
    groups = defaultdict(dict)
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            lv, pid = float(r['keep_ratio']), int(r['prompt_id'])
            if r['scorer'] != 'kvp' or lv not in LEVELS:
                continue
            if pid not in calibration or r['task'] != sp['tasks'][pid // 500]:
                raise ValueError(f'Invalid profiling task/ID: {pid}')
            L, h = int(r['layer']), int(r['kv_head'])
            v = float(r['kl_mean'])
            P, n = int(r['prompt_len']), int(r['n_evicted'])
            if not (0 <= L < 28 and 0 <= h < 4 and math.isfinite(v) and v >= 0):
                raise ValueError(f'Invalid KL/head for {pid}')
            if int(r['n_pos']) < 1 or P - n != max(1, int(lv * P)):
                raise ValueError(f'Invalid continuation/token count for {pid}')
            key, head = (pid, lv), 4*L+h
            if head in groups[key]:
                raise ValueError(f'Duplicate profile row: {key}, {head}')
            groups[key][head] = v
    # Validate the full KVP profile before applying any baseline coverage mask.
    full = {(i, lv) for i in calibration for lv in LEVELS}
    if set(groups) != full or any(set(g) != set(range(H)) for g in groups.values()):
        raise ValueError('KVP profile must cover 65 prompts x 112 heads x all four weighting levels')
    sums = torch.zeros(len(LEVELS), H)
    counts = torch.zeros(len(LEVELS))
    for pid, lv in sorted(selected):
        v = torch.tensor([groups[(pid, lv)][h] for h in range(H)])
        if v.sum() > 0:
            k = LEVELS.index(lv)
            sums[k] += v / v.sum()
            counts[k] += 1
    if not (counts > 0).all():
        raise ValueError('At least one weighting level has no positive KL signal')
    return sums / counts[:, None], {str(lv): int(counts[k]) for k, lv in enumerate(LEVELS)}


def derive_weights(share, labels, train_ids, n_train):
    if n_train < 1:
        raise ValueError('n_train must be positive')
    # Preserve Wenbo's deterministic training-label selection and normalization.
    ids = train_ids[::max(1, len(train_ids)//n_train)][:n_train]
    lost = torch.zeros(len(LEVELS), H)
    for pid in ids:
        d = torch.load(Path(labels)/'train'/f'{pid}.pt', map_location='cpu', weights_only=True)
        fut = d['fut'].float()
        if fut.ndim != 2 or fut.shape[0] != H or fut.shape[1] <= NPRI:
            raise ValueError(f'Invalid future-attention shape: {pid}')
        if d['score'].shape != fut.shape or not torch.isfinite(fut).all() or (fut < 0).any():
            raise ValueError(f'Invalid saved label: {pid}')
        P = fut.shape[1]
        f = fut[:, NPRI:]
        cf = torch.cat([torch.zeros(H, 1), f.cumsum(1)], 1)
        tot = f.sum(1).clamp_min(1e-9)
        for k, lv in enumerate(LEVELS):
            n = max(0, int(lv*P)-NPRI)
            lost[k] += (tot-cf[:, n])/tot
    lost /= len(ids)
    if not (lost.sum(1) > 0).all():
        raise ValueError('No remaining future attention at a weighting level')
    lost_share = lost / lost.sum(1, keepdim=True)
    alpha = share.mean(0) / lost_share.mean(0).clamp_min(1e-9)
    alpha /= alpha.mean()
    if alpha.shape != (H,) or not torch.isfinite(alpha).all() or (alpha < 0).any():
        raise ValueError('Invalid derived head weights')
    return alpha, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', required=True)
    ap.add_argument('--splits', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-train', type=int, default=260)
    ap.add_argument('--match-snapkv', nargs='+', default=[])
    ap.add_argument('--mode', choices=['smoke', 'train'], default='train')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out/'run_config.json').exists():
        raise ValueError('Output already prepared; use a fresh run directory')
    sp = check_splits(a.splits)
    identities = []
    for split in ('train', 'val'):
        for pid in sp[split]:
            p = Path(a.labels)/split/f'{pid}.pt'
            st = p.stat()
            if not st.st_size:
                raise ValueError(f'Empty label: {p}')
            identities.append(dict(split=split, id=pid, bytes=st.st_size, mtime_ns=st.st_mtime_ns))
    shares, counts = profile_shares(a.profile, sp, a.match_snapkv)
    weights, ids = derive_weights(shares, a.labels, sp['train'], a.n_train)
    torch.save(weights, out/'head_weights_kvp.pt')
    used = dict(sp)
    if a.mode == 'smoke':
        used.pop('sha1', None)
        for split in ('train', 'val'):
            pool = sp[split]
            used[split] = pool[::max(1, len(pool)//4)][:4]
    (out/'splits.json').write_text(json.dumps(used, indent=2)+'\n')
    here = Path(__file__).resolve().parent
    source_paths = [here/'prepare.py', here/'finish.py', here/'run.sh', here/'config.sh',
                    here/'vendor/wenbo/train_alloc.py', here/'vendor/wenbo/alloc_lib.py']
    record = dict(mode=a.mode, seed=a.seed, profile=str(Path(a.profile).resolve()),
                  profile_sha256=sha256(a.profile), original_splits=str(Path(a.splits).resolve()),
                  splits_sha256=sha256(a.splits), split_sha1=sp.get('sha1'),
                  original_counts={s:len(sp[s]) for s in ('train','val','test')},
                  training_counts={s:len(used[s]) for s in ('train','val')},
                  weighting_levels=LEVELS, profiling_counts_per_level=counts,
                  matching_snapkv_profiles={str(p):sha256(p) for p in a.match_snapkv},
                  weight_label_ids=ids, labels_root=str(Path(a.labels).resolve()),
                  label_identities=identities, label_identity_kind='path+size+mtime; not content hashes',
                  weights_sha256=sha256(out/'head_weights_kvp.pt'), torch=torch.__version__,
                  source={str(p.relative_to(here)):sha256(p) for p in source_paths if p.exists()})
    (out/'run_config.json').write_text(json.dumps(record, indent=2)+'\n')
    print('Prepared:',record['original_counts'],'KL prompts per level:',counts, flush=True)
    print('Weights min/median/max:',weights.min().item(),weights.median().item(),weights.max().item(),flush=True)


if __name__ == '__main__':
    main()
