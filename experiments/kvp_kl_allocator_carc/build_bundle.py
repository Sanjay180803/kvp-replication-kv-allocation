"""Build a portable source/profile bundle; no environments, model weights or labels."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile-dir', type=Path)
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    profile = a.profile_dir or here.parent/'kvp_kl_carc/results/profile_14372014'
    sources = {name:here/name for name in (
        'README.md', 'config.example.sh', 'train.sbatch', 'submit.sh', 'run.sh',
        'prepare.py', 'finish.py', 'build_bundle.py', 'SOURCE_MANIFEST.json', '.gitignore',
        'label.sbatch', 'submit_labels.sh', 'run_labels.sh', 'label_carc.py', 'label_io.py',
        'ruler_scoring.py', 'patch_a40_attention.py', 'inputs/splits.json', 'inputs/ruler845_b128.yaml',
        'vendor/wenbo/train_alloc.py', 'vendor/wenbo/alloc_lib.py',
        'vendor/wenbo/splits.py', 'vendor/wenbo/head_weights.py', 'vendor/wenbo/label.py','vendor/wenbo/eval_alloc.py',
        'tests/test_setup.py','tests/test_loto.py')}
    for name in ('head_kl.csv', 'experiment.json', 'coverage.json'):
        sources['profiles/'+name] = profile/name
    for p in (here/'loto').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='settings.sh':
            sources[str(p.relative_to(here))]=p
    target = here/'artifacts/kvp-kl-allocator-carc.tar.gz'
    target.parent.mkdir(exist_ok=True)
    manifest = {n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in sources.items()}
    with tarfile.open(target, 'w:gz') as t:
        for name,p in sources.items():
            t.add(p,arcname='kvp-kl-allocator-carc/'+name,recursive=False)
        blob = (json.dumps(manifest,indent=2)+'\n').encode()
        info = tarfile.TarInfo('kvp-kl-allocator-carc/BUNDLE_MANIFEST.json')
        info.size=len(blob)
        t.addfile(info,io.BytesIO(blob))
    print(f'{target} ({target.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
