"""Validate saved allocator outputs and report the best validation checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    run = Path(a.run)
    result = {}
    for mode in ('fixed', 'input'):
        p = run/'work'/f'{mode}_hw_s{a.seed}'
        summary = json.loads((p/'summary.json').read_text())
        ck = torch.load(p/'best.pt', map_location='cpu', weights_only=False)
        # This is a checkpoint produced by this job, containing argparse metadata.
        if ck['mode'] != mode or ck['step'] != summary['best_step']:
            raise ValueError(f'Checkpoint/summary mismatch: {p}')
        if not all(torch.isfinite(t).all() for t in ck['model'].values()):
            raise ValueError(f'Nonfinite allocator parameters: {p}')
        result[mode] = dict(summary, checkpoint=str(p/'best.pt'),
                            sha256=hashlib.sha256((p/'best.pt').read_bytes()).hexdigest())
    (run/'training_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
