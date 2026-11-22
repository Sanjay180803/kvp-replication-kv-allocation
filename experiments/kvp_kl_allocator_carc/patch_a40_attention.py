"""Apply the team's A40 FlexAttention workaround before importing Apple modules."""
import argparse
import ast
from pathlib import Path

OPTIONS = {'BLOCK_M': 64, 'BLOCK_N': 64, 'num_stages': 1}


def patch_source(source):
    calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
             and {'key', 'value', 'block_mask'}.issubset({k.arg for k in n.keywords})
             and any(k.arg == 'block_mask' and isinstance(k.value, ast.Name)
                     and k.value.id == 'block_mask' for k in n.keywords)]
    if len(calls) != 1:
        raise ValueError(f'Expected exactly one Apple block-mask attention call; found {len(calls)}')
    call = calls[0]
    old = next((k for k in call.keywords if k.arg == 'kernel_options'), None)
    if old:
        if ast.literal_eval(old.value) == OPTIONS:
            return source
        raise ValueError('Existing kernel_options differ; inspect the Apple checkout before patching')
    lines = source.splitlines(keepends=True)
    if lines[call.end_lineno - 1].strip() != ')':
        raise ValueError('Unexpected attention call layout; refusing to rewrite it')
    kw = next(k for k in call.keywords if k.arg == 'block_mask')
    indent = lines[kw.lineno - 1][:kw.col_offset]
    lines.insert(call.end_lineno - 1, indent + 'kernel_options=' + repr(OPTIONS) + ',\n')
    updated = ''.join(lines)
    ast.parse(updated)
    return updated


def apply_patch(apple_repo):
    target = Path(apple_repo)/'src/kvcompression/kv_cache/compressible_attention.py'
    old = target.read_text()
    new = patch_source(old)
    if old == new:
        print('A40 FlexAttention kernel options already configured.', flush=True)
        return
    backup = target.with_name(target.name + '.before_kvp_alloc_a40')
    if not backup.exists():
        with backup.open('x') as f:
            f.write(old)
    tmp = target.with_name(target.name + '.kvp_alloc_tmp')
    tmp.write_text(new)
    tmp.replace(target)
    print(f'Applied A40 FlexAttention options {OPTIONS}. Backup: {backup}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--apple-repo', required=True)
    apply_patch(ap.parse_args().apple_repo)
