"""Build the train / val / test prompt splits for the allocator and check them for leakage.

RULER-4K (simonjegou/ruler, data_dir=4096, split=test) has 13 tasks x 500 rows; task t = rows 500t..500t+499.
Within each task block (team convention, see team README and scripts/rl_alloc.py):
  rows   0-399  KVP region: the team's 845 KVP training prompts (65 per task) and the 260 test prompts
                (first 20 rows per task not used for KVP training) live here. We never train on this region.
  rows 400-404  head profiling (team)          -> not used here
  rows 405-409  validation                      -> our VAL (65 prompts)
  rows 410-499  allocator training (team used 24/task) -> our TRAIN (90/task = 1,170 prompts)
TEST = the exact 260 prompt IDs of the team's results (read from a result file), so numbers are comparable.

Leave-one-task-out (LOTO) folds: for held-out task T, train and val use only the OTHER 12 tasks, and the
test set is T's 20 test prompts. Model selection inside a fold must use that fold's val only.

Checks (fail loudly): every block holds one task; TRAIN, VAL, TEST pairwise disjoint; TRAIN/VAL outside the
KVP region; TEST inside it; 20 test prompts per task. Also reports exact answer overlap between TEST and
TRAIN/VAL (RULER reuses haystacks, so the needle/answer is the part that must not repeat).

Usage (repo root, uv env):  uv run python $ALLOC/splits.py --test_ids_from <team results jsonl> --out $DATA/splits.json
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict

from datasets import load_dataset

N_TASKS, BLOCK = 13, 500
VAL_ROWS = range(405, 410)
TRAIN_ROWS = range(410, 500)
KVP_REGION_END = 400


def ans_key(a):
    a = list(a) if isinstance(a, (list, tuple)) else [a]
    return tuple(sorted(str(x).strip().lower() for x in a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_ids_from", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = load_dataset("simonjegou/ruler", data_dir="4096", split="test")
    assert len(ds) == N_TASKS * BLOCK, len(ds)
    tasks = ds["task"]
    block_task = []
    for t in range(N_TASKS):
        names = set(tasks[t * BLOCK:(t + 1) * BLOCK])
        assert len(names) == 1, f"block {t} mixes tasks {names}"
        block_task.append(names.pop())
    assert len(set(block_task)) == N_TASKS

    test = sorted({json.loads(l)["id"] for l in open(args.test_ids_from)})
    val = [BLOCK * t + r for t in range(N_TASKS) for r in VAL_ROWS]
    train = [BLOCK * t + r for t in range(N_TASKS) for r in TRAIN_ROWS]

    S = {"train": set(train), "val": set(val), "test": set(test)}
    for a in S:
        for b in S:
            if a < b:
                assert not (S[a] & S[b]), f"{a} and {b} overlap: {sorted(S[a] & S[b])[:5]}"
    assert all(i % BLOCK >= KVP_REGION_END for i in train + val), "train/val inside the KVP region"
    assert all(i % BLOCK < KVP_REGION_END for i in test), "test outside the KVP region"
    per_task = Counter(tasks[i] for i in test)
    assert len(test) == 260 and set(per_task.values()) == {20}, per_task

    # answer overlap: drop train/val prompts whose exact answer equals a test answer (strict; different
    # documents can share a short answer by chance, but the model should never be tuned on a test answer)
    answers = ds["answer"]
    test_ans = {ans_key(answers[i]): i for i in test}
    overlap = sorted({test_ans[ans_key(answers[i])] for i in train + val if ans_key(answers[i]) in test_ans})
    dropped = [i for i in train + val if ans_key(answers[i]) in test_ans]
    train = [i for i in train if i not in dropped]
    val = [i for i in val if i not in dropped]
    by_task = Counter(tasks[i] for i in overlap)
    assert not ({ans_key(answers[i]) for i in train + val} & set(test_ans))

    task_of = {i: tasks[i] for i in train + val + test}
    loto = {}
    for T in block_task:
        loto[T] = {"train": [i for i in train if task_of[i] != T],
                   "val": [i for i in val if task_of[i] != T],
                   "test": [i for i in test if task_of[i] == T]}

    out = {"dataset": "simonjegou/ruler 4096 test", "tasks": block_task,
           "train": train, "val": val, "test": test, "task_of": {str(k): v for k, v in task_of.items()},
           "loto": loto,
           "test_answer_overlap_with_trainval": {"n": len(overlap), "by_task": dict(by_task), "test_ids": overlap,
                                                 "dropped_trainval_ids": dropped}}
    blob = json.dumps(out, sort_keys=True)
    out["sha1"] = hashlib.sha1(blob.encode()).hexdigest()
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"train {len(train)}  val {len(val)}  test {len(test)}  -> {args.out}  sha1 {out['sha1'][:10]}")
    print(f"test prompts sharing an exact answer with train/val: {len(overlap)} {dict(by_task)}; "
          f"dropped train/val prompts: {dropped}")


if __name__ == "__main__":
    main()
