"""LongBench (v1, English subset) in the same row format as simonjegou/ruler, for label.py / eval_alloc.py.

Rows: {"context": <full LongBench prompt, middle-truncated>, "question": "", "answer_prefix": "",
       "answer": [answers], "task": <dataset>, "max_new_tokens": <LongBench max gen>, "all_classes": [...]}
so build_prompt() puts the whole official LongBench prompt in the user turn (chat template applied to all tasks).

  build:  python alloc/lb_data.py build --per_task 40 --maxp 3900 --out lb_rows_L4096.jsonl --ids alloc_ids_LLB.json
  use:    LONGBENCH=lb_rows_L4096.jsonl  -> label.py / eval_alloc.py load these rows instead of RULER
          (ids = row index; scoring switches to the official LongBench metric for these tasks)
Metrics follow LongBench's metrics.py: qa_f1 (qasper, multifieldqa_en, hotpotqa, 2wikimqa),
retrieval (passage_retrieval_en), classification (trec, first line only). Scores x100 to match RULER.
"""
import argparse
import json
import os
import re
import string
import zipfile
from collections import Counter

PROMPTS = {
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
}
MAXGEN = {"qasper": 128, "multifieldqa_en": 64, "hotpotqa": 32, "2wikimqa": 32, "passage_retrieval_en": 32, "trec": 64}


# ---------------- metrics (LongBench metrics.py) ----------------
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred, gt):
    p, g = normalize_answer(pred).split(), normalize_answer(gt).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def retrieval(pred, gt):
    m = re.findall(r"Paragraph (\d+)", gt)
    gid = m[0] if m else gt
    nums = re.findall(r"\d+", pred)
    return sum(n == str(gid) for n in nums) / len(nums) if nums else 0.0


def classification(pred, gt, all_classes):
    em = [c for c in all_classes if c in pred]
    for t in list(em):
        if t in gt and t != gt:
            em.remove(t)
    return 1.0 / len(em) if gt in em else 0.0


def score(pred, row, ruler_score):
    """Drop-in for ruler_score(pred, row['answer'], row['task']) in label.py / eval_alloc.py."""
    task = row["task"]
    if task not in PROMPTS:
        return ruler_score(pred, row["answer"], task)
    answers = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
    if task == "trec":
        pred = pred.lstrip("\n").split("\n")[0]
        return 100.0 * max(classification(pred, a, row["all_classes"]) for a in answers)
    if task == "passage_retrieval_en":
        return 100.0 * max(retrieval(pred, a) for a in answers)
    return 100.0 * max(qa_f1(pred, a) for a in answers)


# ---------------- data ----------------
def load_ds(load_dataset):
    """Drop-in for load_dataset('simonjegou/ruler', ...) in label.py / eval_alloc.py."""
    lb = os.environ.get("LONGBENCH")
    if lb:
        return [json.loads(l) for l in open(lb)]
    return load_dataset("simonjegou/ruler", data_dir=os.environ.get("RULER_LEN", "4096"), split="test")


def build(args):
    from huggingface_hub import hf_hub_download
    from label import MODEL_DIR
    from torchtune.models.qwen2_5 import qwen2_5_tokenizer
    tok = qwen2_5_tokenizer(path=str(MODEL_DIR / "vocab.json"), merges_file=str(MODEL_DIR / "merges.txt"))
    zp = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    rows, n_trunc = [], 0
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        for task in args.tasks:
            f = next(n for n in names if n.endswith(f"/{task}.jsonl") or n == f"{task}.jsonl")
            lines = z.read(f).decode().splitlines()[:args.per_task]
            for l in lines:
                ex = json.loads(l)
                prompt = PROMPTS[task].format(context=ex["context"], input=ex["input"])
                ids = tok.encode(prompt, add_bos=False, add_eos=False)
                if len(ids) > args.maxp:          # LongBench-style middle truncation
                    h = args.maxp // 2
                    prompt = tok.decode(ids[:h], skip_special_tokens=True) + tok.decode(ids[-h:], skip_special_tokens=True)
                    n_trunc += 1
                rows.append({"context": prompt, "question": "", "answer_prefix": "", "answer": ex["answers"],
                             "task": task, "max_new_tokens": MAXGEN[task], "all_classes": ex.get("all_classes") or [],
                             "lb_id": ex.get("_id")})
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump({"test": list(range(len(rows))), "task_of": {str(i): r["task"] for i, r in enumerate(rows)}},
              open(args.ids, "w"))
    print(f"wrote {len(rows)} rows ({n_trunc} middle-truncated to {args.maxp} tokens) -> {args.out}, ids -> {args.ids}")
    print(Counter(r["task"] for r in rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--tasks", nargs="+", default=list(PROMPTS))
    ap.add_argument("--per_task", type=int, default=40)
    ap.add_argument("--maxp", type=int, default=3900)
    ap.add_argument("--out", default="lb_rows_L4096.jsonl")
    ap.add_argument("--ids", default="alloc_ids_LLB.json")
    build(ap.parse_args())
