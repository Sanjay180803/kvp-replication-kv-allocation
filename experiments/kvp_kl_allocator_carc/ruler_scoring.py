"""The team's RULER string-match scoring, used only for label diagnostics."""


def score(pred, refs, task):
    refs = list(refs) if isinstance(refs, (list, tuple)) else [refs]
    p = pred.lower()
    hits = [str(r).lower() in p for r in refs]
    if task.startswith('qa'):
        return 100.0 * max(hits)
    return 100.0 * sum(hits) / len(hits)
