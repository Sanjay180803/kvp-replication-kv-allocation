"""Core pieces of the allocator (no language model needed): data, budgets, allocation rule, reward, references.

Notation (per prompt, P prompt tokens, H = 112 KV heads, head = 4*layer + kv_head):
  rank order   each head's frozen KVP ranking; keeping b tokens keeps the first b in rank order.
  NPRI = 20    the first 20 rank positions are KVP's always-kept tokens (4 sinks + 16 recent); every
               head keeps them, and they count toward the budget.
  budget       ratio r -> total B = 112 * int(r * P) tokens (exactly the team's uniform total);
               free slots B' = B - 112 * NPRI are shared among heads.
  fut          future attention per token (labels; see label.py); cumfut[h, n] = attention kept when
               head h keeps n of its free slots.
  value        allocator output: value[h, k] = w_h * softmax_k(score[h, NPRI+k] / t_h). Values fall along
               rank order, so keeping the top B' values over all heads keeps a prefix of every head.

Rewards follow KVP (paper eq. 3.4-3.7) with "one head, all budgets" -> "all heads, all budgets":
  evicted(B) = sum_h (total fut of head h on free slots) - sum_h cumfut[h, n_h(B)]
  reward     = - sum_{B in grid} evicted(alloc, B) / sum_{B in grid} evicted(best merge, B)   (<= -1)
The best merge maximises kept attention for the given frozen rankings (Lagrangian + greedy fill).
"""
import json
import math
from pathlib import Path

import torch

H, NPRI = 112, 20
TRAIN_RATIOS = (0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25)
N_FEAT = 11


# ----------------------------------------------------------------------------- data
def budgets_free(P, ratios):
    """Free slots per ratio for prompt length P (int tensor [R])."""
    return torch.tensor([max(0, H * int(r * P) - H * NPRI) for r in ratios], dtype=torch.long)


def head_features(score, P):
    """Allocator input from KVP scores only (rank order), [H, N_FEAT]. score: [H, P] float."""
    s = score[:, NPRI:].float()
    Pf = s.shape[1]
    p = torch.softmax(s, dim=1)
    cp = p.cumsum(1)
    feats = []
    for q in (0.01, 0.02, 0.05, 0.10, 0.25):
        k = max(1, int(q * Pf))
        c = cp[:, k - 1].clamp(1e-6, 1 - 1e-6)
        feats.append(torch.log(c) - torch.log1p(-c))
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(1) / math.log(Pf)
    feats += [ent, s[:, 0] - s.mean(1), s.std(1), s[:, 0] - s[:, min(9, Pf - 1)],
              s[:, 0] - s[:, Pf // 10], torch.full((H,), math.log(P))]
    return torch.stack(feats, 1)


class PromptSet:
    """All prompts of one split, padded to the longest prompt, kept on one device."""

    def __init__(self, label_dir, ids, device, need_fut=True, head_weights=None):
        self.ids, self.device = list(ids), device
        recs = [torch.load(Path(label_dir) / f"{i}.pt") for i in self.ids]
        self.P = torch.tensor([r["score"].shape[1] for r in recs])
        Pm = int(self.P.max()) - NPRI
        N = len(recs)
        self.score = torch.full((N, H, Pm), -1e4, dtype=torch.float16)
        self.mask = torch.zeros(N, 1, Pm, dtype=torch.bool)
        self.feat = torch.zeros(N, H, N_FEAT)
        self.cumfut = torch.zeros(N, H, Pm + 1) if need_fut else None
        self.futtot = torch.zeros(N, H) if need_fut else None
        for n, r in enumerate(recs):
            P = r["score"].shape[1]
            self.score[n, :, :P - NPRI] = r["score"][:, NPRI:]
            self.mask[n, 0, :P - NPRI] = True
            self.feat[n] = head_features(r["score"].float(), P)
            if need_fut:
                f = r["fut"].float()[:, NPRI:]
                if head_weights is not None:          # importance-weighted attention (see head_weights.py)
                    f = f * head_weights[:, None]
                self.cumfut[n, :, 1:P - NPRI + 1] = f.cumsum(1)
                self.cumfut[n, :, P - NPRI + 1:] = self.cumfut[n, :, P - NPRI:P - NPRI + 1]
                self.futtot[n] = f.sum(1)
        for k in ("score", "mask", "feat", "cumfut", "futtot", "P"):
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, v.to(device))

    def __len__(self):
        return len(self.ids)

    def budgets(self, idx, ratios):
        return torch.stack([budgets_free(int(self.P[i]), ratios) for i in idx]).to(self.device)  # [b, R]


# ----------------------------------------------------------------------------- allocation rule
def values(score, mask, logw, logt):
    """score [b,H,L] (fp16), mask [b,1,L], logw/logt [..., b, H] -> log-values [..., b, H, L]."""
    s = score.float()
    z = s / logt.exp()[..., None]
    z = z.masked_fill(~mask, -1e9)
    return torch.log_softmax(z, -1) + logw[..., None]


def counts_from_values(logv, budgets):
    """Free-slot counts per head for every budget. logv [..., H, L]; budgets [..., R] -> [..., R, H]."""
    shp = logv.shape[:-2]
    flat = logv.reshape(*shp, -1)
    srt = flat.sort(-1, descending=True).values
    thr = srt.gather(-1, (budgets - 1).clamp_min(0))                           # [..., R]
    n = (logv[..., None, :, :] >= thr[..., :, None, None]).sum(-1)            # [..., R, H]
    n = n * (budgets > 0)[..., None]
    # ties can overshoot by a few slots: trim the largest heads deterministically
    over = n.sum(-1) - budgets
    if (over > 0).any():
        n = trim(n, over)
    return n


def trim(n, over):
    n = n.clone()
    for _ in range(int(over.max())):
        need = over > 0
        if not need.any():
            break
        j = n.argmax(-1, keepdim=True)
        n.scatter_add_(-1, j, -need.long()[..., None])
        over = over - need.long()
    return n


def kept_mass(cumfut, n):
    """cumfut [b,H,L+1], n [..., b, R, H] -> kept attention [..., b, R]."""
    idx = n.transpose(-1, -2)                                    # [..., b, H, R]
    cf = cumfut.expand(*idx.shape[:-1], cumfut.shape[-1])
    return cf.gather(-1, idx).sum(-2)


# ----------------------------------------------------------------------------- references
@torch.no_grad()
def best_merge(cumfut, P_free, budgets, iters=50):
    """Best counts for given frozen rankings (max kept attention). cumfut [H, L+1] (one prompt),
    P_free = valid length L_p, budgets [R] -> counts [R, H]. Lagrangian on the concave envelope + greedy fill."""
    C = cumfut[:, :P_free + 1].double()
    ar = torch.arange(P_free + 1, device=C.device, dtype=C.dtype)
    B = budgets.to(C.device).double()                                      # [R]
    R = B.shape[0]
    lo = torch.zeros(R, dtype=C.dtype, device=C.device)
    hi = torch.full((R,), float((C[:, 1:] - C[:, :-1]).max()) + 1.0, dtype=C.dtype, device=C.device)
    for _ in range(iters):
        lam = (lo + hi) / 2
        n = (C[None] - lam[:, None, None] * ar).argmax(-1)                 # [R, H]
        too_many = n.sum(-1) > B
        lo = torch.where(too_many, lam, lo)
        hi = torch.where(too_many, hi, lam)
    n_hi = (C[None] - hi[:, None, None] * ar).argmax(-1)                    # sum <= B
    n_lo = (C[None] - lo[:, None, None] * ar).argmax(-1)                    # sum >  B (or = when exact)
    # the remaining slots lie on hull segments whose slope is ~lambda*: any split along them is optimal
    left = (B.long() - n_hi.sum(-1)).clamp_min(0)                            # [R]
    room = (n_lo - n_hi).clamp_min(0)                                        # [R, H]
    before = room.cumsum(-1) - room
    add = (left[:, None] - before).clamp(min=0)
    add = torch.minimum(add, room)
    n = n_hi + add
    left = B.long() - n.sum(-1)                                              # only if the bracket was degenerate
    if (left > 0).any():
        for r in range(R):
            k = 0
            while left[r] > 0 and k < 10 * H:
                j = k % H
                if n[r, j] < P_free:
                    n[r, j] += 1; left[r] -= 1
                k += 1
    return n


def static_split(ps, idx, ratios, grid=2000):
    """Fixed fractions per head and budget, planned on the train-average of share-normalised kept-attention
    curves (the team's 'shareplan' idea, in attention units). Returns fractions [R, H] of free slots."""
    acc = torch.zeros(H, grid + 1, device=ps.device)
    g = torch.linspace(0, 1, grid + 1, device=ps.device)
    for i in idx:
        L = int(ps.P[i]) - NPRI
        pos = (g * L).round().long()
        c = ps.cumfut[i][:, pos]
        acc += c / ps.futtot[i].sum().clamp_min(1e-9)
    acc /= len(idx)
    fr = []
    for r in ratios:
        # budget in fraction-of-free-slots units, averaged over the given prompts
        tot = sum((H * int(r * int(ps.P[i])) - H * NPRI) / (int(ps.P[i]) - NPRI) for i in idx) / len(idx)
        steps = int(round(tot * grid))
        n = best_merge(acc, grid, torch.tensor([steps]))[0]
        fr.append(n.float() / grid)
    return torch.stack(fr)


def counts_from_fractions(fr, P, ratios):
    """Turn per-head fractions [R,H] into exact free-slot counts for a prompt of length P."""
    L = P - NPRI
    B = budgets_free(P, ratios)
    out = []
    for r_i in range(len(ratios)):
        raw = fr[r_i] * L
        n = raw.floor().long().clamp(0, L)
        rem = int(B[r_i]) - int(n.sum())
        frac = raw - raw.floor()
        order = frac.argsort(descending=True) if rem >= 0 else n.argsort(descending=True)
        k = 0
        while rem != 0:
            j = order[k % H]
            if rem > 0 and n[j] < L:
                n[j] += 1; rem -= 1
            elif rem < 0 and n[j] > 0:
                n[j] -= 1; rem += 1
            k += 1
        out.append(n)
    return torch.stack(out)


def save_allocation(path, name, ratio, per_prompt_free_counts):
    """per_prompt_free_counts: {id: tensor [H] free counts} -> eval_alloc.py JSON (total counts incl. NPRI)."""
    json.dump({"name": name, "ratio": ratio,
               "per_prompt": {str(i): [int(c) + NPRI for c in n.tolist()] for i, n in per_prompt_free_counts.items()}},
              open(path, "w"))
