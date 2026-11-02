#!/usr/bin/env python
"""Per-KV-head cache sensitivity profiling for Qwen2.5-7B-Instruct on RULER-4K.

For every prompt, KV head (layer L, kv head h) and keep ratio r, shrink ONLY that
head's prompt cache (SnapKV ranking, or StreamingLLM) and measure how much the
next-token distributions over a teacher-forced 32-token continuation change.

Key implementation idea (exact, not an approximation)
-----------------------------------------------------
Post-prefill eviction only changes what continuation queries (positions >= P) see.
Prompt positions never attend to the continuation (causal), so every hidden state
at positions < P is identical to the full-cache pass. So the "compressed" forward
on prompt + c_1..c_K is computed as:
    * one full pass over prompt + c_1..c_K that captures the prompt K/V (post-RoPE)
      of every layer and the SnapKV scores, then
    * many cheap passes over only the K continuation tokens, which attend to the
      captured prompt K/V through an explicit per-item, per-query-head mask.
Each batch item carries its own eviction tensor [layers, kv_heads, P], so one
batch can mix different (L, h, r). Sanity check 5 verifies this equals the plain
full-length masked forward described in the task.

Usage: see README_profiling.md.
"""
import argparse
import csv
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

ATTN_NAME = "kvprof_sdpa"
WINDOW = 64          # SnapKV observation window (matches eval_ruler.py SnapKVPress)
POOL_K = 5           # SnapKV avg-pool kernel (padding = POOL_K // 2)
N_SINK = 4           # StreamingLLM sink tokens
K_CONT = 32          # teacher-forced continuation length (generate K+1)
RATIOS = (0.1, 0.25, 0.5)
CSV_FIELDS = ["prompt_id", "task", "scorer", "keep_ratio", "layer", "kv_head",
              "kl_mean", "kl_max", "top1_agree", "prompt_len", "n_evicted", "n_pos"]


# ----------------------------------------------------------------------------------
# Attention hook
# ----------------------------------------------------------------------------------
class State:
    """Global state read by the custom attention function."""
    mode = "off"          # off | capture | cont | full
    P = None              # prompt length
    # capture outputs
    prefix_k = {}         # layer -> [1, n_kv, P, d] post-RoPE keys
    prefix_v = {}         # layer -> [1, n_kv, P, d]
    snap_scores = {}      # layer -> [n_kv, P] fp32 SnapKV scores
    # eviction for cont/full modes: [B, n_layers, n_kv, P] bool, True = evicted
    evict = None


def snapkv_scores(query, key, P, scaling, n_groups):
    """Exactly SnapKVPress.sort in scripts/eval_ruler.py, for every KV head at once.

    query: [1, n_q, S, d], key: [1, n_kv, S, d] post-RoPE (S >= P). Returns [n_kv, P] fp32,
    window keys set to +inf (always kept).
    """
    w = min(WINDOW, P)
    q = query[:, :, P - w:P].float()                          # [1, n_q, w, d]
    k = repeat_kv(key[:, :, :P], n_groups).float()            # [1, n_q, P, d]
    attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])   # [1, n_q, w, P]
    pos = torch.arange(P, device=q.device)
    rows = torch.arange(P - w, P, device=q.device)
    attn = attn.masked_fill(pos[None, :] > rows[:, None], float("-inf"))
    n_kv = key.shape[1]
    s = attn.softmax(-1).view(n_kv, n_groups, w, P).mean(dim=(1, 2))      # mean over group heads + window
    s = F.avg_pool1d(s[:, None], POOL_K, stride=1, padding=POOL_K // 2)[:, 0]
    s[:, P - w:] = float("inf")
    return s


def eviction_mask_rows(evict_layer, n_groups, n_rows, prefix_len):
    """evict_layer [B, n_kv, P] -> allowed mask over prefix cols [B, n_q, n_rows, P]."""
    keep = ~evict_layer.repeat_interleave(n_groups, dim=1)   # [B, n_q, P]
    return keep[:, :, None, :].expand(-1, -1, n_rows, prefix_len)


def kvprof_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    mode = State.mode
    if mode == "off":
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      scaling=scaling, dropout=dropout, **kwargs)
    L = module.layer_idx
    G = module.num_key_value_groups
    P = State.P

    if mode == "capture":
        # full-cache pass over prompt + continuation (batch 1, causal)
        State.prefix_k[L] = key[:, :, :P].detach().clone()
        State.prefix_v[L] = value[:, :, :P].detach().clone()
        State.snap_scores[L] = snapkv_scores(query, key, P, scaling, G)
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      scaling=scaling, dropout=dropout, **kwargs)

    B, n_q, Q, d = query.shape
    ev = State.evict[:, L]                                    # [B, n_kv, P]

    if mode == "cont":
        # query/key/value cover only the K continuation tokens (positions P..P+K-1)
        pk = State.prefix_k[L].expand(B, -1, -1, -1)
        pv = State.prefix_v[L].expand(B, -1, -1, -1)
        k = repeat_kv(torch.cat([pk, key], dim=2), G)
        v = repeat_kv(torch.cat([pv, value], dim=2), G)
        causal = torch.ones(Q, Q, dtype=torch.bool, device=query.device).tril()
        mask = torch.cat([eviction_mask_rows(ev, G, Q, P),
                          causal[None, None].expand(B, n_q, Q, Q)], dim=-1)   # [B, n_q, Q, P+Q]
    elif mode == "full":
        # plain full-length masked forward (used only by sanity check 5)
        S = Q
        k, v = repeat_kv(key, G), repeat_kv(value, G)
        mask = torch.ones(S, S, dtype=torch.bool, device=query.device).tril()
        mask = mask[None, None].expand(B, n_q, S, S).clone()
        mask[:, :, P:, :P] &= eviction_mask_rows(ev, G, S - P, P)
    else:
        raise ValueError(mode)

    out = F.scaled_dot_product_attention(query, k, v, attn_mask=mask, scale=scaling)
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register(ATTN_NAME, kvprof_attention)
AttentionMaskInterface.register(ATTN_NAME, sdpa_mask)


# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------
def load_profiling_rows(n_per_task, first_offset=400, block=500, n_tasks=13):
    from datasets import load_dataset
    ds = load_dataset("simonjegou/ruler", data_dir="4096", split="test")
    assert len(ds) == n_tasks * block, len(ds)
    tasks = ds["task"]
    names = []
    for t in range(n_tasks):
        blk = set(tasks[block * t: block * (t + 1)])
        assert len(blk) == 1, f"task block {t} is not pure: {blk}"
        names.append(blk.pop())
    print("RULER tasks:", names)
    assert n_per_task <= 10, "rows 500t+410.. are reserved for policy training"
    idx = [block * t + first_offset + j for t in range(n_tasks) for j in range(n_per_task)]
    assert all(i % block >= 400 and i % block < 410 for i in idx)
    return [dict(ds[i], prompt_id=i) for i in idx]


def build_prompt(tok, row, style="torchtune"):
    """torchtune style (eval_ruler.py): no default system prompt, open assistant turn.
    hf style: tokenizer.apply_chat_template, which adds Qwen's default system prompt."""
    if style == "hf":
        msgs = [{"role": "user", "content": row["context"] + row["question"]}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + row["answer_prefix"]
    return ("<|im_start|>user\n" + row["context"] + row["question"] + "<|im_end|>\n"
            "<|im_start|>assistant\n" + row["answer_prefix"])


# ----------------------------------------------------------------------------------
# Core per-prompt computation
# ----------------------------------------------------------------------------------
@torch.no_grad()
def reference_pass(model, tok, prompt_ids):
    """Greedy K+1 continuation, then full-cache teacher-forced pass with capture."""
    dev = model.device
    P = prompt_ids.shape[1]
    State.mode = "off"
    gen = model.generate(prompt_ids, attention_mask=torch.ones_like(prompt_ids), max_new_tokens=K_CONT + 1, do_sample=False,
                         temperature=None, top_p=None, top_k=None,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    cont = gen[0, P:]
    g_eos = model.generation_config.eos_token_id
    eos = {tok.eos_token_id} | set(g_eos if isinstance(g_eos, (list, tuple)) else [g_eos])
    for i, t in enumerate(cont.tolist()):        # stop at EOS (keep the EOS token)
        if t in eos:
            cont = cont[: i + 1]
            break
    K = len(cont) - 1
    assert K >= 1, "continuation too short"
    inp = torch.cat([prompt_ids[0], cont[:K]])[None].to(dev)

    State.mode, State.P = "capture", P
    State.prefix_k, State.prefix_v, State.snap_scores = {}, {}, {}
    logits = model(inp, use_cache=False).logits[0, P - 1: P + K].float()
    State.mode = "off"
    # sanity 3: teacher-forced argmax reproduces the greedy continuation c_1..c_{K+1}
    tf_match = (logits.argmax(-1) == cont.to(dev)).float().mean().item()
    full_logp = logits[1:].log_softmax(-1)       # positions P..P+K-1
    return cont, K, full_logp, tf_match


@torch.no_grad()
def cont_forward(model, cont_ids, P, evict):
    """Teacher-forced pass over the K continuation tokens only. evict [B,nL,nkv,P]."""
    B = evict.shape[0]
    K = cont_ids.shape[0]
    State.mode, State.P, State.evict = "cont", P, evict
    pos = torch.arange(P, P + K, device=model.device)[None].expand(B, -1)
    ids = cont_ids[None].expand(B, -1).to(model.device)
    logits = model(input_ids=ids, position_ids=pos, use_cache=False).logits.float()
    State.mode, State.evict = "off", None
    return logits.log_softmax(-1)                # [B, K, V]


def metrics(ref_logp, logp):
    """ref_logp [K,V], logp [B,K,V] -> kl_mean, kl_max, top1_agree each [B]."""
    kl = (ref_logp.exp() * (ref_logp - logp)).sum(-1).clamp_min(0)     # [B, K]
    agree = (logp.argmax(-1) == ref_logp.argmax(-1)).float().mean(-1)
    return kl.mean(-1), kl.max(-1).values, agree


@torch.no_grad()
def kl_vs_control(model, cont_ids, P, evict):
    """Run `evict` items together with one evict-nothing CONTROL item in the same batch and
    measure KL against the control. Same batch -> same GPU kernels, so bf16 noise that depends
    on batch shape cancels exactly (r = 1.0 gives KL = 0)."""
    zero = torch.zeros_like(evict[:1])
    lp = cont_forward(model, cont_ids, P, torch.cat([zero, evict]))
    return metrics(lp[0], lp[1:])


def evict_set(scorer, scores_lh, P, r):
    """Bool [P], True = evicted. Keeps max(1, int(r*P)) prompt tokens (eval_ruler.py rule)."""
    n_keep = min(P, max(1, int(r * P)))
    ev = torch.ones(P, dtype=torch.bool, device=scores_lh.device)
    if scorer == "snapkv":
        assert n_keep >= WINDOW
        ev[scores_lh.topk(n_keep).indices] = False
    elif scorer == "streaming":
        ev[:N_SINK] = False
        ev[P - (n_keep - N_SINK):] = False
    else:
        raise ValueError(scorer)
    return ev


def profile_prompt(model, tok, row_meta, prompt_ids, scorers, ratios, batch_size, writer=None, log=print):
    cfg = model.config
    nL, nkv = cfg.num_hidden_layers, cfg.num_key_value_heads
    P = prompt_ids.shape[1]
    t0 = time.time()
    cont, K, full_logp, tf_match = reference_pass(model, tok, prompt_ids)
    dev = model.device

    jobs = []   # (scorer, r, L, h, evict_bool[P])
    for sc in scorers:
        for r in ratios:
            for L in range(nL):
                for h in range(nkv):
                    s = State.snap_scores[L][h] if sc == "snapkv" else State.snap_scores[L][0]
                    jobs.append((sc, r, L, h, evict_set(sc, s, P, r)))
    rows = []
    step = max(1, batch_size - 1)          # one slot per batch goes to the control item
    for i in range(0, len(jobs), step):
        chunk = jobs[i: i + step]
        ev = torch.zeros(len(chunk), nL, nkv, P, dtype=torch.bool, device=dev)
        for b, (_, _, L, h, e) in enumerate(chunk):
            ev[b, L, h] = e
        klm, klx, agr = kl_vs_control(model, cont[:K], P, ev)
        for b, (sc, r, L, h, e) in enumerate(chunk):
            rows.append(dict(prompt_id=row_meta["prompt_id"], task=row_meta["task"], scorer=sc,
                             keep_ratio=r, layer=L, kv_head=h, kl_mean=f"{klm[b].item():.6g}",
                             kl_max=f"{klx[b].item():.6g}", top1_agree=f"{agr[b].item():.4f}",
                             prompt_len=P, n_evicted=int(e.sum().item()), n_pos=K))
    if writer is not None:
        writer.writerows(rows)
    log(f"[prompt {row_meta['prompt_id']} {row_meta['task']}] P={P} K={K} tf_match={tf_match:.3f}  "
        f"{len(jobs)} runs in {time.time() - t0:.1f}s")
    return rows, dict(P=P, K=K, cont=cont)


# ----------------------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------------------
@torch.no_grad()
def sanity_checks(model, tok, prompts, n_prompts=2, seed=0):
    rng = random.Random(seed)
    cfg = model.config
    nL, nkv = cfg.num_hidden_layers, cfg.num_key_value_heads
    dev = model.device
    ok = True
    for prompt_ids in prompts[:n_prompts]:
        P = prompt_ids.shape[1]
        cont, K, full_logp, tf_match = reference_pass(model, tok, prompt_ids)
        heads = [(rng.randrange(nL), rng.randrange(nkv)) for _ in range(4)]

        # 0. noise floor of the OLD approach (reference in a separate batch-1 pass), for info only
        zero = torch.zeros(1, nL, nkv, P, dtype=torch.bool, device=dev)
        ref1 = cont_forward(model, cont[:K], P, zero)[0]
        noise_old = metrics(ref1, cont_forward(model, cont[:K], P, zero.expand(4, -1, -1, -1)))[0].max().item()

        # 1. r = 1.0 -> nothing evicted -> KL exactly 0 against the in-batch control
        ev = torch.zeros(len(heads), nL, nkv, P, dtype=torch.bool, device=dev)
        for b, (L, h) in enumerate(heads):
            ev[b, L, h] = evict_set("snapkv", State.snap_scores[L][h], P, 1.0)
        kl1 = kl_vs_control(model, cont[:K], P, ev)[0]
        c1 = bool((kl1 < 1e-7).all())
        print(f"[check1] r=1.0 heads={heads} n_evicted={int(ev.sum())} kl_mean={[f'{x:.1e}' for x in kl1.tolist()]} "
              f"(old separate-reference noise was {noise_old:.1e}) -> {'PASS' if c1 else 'FAIL'}")

        # 2. mask applied at r=0.1
        L, h = heads[0]
        e = evict_set("snapkv", State.snap_scores[L][h], P, 0.1)
        ev = torch.zeros(1, nL, nkv, P, dtype=torch.bool, device=dev); ev[0, L, h] = e
        kl2 = kl_vs_control(model, cont[:K], P, ev)[0]
        c2 = kl2.item() > 0 and not e[P - WINDOW:].any()
        print(f"[check2] head ({L},{h}) r=0.1 blocked keys={int(e.sum())} (0.9*P={0.9*P:.0f}, P-64={P-64}) "
              f"window evicted={int(e[P-WINDOW:].sum())} kl_mean={kl2.item():.3e} -> {'PASS' if c2 else 'FAIL'}")

        # 3. reference continuation == model.generate greedy (teacher-forced argmax agreement)
        c3 = tf_match >= 0.95
        print(f"[check3] teacher-forced argmax matches greedy continuation on {tf_match:.3f} of {K+1} tokens "
              f"-> {'PASS' if c3 else 'FAIL'}")

        # 4. all heads at r=0.1 >> any single head
        ev_all = torch.zeros(1, nL, nkv, P, dtype=torch.bool, device=dev)
        for L_ in range(nL):
            for h_ in range(nkv):
                ev_all[0, L_, h_] = evict_set("snapkv", State.snap_scores[L_][h_], P, 0.1)
        kl_all = kl_vs_control(model, cont[:K], P, ev_all)[0].item()
        singles = torch.zeros(nL * nkv, nL, nkv, P, dtype=torch.bool, device=dev)
        for L_ in range(nL):
            for h_ in range(nkv):
                singles[L_ * nkv + h_, L_, h_] = ev_all[0, L_, h_]
        kl_single = torch.cat([kl_vs_control(model, cont[:K], P, singles[i:i + 47])[0]
                               for i in range(0, len(singles), 47)])
        c4 = kl_all > kl_single.max().item()
        q = kl_single.quantile(torch.tensor([0.1, 0.5, 0.9], device=dev)).tolist()
        print(f"[check4] all-heads r=0.1 KL={kl_all:.3e} vs max single-head KL={kl_single.max():.3e} "
              f"(ratio {kl_all / max(kl_single.max().item(), 1e-12):.1f}x); single-head KL p10/p50/p90="
              f"{q[0]:.1e}/{q[1]:.1e}/{q[2]:.1e} -> {'PASS' if c4 else 'FAIL'}")

        # 5. cached continuation path == plain full-length masked forward (both vs their own control),
        #    on the most sensitive single head so there is real signal to compare
        top = int(kl_single.argmax())
        ev5 = singles[top:top + 1]
        inp = torch.cat([prompt_ids[0], cont[:K]])[None].to(dev).expand(2, -1)
        State.mode, State.P, State.evict = "full", P, torch.cat([torch.zeros_like(ev5), ev5])
        lp = model(inp, use_cache=False, logits_to_keep=K).logits.float().log_softmax(-1)
        State.mode, State.evict = "off", None
        kl_full = metrics(lp[0], lp[1:])[0].item()
        kl_cached = kl_vs_control(model, cont[:K], P, ev5)[0].item()
        rel = abs(kl_full - kl_cached) / max(kl_full, kl_cached, 1e-12)
        c5 = rel < 0.25
        print(f"[check5] head ({top // nkv},{top % nkv}) r=0.1: KL full-length masked pass={kl_full:.3e} vs "
              f"cached path={kl_cached:.3e} (rel diff {rel:.1%}) -> {'PASS' if c5 else 'FAIL'}")
        ok &= c1 and c2 and c3 and c4 and c5
    print("SANITY:", "ALL PASS" if ok else "SOME CHECKS FAILED")
    return ok


# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out_dir", default="results/profiling")
    ap.add_argument("--n_per_task", type=int, default=3, help="3 -> 39 prompts, 5 -> 65 prompts")
    ap.add_argument("--scorers", default="snapkv", help="comma list: snapkv,streaming")
    ap.add_argument("--ratios", default=",".join(map(str, RATIOS)))
    ap.add_argument("--batch_size", type=int, default=48, help="(head, ratio) runs per forward")
    ap.add_argument("--sanity_only", action="store_true")
    ap.add_argument("--skip_sanity", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--prompt_style", default="torchtune", choices=["torchtune", "hf"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype),
                                                 attn_implementation=ATTN_NAME, device_map="cuda")
    model.eval()
    assert not getattr(model.config, "use_sliding_window", False)

    rows = load_profiling_rows(args.n_per_task)
    encoded = [tok(build_prompt(tok, r, args.prompt_style), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
               for r in rows]
    print("prompt start:", repr(tok.decode(encoded[0][0, :12])), "... end:", repr(tok.decode(encoded[0][0, -12:])))
    print(f"{len(rows)} prompts, lengths {min(e.shape[1] for e in encoded)}..{max(e.shape[1] for e in encoded)}")

    if not args.skip_sanity:
        ok = sanity_checks(model, tok, [encoded[0], encoded[len(encoded) // 2], encoded[-1]], n_prompts=3)
        if args.sanity_only:
            return
        if not ok:
            raise SystemExit("sanity checks failed; not running the full profile")

    scorers = args.scorers.split(",")
    ratios = [float(x) for x in args.ratios.split(",")]
    csv_path = os.path.join(args.out_dir, "head_kl.csv")
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                done.add((int(r["prompt_id"]), r["scorer"]))
    t0 = time.time()
    meta_path = os.path.join(args.out_dir, "run_meta.json")
    for row, ids in zip(rows, encoded):
        todo = [s for s in scorers if (row["prompt_id"], s) not in done]
        if not todo:
            continue
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                w.writeheader()
            profile_prompt(model, tok, row, ids, todo, ratios, args.batch_size, writer=w)
    meta = dict(gpu=torch.cuda.get_device_name(0), runtime_s=round(time.time() - t0, 1),
                n_prompts=len(rows), scorers=scorers, ratios=ratios, model=args.model,
                torch=torch.__version__, transformers=__import__("transformers").__version__,
                note="runtime covers only prompts processed in this invocation")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
