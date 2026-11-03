#!/usr/bin/env python
"""Offline -> online RL for per-input KV-cache budget allocation (EE599 concepts, end to end).

Problem (contextual bandit, one decision per prompt)
  state  s : per-head features of the prompt (rl_alloc.head_features) for all 112 KV heads
  action a : one budget level per head (rl_alloc.LEVELS), projected onto the memory budget
  reward r : -log KL(full cache || compressed) over the answer tokens (exact, "verifiable")

  collect       logged dataset from a KNOWN behavior policy (exact log pi_b)      [Wk 8]
  train_critic  bootstrapped ENSEMBLE reward model Q(s,a); spread = uncertainty    [Wk 3, 5]
  train_policy  bc [Wk 1] | awr [Wk 3, 8] | dpo [Wk 11] |
                mbpo = model-based PG, pessimism + KL to reference, group baseline [Wk 2,5,8,11,12]
  evaluate      true KL, critic gap (reward hacking), OPE: IS/SNIS/DM/DR vs truth  [Wk 8, 11]
  online        optimistic UCB data collection + 50/50 offline/online AWR updates  [Wk 6, 8]
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_alloc as ra  # noqa: E402

LV = np.asarray(ra.LEVELS)
NL = len(ra.LEVELS)
H = 112
TAUS = (0.5, 1.0, 2.0)


def behavior_base(static_lv):
    base = -torch.abs(torch.arange(NL)[None].float() - torch.as_tensor(static_lv)[:, None].float())
    base[torch.arange(H), torch.as_tensor(static_lv)] += 1.5
    return base


def behavior_sample(base, m, gen=None):
    acts = []
    for _ in range(m):
        tau = TAUS[int(torch.randint(len(TAUS), (1,), generator=gen))]
        acts.append(torch.distributions.Categorical(logits=base / tau).sample())
    acts = torch.stack(acts)
    return acts, behavior_logp(base, acts)


def behavior_logp(base, acts):
    comps = []
    for tau in TAUS:
        lp = torch.log_softmax(base / tau, -1)
        comps.append(lp.gather(1, acts.T).sum(0))
    return torch.logsumexp(torch.stack(comps), 0) - math.log(len(TAUS))


def collect_prompt(model, ep, static_lv, budget, prior, target, m, batch_size):
    base = behavior_base(static_lv)
    acts, logpb = behavior_sample(base, m)
    proj = np.stack([ra.project(a, budget, prior) for a in acts.numpy()])
    cand = np.concatenate([LV[proj], np.full((1, H), target), LV[ra.project(static_lv, budget, prior)][None]])
    kl = ep.kl(model, cand, batch_size)
    r = -np.log(kl + 1e-6)
    return dict(pid=int(ep.row["prompt_id"]), task=ep.row.get("task", ""), P=ep.P, K=ep.K,
                feats=ep.feats.float().cpu().half(), acts=acts.to(torch.int8), proj=torch.as_tensor(proj, dtype=torch.int8),
                logpb=logpb.float(), r=torch.as_tensor(r[:m], dtype=torch.float32),
                r_uniform=float(r[m]), r_static=float(r[m + 1]))


def load_records(run_dir, split):
    recs = []
    for f in sorted(Path(run_dir, "data").glob(f"{split}_shard*.pt")):
        recs += torch.load(f, weights_only=False)
    assert recs, f"no {split} data in {run_dir}/data (run --step collect first)"
    return recs


class Critic(nn.Module):
    def __init__(self, n_feat=ra.N_FEAT, hid=128, emb=16):
        super().__init__()
        self.head_emb, self.lvl_emb = nn.Embedding(H, emb), nn.Embedding(NL, emb)
        self.phi = nn.Sequential(nn.Linear(n_feat + 2 * emb, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, feats, a):
        ids = torch.arange(H, device=feats.device)
        x = torch.cat([feats, self.head_emb(ids).expand(feats.shape[0], -1, -1), self.lvl_emb(a.long())], -1)
        return self.rho(self.phi(x).mean(1)).squeeze(-1)


class Ensemble(nn.Module):
    def __init__(self, n=5):
        super().__init__()
        self.members = nn.ModuleList([Critic() for _ in range(n)])
        self.register_buffer("mu", torch.zeros(())); self.register_buffer("sd", torch.ones(()))

    def forward(self, feats, a):
        return torch.stack([m(feats, a) for m in self.members]) * self.sd + self.mu


def flat_rows(recs):
    feats = torch.stack([r["feats"].float() for r in recs])
    pi, act, rew, adv = [], [], [], []
    for i, r in enumerate(recs):
        m = len(r["r"])
        pi += [i] * m
        act.append(r["proj"].long()); rew.append(r["r"])
        adv.append(r["r"] - r["r"].mean())
    return feats, torch.tensor(pi), torch.cat(act), torch.cat(rew), torch.cat(adv)


def fit_ensemble(ens, feats, pi, act, rew, epochs, lr, dev, log, bs=512, bootstrap=True):
    feats, pi, act, rew = feats.to(dev), pi.to(dev), act.to(dev), rew.to(dev)
    ens.mu.fill_(rew.mean()); ens.sd.fill_(rew.std() + 1e-6)
    y = (rew - ens.mu) / ens.sd
    for k, m in enumerate(ens.members):
        idx = torch.randint(len(y), (len(y),), device=dev) if bootstrap else torch.arange(len(y), device=dev)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        for ep in range(epochs):
            perm = idx[torch.randperm(len(idx), device=dev)]
            tot = 0.0
            for j in range(0, len(perm), bs):
                b = perm[j:j + bs]
                loss = F.mse_loss(m(feats[pi[b]], act[b]), y[b])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(b)
        log(f"[critic] member {k}: train MSE (normalized) {tot / len(perm):.4f}")


@torch.no_grad()
def critic_report(ens, recs, dev):
    feats, pi, act, rew, _ = flat_rows(recs)
    pred = []
    for j in range(0, len(rew), 2048):
        pred.append(ens(feats[pi[j:j + 2048]].to(dev), act[j:j + 2048].to(dev)).mean(0).cpu())
    pred = torch.cat(pred)
    pear = torch.corrcoef(torch.stack([pred, rew]))[0, 1].item()
    sp, o = [], 0
    for r in recs:
        m = len(r["r"]); p, t = pred[o:o + m], r["r"]; o += m
        rp, rt = p.argsort().argsort().float(), t.argsort().argsort().float()
        sp.append(torch.corrcoef(torch.stack([rp, rt]))[0, 1].item())
    return dict(mse=float(((pred - rew) ** 2).mean()), pearson=pear, within_prompt_spearman=float(np.nanmean(sp)))


def new_policy(static_lv, dev):
    return ra.Policy(H, ra.N_FEAT, init_levels=static_lv).to(dev)


def batched_logits(policy, feats):
    emb = policy.emb.weight[None].expand(feats.shape[0], -1, -1)
    return policy.bias + policy.mlp(torch.cat([feats, emb], -1))


def policy_logp(policy, feats, a):
    lp = torch.log_softmax(batched_logits(policy, feats), -1)
    return lp.gather(-1, a.long().unsqueeze(-1)).squeeze(-1).sum(-1)


def train_offline(args, recs, static_lv, budget, prior, dev, log, ens=None):
    feats, pi, act, rew, adv = flat_rows(recs)
    feats, pi, act, rew, adv = feats.to(dev), pi.to(dev), act.to(dev), rew.to(dev), adv.to(dev)
    policy = new_policy(static_lv, dev)
    ref = new_policy(static_lv, dev).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    bs = args.pbatch
    if args.algo == "bc":
        keep, o = [], 0
        for r in recs:
            m = len(r["r"]); k = max(1, int(math.ceil(args.bc_top * m)))
            keep += (o + torch.topk(r["r"], k).indices).tolist(); o += m
        keep = torch.tensor(keep, device=dev)
    if args.algo == "dpo":
        W, L, o = [], [], 0
        for r in recs:
            m = len(r["r"]); order = r["r"].argsort(descending=True)
            for _ in range(args.dpo_pairs):
                i, j = sorted(random.sample(range(m), 2))
                W.append(o + order[i].item()); L.append(o + order[j].item())
            o += m
        W, L = torch.tensor(W, device=dev), torch.tensor(L, device=dev)
    for step in range(1, args.psteps + 1):
        if args.algo == "bc":
            b = keep[torch.randint(len(keep), (bs,), device=dev)]
            loss = -policy_logp(policy, feats[pi[b]], act[b]).mean() / H
        elif args.algo == "awr":
            b = torch.randint(len(rew), (bs,), device=dev)
            w = torch.exp(adv[b] / args.awr_beta).clamp(max=args.awr_wmax)
            loss = -(w * policy_logp(policy, feats[pi[b]], act[b])).mean() / H
        elif args.algo == "dpo":
            b = torch.randint(len(W), (bs,), device=dev)
            f = feats[pi[W[b]]]
            lw = policy_logp(policy, f, act[W[b]]) - policy_logp(ref, f, act[W[b]])
            ll = policy_logp(policy, f, act[L[b]]) - policy_logp(ref, f, act[L[b]])
            loss = -F.logsigmoid(args.dpo_beta * (lw - ll)).mean()
        elif args.algo == "mbpo":
            assert ens is not None, "--algo mbpo needs a trained critic"
            pb = torch.randint(len(recs), (args.mb_prompts,), device=dev)
            f = torch.stack([recs[i]["feats"].float() for i in pb.tolist()]).to(dev)
            logits = batched_logits(policy, f)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample((args.group,))
            logp = dist.log_prob(a).sum(-1)
            pa = np.stack([[ra.project(a[g, i].cpu().numpy(), budget, prior) for i in range(a.shape[1])]
                           for g in range(args.group)])
            with torch.no_grad():
                q = ens(f.repeat(args.group, 1, 1), torch.as_tensor(pa, device=dev).view(-1, H))
                r_hat = (q.mean(0) - args.kappa * q.std(0)).view(args.group, -1)
            base = (r_hat.sum(0, keepdim=True) - r_hat) / (args.group - 1)
            A = (r_hat - base) / ((r_hat - base).std(0, keepdim=True) + 1e-6)
            kl_ref = (dist.probs * (torch.log_softmax(logits, -1)
                                    - torch.log_softmax(batched_logits(ref, f), -1))).sum(-1).sum(-1)
            loss = -(A.detach() * logp).mean() + args.kl_coef * kl_ref.mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % max(1, args.psteps // 10) == 0:
            log(f"[{args.algo}] step {step}/{args.psteps} loss {loss.item():.4f}")
    return policy


@torch.no_grad()
def evaluate(args, model, tok, rows, policies, static_lv, budget, prior, dev, log, ens=None, val_recs=None):
    by_pid = {r["pid"]: r for r in (val_recs or [])}
    res = {k: [] for k in ["uniform", "static", *policies]}
    pred = {k: [] for k in res}
    ope = {k: {e: [] for e in ["truth", "IS", "IS_clip", "SNIS_num", "SNIS_den", "DM", "DR"]} for k in policies}
    for row in rows:
        ep = ra.make_episode(model, tok, row, args)
        if ep is None:
            continue
        names, cand = ["uniform", "static"], [np.full(H, args.target), LV[ra.project(static_lv, budget, prior)]]
        sampled = {}
        for k, pol in policies.items():
            lg = pol(ep.feats.to(dev).float())
            names.append(k); cand.append(LV[ra.project(lg.argmax(-1).cpu().numpy(), budget, prior)])
            if args.ope_samples:
                a = torch.distributions.Categorical(logits=lg).sample((args.ope_samples,))
                pa = np.stack([ra.project(x, budget, prior) for x in a.cpu().numpy()])
                sampled[k] = (lg, pa)
                cand += list(LV[pa])
        kl = ep.kl(model, np.stack(cand), args.batch_size)
        r = -np.log(kl + 1e-6)
        for n, v in zip(names, r[:len(names)]):
            res[n].append(float(v))
        if ens is not None:
            f = ep.feats.to(dev).float()[None]
            idx = [int(np.searchsorted(LV, x + 1e-9) - 1) for x in np.stack(cand[:len(names)]).ravel()]
            q = ens(f.repeat(len(names), 1, 1), torch.tensor(idx, device=dev).view(len(names), H)).mean(0)
            for n, v in zip(names, q.cpu().numpy()):
                pred[n].append(float(v))
        o = len(names)
        for k, (lg, pa) in sampled.items():
            truth = r[o:o + args.ope_samples]; o += args.ope_samples
            ope[k]["truth"].append(float(truth.mean()))
            rec = by_pid.get(int(row["prompt_id"]))
            if rec is None or ens is None:
                continue
            lp_pi = torch.log_softmax(lg, -1).gather(-1, rec["acts"].long().to(dev).T).sum(0).cpu()
            w = (lp_pi - rec["logpb"]).clamp(max=50).exp()
            f = ep.feats.to(dev).float()[None]
            q_log = ens(f.repeat(len(rec["r"]), 1, 1), rec["proj"].long().to(dev)).mean(0).cpu()
            q_pi = ens(f.repeat(len(pa), 1, 1), torch.as_tensor(pa, device=dev)).mean(0).mean().item()
            wc = w.clamp(max=args.is_clip)
            ope[k]["IS"].append(float((w * rec["r"]).mean()))
            ope[k]["IS_clip"].append(float((wc * rec["r"]).mean()))
            ope[k]["SNIS_num"].append(float((w * rec["r"]).sum())); ope[k]["SNIS_den"].append(float(w.sum()))
            ope[k]["DM"].append(q_pi)
            ope[k]["DR"].append(q_pi + float((wc * (rec["r"] - q_log)).mean()))
    out = {}
    for n in res:
        out[n] = dict(true_r=float(np.mean(res[n])), mean_kl=float(np.mean(np.exp(-np.array(res[n])))), n=len(res[n]))
        if pred[n]:
            out[n]["critic_r"] = float(np.mean(pred[n]))
            out[n]["hacking_gap"] = out[n]["critic_r"] - out[n]["true_r"]
    for k, d in ope.items():
        if d["truth"]:
            e = dict(truth=float(np.mean(d["truth"])))
            if d["DM"]:
                e.update(IS=float(np.mean(d["IS"])), IS_clip=float(np.mean(d["IS_clip"])),
                         SNIS=float(np.sum(d["SNIS_num"]) / max(np.sum(d["SNIS_den"]), 1e-12)),
                         DM=float(np.mean(d["DM"])), DR=float(np.mean(d["DR"])))
            out[k]["ope"] = e
    for n, d in out.items():
        log(f"[eval] {n:<10} true -logKL {d['true_r']:.3f}  meanKL {d['mean_kl']:.4g}"
            + (f"  critic {d['critic_r']:.3f} gap {d['hacking_gap']:+.3f}" if "critic_r" in d else "")
            + (f"  OPE {json.dumps({k: round(v, 3) for k, v in d['ope'].items()})}" if "ope" in d else ""))
    return out


def online(args, model, tok, train_rows, val_rows, policy, ens, recs, static_lv, budget, prior, dev, log, run):
    feats_off, pi_off, act_off, rew_off, adv_off = [t.to(dev) for t in flat_rows(recs)]
    on = dict(feats=[], act=[], rew=[], adv=[], pi=[])
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    best = -1e9
    for step, row in enumerate(train_rows, 1):
        ep = ra.make_episode(model, tok, row, args)
        if ep is None:
            continue
        f = ep.feats.to(dev).float()
        with torch.no_grad():
            lg = policy(f)
            a = torch.distributions.Categorical(logits=lg).sample((args.n_cand,))
            pa = np.unique(np.stack([ra.project(x, budget, prior) for x in a.cpu().numpy()]), axis=0)
            q = ens(f[None].repeat(len(pa), 1, 1), torch.as_tensor(pa, device=dev))
            ucb = (q.mean(0) + args.kappa_ucb * q.std(0)).cpu().numpy()
            pick = pa[np.argsort(-ucb)[:args.group]]
        r = -np.log(ep.kl(model, LV[pick], args.batch_size) + 1e-6)
        on["feats"].append(f.cpu()); k = len(on["feats"]) - 1
        for x, v in zip(pick, r):
            on["pi"].append(k); on["act"].append(torch.as_tensor(x)); on["rew"].append(float(v))
            on["adv"].append(float(v - r.mean()))
        F_on = torch.stack(on["feats"]).to(dev); A_on = torch.stack(on["act"]).to(dev)
        D_on = torch.tensor(on["adv"], device=dev); P_on = torch.tensor(on["pi"], device=dev)
        for _ in range(args.updates_per_step):
            h = args.pbatch // 2
            bo = torch.randint(len(rew_off), (h,), device=dev); bn = torch.randint(len(D_on), (h,), device=dev)
            fb = torch.cat([feats_off[pi_off[bo]], F_on[P_on[bn]]]); ab = torch.cat([act_off[bo], A_on[bn]])
            w = torch.exp(torch.cat([adv_off[bo], D_on[bn]]) / args.awr_beta).clamp(max=args.awr_wmax)
            loss = -(w * policy_logp(policy, fb, ab)).mean() / H
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        log(f"[online] step {step} {row.get('task', ''):<16} picked r {r.mean():.3f} (best {r.max():.3f}) "
            f"ucb-top {ucb.max():.3f}")
        if step % args.refit_every == 0:
            fa = torch.cat([feats_off.cpu(), F_on.cpu()])
            pa_ = torch.cat([pi_off.cpu(), P_on.cpu() + len(feats_off)])
            fit_ensemble(ens, fa, pa_, torch.cat([act_off.cpu(), A_on.cpu()]),
                         torch.cat([rew_off.cpu(), torch.tensor(on["rew"])]), 2, 3e-4, dev, log)
        if step % args.eval_every == 0:
            ev = evaluate(args, model, tok, val_rows, {"online": policy}, static_lv, budget, prior, dev, log, ens)
            json.dump(ev, open(run / f"eval_online_step{step}.json", "w"), indent=1)
            torch.save(policy.state_dict(), run / "policy_online_last.pt")
            if ev["online"]["true_r"] > best:
                best = ev["online"]["true_r"]; torch.save(policy.state_dict(), run / "policy_online.pt")
                log(f"[online] new best true -logKL {best:.3f}")
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["collect", "train_critic", "train_policy", "evaluate", "online"])
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--init_alloc", required=True)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--model", default=os.path.expandvars("/scratch2/$USER/kvp/models/Qwen2.5-7B-Instruct"))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--min_k", type=int, default=3)
    ap.add_argument("--prompt_style", default="torchtune", choices=["torchtune", "hf"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--n_per_task", type=int, default=90)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--cepochs", type=int, default=15)
    ap.add_argument("--algo", default="awr", choices=["bc", "awr", "dpo", "mbpo"])
    ap.add_argument("--psteps", type=int, default=3000)
    ap.add_argument("--pbatch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bc_top", type=float, default=0.1)
    ap.add_argument("--awr_beta", type=float, default=0.5)
    ap.add_argument("--awr_wmax", type=float, default=20.0)
    ap.add_argument("--dpo_beta", type=float, default=0.1)
    ap.add_argument("--dpo_pairs", type=int, default=32)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--kl_coef", type=float, default=0.05)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--mb_prompts", type=int, default=16)
    ap.add_argument("--policies", default="")
    ap.add_argument("--n_val_per_task", type=int, default=5)
    ap.add_argument("--ope_samples", type=int, default=8)
    ap.add_argument("--is_clip", type=float, default=100.0)
    ap.add_argument("--init_policy", default="awr")
    ap.add_argument("--n_cand", type=int, default=32)
    ap.add_argument("--kappa_ucb", type=float, default=1.0)
    ap.add_argument("--updates_per_step", type=int, default=4)
    ap.add_argument("--refit_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=150)
    ap.add_argument("--online_prompts", type=int, default=400)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    run = Path(args.run_dir or f"rl_runs/offline_t{args.target}"); (run / "data").mkdir(parents=True, exist_ok=True)
    logf = open(run / f"{args.step}{'_' + args.algo if args.step == 'train_policy' else ''}.log", "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    log(f"=== {args.step} {time.strftime('%Y-%m-%d %H:%M')} args={vars(args)}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    budget = args.target * H
    static_lv = ra.alloc_to_levels(args.init_alloc, H)
    prior = LV[static_lv] + 1e-3 * np.arange(H)[::-1] / H

    def load_ens():
        ens = Ensemble(args.ensemble).to(dev)
        ens.load_state_dict(torch.load(run / "critic.pt", map_location=dev))
        return ens.eval()

    if args.step == "train_critic":
        recs, vrecs = load_records(run, "train"), load_records(run, "val")
        ens = Ensemble(args.ensemble).to(dev)
        fit_ensemble(ens, *flat_rows(recs)[:4], args.cepochs, 1e-3, dev, log)
        ens.eval()
        rep = dict(train=critic_report(ens, recs, dev), val=critic_report(ens, vrecs, dev))
        log(f"[critic] {json.dumps(rep)}")
        json.dump(rep, open(run / "critic_report.json", "w"), indent=1)
        torch.save(ens.state_dict(), run / "critic.pt")
        return

    if args.step == "train_policy":
        recs = load_records(run, "train")
        ens = load_ens() if args.algo == "mbpo" else None
        pol = train_offline(args, recs, static_lv, budget, prior, dev, log, ens)
        name = args.algo + (f"_k{args.kappa}" if args.algo == "mbpo" else "")
        torch.save(pol.state_dict(), run / f"policy_{name}.pt")
        log(f"[saved] {run}/policy_{name}.pt")
        return

    model, tok = ra.load_model(args.model)
    ds = ra.load_ruler()

    if args.step == "collect":
        rows = ra.rows_by_offset(ds, 410 if args.split == "train" else 405,
                                 args.n_per_task if args.split == "train" else args.n_val_per_task)
        rows = rows[args.shard::args.n_shards]
        out, t0 = [], time.time()
        for n, row in enumerate(rows):
            ep = ra.make_episode(model, tok, row, args)
            if ep is None:
                continue
            out.append(collect_prompt(model, ep, static_lv, budget, prior, args.target, args.m, args.batch_size))
            if n % 10 == 0:
                rr = out[-1]
                log(f"[collect] {n + 1}/{len(rows)} {row['task']:<16} logged r mean {rr['r'].mean():.3f} "
                    f"max {rr['r'].max():.3f} | uniform {rr['r_uniform']:.3f} static {rr['r_static']:.3f} "
                    f"({(time.time() - t0) / (n + 1):.1f}s/prompt)")
            if n % 50 == 0:
                torch.save(out, run / "data" / f"{args.split}_shard{args.shard}.pt")
        torch.save(out, run / "data" / f"{args.split}_shard{args.shard}.pt")
        log(f"[collect] saved {len(out)} prompts -> {run}/data/{args.split}_shard{args.shard}.pt")
        return

    val_rows = ra.rows_by_offset(ds, 405, args.n_val_per_task)
    ens = load_ens() if (run / "critic.pt").exists() else None

    if args.step == "evaluate":
        pols = {}
        for name in [p for p in args.policies.replace("+", ",").split(",") if p]:
            pol = new_policy(static_lv, dev)
            pol.load_state_dict(torch.load(run / f"policy_{name}.pt", map_location=dev)); pol.eval()
            pols[name] = pol
        vrecs = load_records(run, "val") if list(Path(run, "data").glob("val_shard*.pt")) else None
        ev = evaluate(args, model, tok, val_rows, pols, static_lv, budget, prior, dev, log, ens, vrecs)
        json.dump(ev, open(run / f"eval_{args.policies.replace(',', '+') or 'baselines'}.json", "w"), indent=1)
        return

    if args.step == "online":
        assert ens is not None, "train the critic first"
        policy = new_policy(static_lv, dev)
        policy.load_state_dict(torch.load(run / f"policy_{args.init_policy}.pt", map_location=dev))
        train_rows = ra.rows_by_offset(ds, 410, args.n_per_task)
        random.shuffle(train_rows)
        online(args, model, tok, train_rows[:args.online_prompts], val_rows, policy, ens.train(),
               load_records(run, "train"), static_lv, budget, prior, dev, log, run)


if __name__ == "__main__":
    main()
