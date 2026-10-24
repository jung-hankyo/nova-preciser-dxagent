"""Hypothesis-recall experiment: deterministic card-based candidate sources vs LLM-only generation.

Problem: a disease that never enters the differential cannot be diagnosed. Earlier runs showed top-3 on
unknown-disease cases capped at exactly the assumed generator recall.

Generator model (baseline, all arms)
    LLM proposal at the start: the true disease is proposed with probability r (a PERSISTENT per-case
    blind spot: re-asking the same model does not help); every other disease of the presenting family is
    proposed with probability 0.6. Critic: when the agent is about to commit and the truth is missing,
    one rescue with probability 0.3.

Added sources (deterministic, from the knowledge base; the arms differ only in these)
    reverse   after each belief update, any known disease not in the differential whose posterior would be
              >= 5% if added (computed exactly: re-run the update with it included) is added (max 2 per turn)
    mimic     the confusable sibling of any leader with posterior >= 0.30 is added if it would reach >= 2%
    both      reverse + mimic

Acceptance (as before): a gain in accuracy or calibration with no metric significantly worse (paired
95% CI). Metrics: top-1, top-3, Brier, ECE, log loss, can't-miss missed, turns, cost; plus the share of
cases whose truth is in the differential at commit (recall), reported but not part of the rule.

Run: python experiments/hyp_recall.py --worlds 3 --cases 1500 --recall 0.6 0.75 0.9
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import world_bench as WB  # noqa: E402
from dxagent.config import get_settings  # noqa: E402
from obs_model import HIGHER_IS_BETTER, ece, ece_boot, verdict  # noqa: E402
from lr_vs_freq import boot_diff, mcnemar_exact, wilson  # noqa: E402

S = get_settings()
P_OTHER = 0.6
CRITIC_RESCUE = 0.3
REVERSE_MIN = 0.05
MIMIC_LEADER = 0.30
MIMIC_MIN = 0.02


def u(case_idx: int, key: str) -> float:
    return random.Random(zlib.crc32(f"{case_idx}:{key}".encode())).random()


def prior_bucket(p: float) -> float:
    return min(WB.PRIOR_BUCKETS, key=lambda b: abs(math.log(b) - math.log(p)))


def sibling(did: str) -> str | None:
    fam, j = did.split("_d")
    j = int(j)
    return f"{fam}_d{j ^ 1}"


REVERSE_PRIOR = {"value": None}  # None: true prior bucket; a float: fixed prior for index-added diseases (as in the package)
INDEX_ADDED: set = set()


def build(fam, known, hyp_ids, dz_by_id):
    cards = {d: known[d] for d in hyp_ids}
    pri = {d: (REVERSE_PRIOR["value"] if (REVERSE_PRIOR["value"] and d in INDEX_ADDED) else prior_bucket(dz_by_id[d].prior)) for d in hyp_ids}
    K = WB.agent_world(fam, cards, pri, 0, None)
    return K, WB.ConfRep(K, cards)


def run(arm: str, fam, known, case, dz_by_id, r: float, full_list: bool = False) -> dict:
    fam_ids = [d.id for d in fam.diseases]
    if full_list:
        hyp = list(fam_ids)
    else:
        hyp = [d for d in fam_ids if (u(case.idx, "truth") < r if d == case.disease else u(case.idx, d) < P_OTHER)]
        if not hyp:
            hyp = [d for d in fam_ids if d != case.disease][:1] or [fam_ids[0]]
    rescued = False
    INDEX_ADDED.clear()
    added = {"reverse": 0, "mimic": 0}
    K, R = build(fam, known, hyp, dz_by_id)
    obs, asked, done = [], set(), set()
    turn, cost, stall = 0, 0.0, 0
    for t_i, codes in case.intake:
        turn = t_i
        for f in codes:
            obs.append(("f", f, case.frep[f], WB.REPORT_CONF[fam.src[f]]))
            asked.add(f)
        if t_i == 3:
            cost += 5.0
    reason = ""
    while True:
        # ---- added hypothesis sources (deterministic)
        if arm in ("reverse", "mimic", "both"):
            changed = True
            while changed:
                changed = False
                b, _ = WB.posterior(K, R, obs)
                cands = []
                if arm in ("reverse", "both"):
                    for d in fam_ids:
                        if d in hyp:
                            continue
                        INDEX_ADDED.add(d)
                        K2, R2 = build(fam, known, hyp + [d], dz_by_id)
                        if not any(R2.llr(d, o) for o in obs):  # evidence-free cards never enter (as in the package)
                            INDEX_ADDED.discard(d)
                            continue
                        p = WB.posterior(K2, R2, obs)[0][d]
                        INDEX_ADDED.discard(d)
                        if p >= REVERSE_MIN:
                            cands.append((p, d, "reverse"))
                    cands = sorted(cands, reverse=True)[:2]
                if arm in ("mimic", "both"):
                    for lead in [d for d in hyp if b[d] >= MIMIC_LEADER]:
                        s = sibling(lead)
                        if s in fam_ids and s not in hyp and all(s != c[1] for c in cands):
                            K2, R2 = build(fam, known, hyp + [s], dz_by_id)
                            p = WB.posterior(K2, R2, obs)[0][s]
                            if p >= MIMIC_MIN:
                                cands.append((p, s, "mimic"))
                for _, d, src in cands:
                    if src == "reverse":
                        INDEX_ADDED.add(d)
                    hyp.append(d)
                    added[src] += 1
                    changed = True
                if changed:
                    K, R = build(fam, known, hyp, dz_by_id)
        b, other = WB.posterior(K, R, obs)
        ids = sorted(b, key=b.get, reverse=True)
        probs = [b[d] for d in ids]
        leader = probs[0]
        if turn >= S.max_turns - 1:
            reason = "budget"
            break
        acts = []
        for f in K.bg:
            if f in asked or not any(f in K.p[d] for d in ids):
                continue
            acts.append(("f", [f], K.src[f]))
        for t in K.outcomes:
            if t in done or not any(t in K.dist[d] for d in ids):
                continue
            gate = {"advanced_imaging": 0.05, "procedure": 0.25}.get(K.ttype[t])
            if gate:
                tp = max([b[d] for d in ids if t in K.dist[d] and any(
                    R.dist_eff(d, t)[o] >= 2 * K.bgt[t][o] for o in K.outcomes[t])] or [0.0])
                if tp < gate:
                    continue
            acts.append(("t", t, K.ttype[t]))
        w = WB.pair_w(ids, dz_by_id, 0.0)
        scored = sorted(((WB.value_v02(K, R, ids, probs, other, a, w), a) for a in acts), key=lambda x: -x[0])
        options = []
        for types, lim in (({"history"}, 3), ({"exam"}, 3), ({"poc", "lab"}, 3)):
            batch = [(v, a) for v, a in scored if a[2] in types and v > 0][:lim]
            if batch:
                options.append((sum(v for v, _ in batch) - S.turn_cost, [a for _, a in batch]))
        for v, a in scored:
            if a[2] in ("imaging", "advanced_imaging", "procedure"):
                options.append((v - S.turn_cost, [a]))
        if not options and scored:
            options.append((scored[0][0] - S.turn_cost, [scored[0][1]]))
        options.sort(key=lambda o: -o[0])
        best = options[0][0] if options else 0.0
        can = turn >= S.min_turns_before_commit
        commit = None
        if can and leader >= S.commit_threshold:
            commit = "threshold"
        elif can and best <= S.voi_epsilon and leader >= S.min_commit_prob:
            commit = "no_value"
        elif can and best <= S.voi_epsilon and stall + 1 >= S.stall_turns:
            commit = "stalled"
        if commit:
            # critic: one rescue attempt when the truth is missing (same in every arm)
            if not full_list and not rescued and case.disease not in hyp:
                rescued = True
                if u(case.idx, "critic") < CRITIC_RESCUE:
                    hyp.append(case.disease)
                    K, R = build(fam, known, hyp, dz_by_id)
                    continue
            reason = commit
            break
        stall = stall + 1 if best <= S.voi_epsilon else 0
        turn += 1
        if not options:
            continue
        for a in options[0][1]:
            if a[0] == "f":
                f = a[1][0]
                obs.append(("f", f, case.frep[f], WB.REPORT_CONF[K.src[f]]))
                asked.add(f)
                cost += 0.0 if a[2] == "history" else 5.0
            else:
                t = a[1]
                obs.append(("t", t, case.trep[t], 0.97))
                done.add(t)
                cost += K.tcost[t]
    b, _ = WB.posterior(K, R, obs)
    ranked = sorted(b, key=b.get, reverse=True)
    dz_t = dz_by_id[case.disease]
    return {"arm": arm, "idx": case.idx, "truth": case.disease, "top1": ranked[0], "p_top1": b[ranked[0]],
            "p_truth": b.get(case.disease, 0.0), "rank": ranked.index(case.disease) + 1 if case.disease in b else 99,
            "in_list": case.disease in b, "list_size": len(hyp), "turns": turn + 1, "cost": cost, "reason": reason,
            "cm": dz_t.cant_miss, "added_reverse": added["reverse"], "added_mimic": added["mimic"]}


def summarize(rs):
    n = len(rs)
    k = sum(r["top1"] == r["truth"] for r in rs)
    cm = [r for r in rs if r["cm"]]
    return {"n": n, "top1": k / n, "ci": wilson(k, n), "top3": sum(r["rank"] <= 3 for r in rs) / n,
            "recall": sum(r["in_list"] for r in rs) / n, "list_size": sum(r["list_size"] for r in rs) / n,
            "brier": sum((r["p_top1"] - (r["top1"] == r["truth"])) ** 2 for r in rs) / n, "ece": ece(rs),
            "logloss": -sum(math.log(max(r["p_truth"], 1e-9)) for r in rs) / n,
            "cm_missed": sum(r["top1"] != r["truth"] for r in cm) / max(1, len(cm)),
            "turns": sum(r["turns"] for r in rs) / n, "cost": sum(r["cost"] for r in rs) / n,
            "added_reverse": sum(r["added_reverse"] for r in rs) / n, "added_mimic": sum(r["added_mimic"] for r in rs) / n}


def paired(ra, rb):
    key = lambda r: (r["world"], r["idx"])
    a = {key(r): r for r in ra}
    b = {key(r): r for r in rb}
    ids = sorted(a)
    A, B = [a[i] for i in ids], [b[i] for i in ids]
    ca = [r["top1"] == r["truth"] for r in A]
    cb = [r["top1"] == r["truth"] for r in B]
    b01 = sum((not x) and y for x, y in zip(ca, cb))
    b10 = sum(x and not y for x, y in zip(ca, cb))
    f = lambda g: boot_diff([g(r) for r in B], [g(r) for r in A])
    cmA = [r for r in A if r["cm"]]
    cmB = [b[key(r)] for r in cmA]
    out = {"top1": f(lambda r: float(r["top1"] == r["truth"])), "mcnemar_p": mcnemar_exact(b01, b10),
           "b_better": b01, "a_better": b10, "top3": f(lambda r: float(r["rank"] <= 3)),
           "brier": f(lambda r: (r["p_top1"] - (r["top1"] == r["truth"])) ** 2),
           "logloss": f(lambda r: -math.log(max(r["p_truth"], 1e-9))), "ece": ece_boot(A, B),
           "cm_missed": boot_diff([float(r["top1"] != r["truth"]) for r in cmB], [float(r["top1"] != r["truth"]) for r in cmA]),
           "turns": f(lambda r: r["turns"]), "cost": f(lambda r: r["cost"]),
           "recall": f(lambda r: float(r["in_list"]))}
    out["verdict"] = verdict(out)
    return out


def validate(n=200, seed=1000):
    """With the full family list and no added sources, this loop must reproduce the validated v0.2 engine."""
    fams = WB.gen_world(seed)
    dz_by_id = {d.id: d for f in fams for d in f.diseases}
    known = {d: WB.known_card(dz_by_id[d], fams[dz_by_id[d].family], 0.5, random.Random(seed + 1)) for d in sorted(dz_by_id)}
    agree = 0
    for i in range(n):
        c = WB.sample_case(fams, i, seed)
        K, R0 = WB.build_agent(fams[c.fam], known, {}, c, 1.0, False)
        a = WB.run(WB.MECHS["v02"], K, R0, c, dz_by_id)
        b = run("baseline", fams[c.fam], known, c, dz_by_id, 1.0, full_list=True)
        agree += (a["top1"], a["turns"], round(a["cost"], 6)) == (b["top1"], b["turns"], round(b["cost"], 6))
    return agree / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=3)
    ap.add_argument("--cases", type=int, default=1500)
    ap.add_argument("--recall", type=float, nargs="+", default=[0.6, 0.75, 0.9])
    ap.add_argument("--arms", nargs="+", default=["baseline", "reverse", "mimic", "both"])
    ap.add_argument("--knowledge-sd", type=float, default=0.5)
    ap.add_argument("--out", default="runs/hyp_recall")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--reverse-prior", type=float, default=None, help="fixed prior for index-added diseases")
    args = ap.parse_args()
    REVERSE_PRIOR["value"] = args.reverse_prior
    if args.validate:
        print("full-list baseline == validated v0.2 engine:", validate())
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for r in args.recall:
        rows = {a: [] for a in args.arms}
        t0 = time.time()
        for w in range(args.worlds):
            seed = 1000 + w
            fams = WB.gen_world(seed)
            dz_by_id = {d.id: d for f in fams for d in f.diseases}
            known = {d: WB.known_card(dz_by_id[d], fams[dz_by_id[d].family], args.knowledge_sd, random.Random(seed + 1)) for d in sorted(dz_by_id)}
            cases = [WB.sample_case(fams, i, seed) for i in range(args.cases)]
            for a in args.arms:
                for c in cases:
                    row = run(a, fams[c.fam], known, c, dz_by_id, r)
                    row["world"] = w
                    rows[a].append(row)
        res = {"summary": {a: summarize(v) for a, v in rows.items()}, "paired_vs_baseline": {}}
        for a in args.arms:
            if a != "baseline":
                res["paired_vs_baseline"][a] = paired(rows["baseline"], rows[a])
        results[str(r)] = res
        print(f"recall {r}: {time.time() - t0:.0f}s", flush=True)
    (out / "summary.json").write_text(json.dumps(results, indent=1))
    print("wrote", out / "summary.json")


if __name__ == "__main__":
    main()
