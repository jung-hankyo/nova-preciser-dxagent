"""Paired comparison: v0.1 (stored likelihood ratios) vs v0.2 (per-disease frequencies).

Design (fair by construction)
-----------------------------
* A hidden GROUND-TRUTH world (the seed knowledge base: 8 diseases, 29 findings, 11 categorical tests)
  generates every patient: true disease ~ prior, each finding ~ Bernoulli(P(f|d)), each test outcome
  ~ Categorical(P(.|d)); unlisted entries follow the background.
* Both agents receive knowledge derived from that truth with the SAME random noise draw
  (odds scaled by up to x/÷f). v0.1 receives it as LR+/LR- per disease with each test collapsed to
  the disease's own "positive" outcome(s) (as the v0.1 seed cards did); v0.2 receives frequencies,
  backgrounds, and full categorical outcome distributions.
* Every patient's answers are pre-sampled (with channel noise: history 10%, exam 5%, tests 3%),
  so both agents get identical replies to identical questions: a paired design.
* Intake (HPI volunteers, past history, vitals) is identical for every condition.

Conditions (representation x objective x commit rule)
    A  v0.1        LR/binary tests  + harm-weighted binary-entropy objective + can't-miss commit rule
    B  v0.2        frequency/categorical + pairwise Bhattacharyya objective  + accuracy commit rule
    C  B, v0.1 objective         (isolates the objective)
    D  B, v0.1 representation    (isolates the representation)
    E  B + can't-miss commit rule (isolates the commit rule)
Condition B is additionally cross-checked against the shipped dxagent package on a subsample.

Run:  python experiments/lr_vs_freq.py --cases 150 --noise 1 2.5 --out runs/lr_vs_freq
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dxagent.config import CHANNEL_RELIABILITY, MIN_TARGET_PROB, get_settings  # noqa: E402
from dxagent.knowledge.cards import CardStore  # noqa: E402

S = get_settings()
EPS = 1e-3
V01_HARM = {"acute_coronary_syndrome": 30, "pulmonary_embolism": 30, "aortic_dissection": 30, "pneumothorax": 10,
            "pneumonia": 10, "heart_failure_exacerbation": 10, "pericarditis": 3, "copd_exacerbation": 3}
REPORT_CONF = {"history": 0.85, "exam": 0.90}
FLIP = {"history": 0.10, "exam": 0.05}
TEST_NOISE = 0.03


def clamp(p): return min(max(p, EPS), 1 - EPS)
def logit(p): return math.log(p / (1 - p))
def sigm(x): return 1 / (1 + math.exp(-x))
def h(p): return 0.0 if p <= 0 or p >= 1 else -(p * math.log(p) + (1 - p) * math.log(1 - p))


# ---------------------------------------------------------------------------------- world and knowledge

@dataclass
class World:
    diseases: list
    prior: dict
    p: dict          # p[d][f] listed findings
    bg: dict         # bg[f]
    src: dict        # src[f] history|exam
    cluster: dict
    dist: dict       # dist[d][t] listed tests
    bgt: dict        # bgt[t]
    outcomes: dict
    ttype: dict
    tcost: dict
    trisk: dict
    cant_miss: dict
    names: dict

    def p_true(self, d, f): return self.p[d].get(f, self.bg[f])
    def dist_true(self, d, t): return self.dist[d].get(t, self.bgt[t])


def _norm(d):
    z = sum(d.values())
    return {k: v / z for k, v in d.items()}


def load_truth() -> World:
    st = CardStore(S.registry_dir, None, card_dirs=[S.fixture_dir])
    ds = [c.id for c in st.all_cards()]
    prior = {c.id: (0.02 if c.cant_miss else 0.08) for c in st.all_cards()}
    fds = {f: st.finding_def(f) for f in st.vocabulary()}
    return World(
        diseases=ds, prior=prior,
        p={c.id: {e.code: e.p for e in c.findings} for c in st.all_cards()},
        bg={f: d.background for f, d in fds.items()}, src={f: d.source for f, d in fds.items()},
        cluster={f: d.cluster for f, d in fds.items()},
        dist={c.id: {t: _norm({o: max(e.dist.get(o, 0.0), EPS) for o in st.test_def(t).outcomes}) for t, e in c.tests.items()}
              for c in st.all_cards()},
        bgt={t.id: dict(t.background) for t in st.tests()}, outcomes={t.id: list(t.outcomes) for t in st.tests()},
        ttype={t.id: t.type for t in st.tests()}, tcost={t.id: t.cost for t in st.tests()}, trisk={t.id: t.risk for t in st.tests()},
        cant_miss={c.id: c.cant_miss for c in st.all_cards()}, names={c.id: c.name for c in st.all_cards()},
    )


def noisy(world: World, f: float, seed: int) -> World:
    """Same knowledge error for every condition: odds of each value scaled by exp(U(-ln f, ln f))."""
    if f <= 1:
        return world
    rng = random.Random(seed)
    a = math.log(f)
    u = lambda: rng.uniform(-a, a)
    p = {d: {k: clamp(sigm(logit(v) + u())) for k, v in fs.items()} for d, fs in world.p.items()}
    bg = {k: clamp(sigm(logit(v) + u())) for k, v in world.bg.items()}
    def nd(dd):
        x = {o: max(v, EPS) * math.exp(u()) for o, v in dd.items()}
        z = sum(x.values())
        return {o: v / z for o, v in x.items()}
    dist = {d: {t: nd(v) for t, v in ts.items()} for d, ts in world.dist.items()}
    bgt = {t: nd(v) for t, v in world.bgt.items()}
    return World(world.diseases, world.prior, p, bg, world.src, world.cluster, dist, bgt, world.outcomes,
                 world.ttype, world.tcost, world.trisk, world.cant_miss, world.names)


@dataclass
class Case:
    idx: int
    disease: str
    finding_report: dict     # f -> reported present (bool), pre-sampled with channel noise
    test_report: dict        # t -> reported outcome
    intake: list             # [(turn, [f...])]


def sample_case(world: World, idx: int, seed: int) -> Case:
    rng = random.Random(seed * 100003 + idx)
    ds, ws = zip(*world.prior.items())
    d = rng.choices(ds, weights=ws)[0]
    rep = {}
    for f in world.bg:
        true = rng.random() < world.p_true(d, f)
        rep[f] = (not true) if rng.random() < FLIP[world.src[f]] else true
    trep = {}
    for t in world.bgt:
        dd = world.dist_true(d, t)
        o = rng.choices(list(dd), weights=list(dd.values()))[0]
        if rng.random() < TEST_NOISE:
            o = rng.choices(list(world.bgt[t]), weights=list(world.bgt[t].values()))[0]
        trep[t] = o
    hist = [f for f in world.bg if world.src[f] == "history"]
    volunteered = [f for f in hist if rep[f]]
    rng.shuffle(volunteered)
    intake = [(1, volunteered[:3]), (2, volunteered[3:5]), (3, ["tachycardia", "temp_gt_38", "crackles", "wheeze"])]
    return Case(idx, d, rep, trep, intake)


# ---------------------------------------------------------------------------------- representations

class Rep:
    """Knowledge as the agent sees it. kind='freq' (v0.2) or 'lr' (v0.1)."""

    def __init__(self, K: World, kind: str, shrink: float = 0.9, report_model: bool = False, ref_sigma: float = 0.0):
        self.K, self.kind, self.c, self.report_model = K, kind, shrink, report_model
        # v0.1 LRs come from studies with their own comparison groups: q(d, x) = comparison-group rate.
        self.q, self.q_plan = {}, {}
        if kind == "lr":
            rng = random.Random(12345)
            per_code: dict = {}
            for d in K.diseases:
                for f in K.p[d]:
                    v = clamp(sigm(logit(K.bg[f]) + rng.gauss(0, ref_sigma))) if ref_sigma else K.bg[f]
                    self.q[(d, f)] = v
                    per_code.setdefault(f, []).append(v)
                for t, dd in K.dist[d].items():
                    for o in dd:
                        v = clamp(sigm(logit(K.bgt[t][o]) + rng.gauss(0, ref_sigma))) if ref_sigma else K.bgt[t][o]
                        self.q[(d, (t, o))] = v
                        per_code.setdefault((t, o), []).append(v)
            import statistics
            self.q_plan = {k: statistics.median(v) for k, v in per_code.items()}  # v0.1 planner background
        if kind == "lr":  # v0.1 cards list, per test, only the disease's own "positive" outcome(s)
            self.pos = {}
            for d in K.diseases:
                for t, dd in K.dist[d].items():
                    ratios = {o: dd[o] / K.bgt[t][o] for o in dd}
                    keep = [o for o, r in ratios.items() if r >= 1.5 and o not in ("normal",)]
                    if not keep:
                        keep = [max(ratios, key=ratios.get)]
                    self.pos[(d, t)] = keep

    # --- findings
    def p_find(self, d, f):
        K = self.K
        if f not in K.p[d]:
            return self.q_plan.get(f, K.bg[f]) if self.kind == "lr" else K.bg[f]
        p = K.p[d][f]
        return clamp(self.c * p + (1 - self.c) * K.bg[f]) if self.kind == "freq" else p

    def listed(self, d, f): return f in self.K.p[d]

    # --- tests
    def test_codes(self, d, t):
        """Binary codes (t, o) disease d lists (lr), or None for categorical."""
        return [(t, o) for o in self.pos.get((d, t), [])]

    def dist_eff(self, d, t):
        K = self.K
        if t not in K.dist[d]:
            return K.bgt[t]
        dd = K.dist[d][t]
        if self.kind == "freq":
            return {o: self.c * dd[o] + (1 - self.c) * K.bgt[t][o] for o in dd}
        return dd

    # --- belief contributions
    def llr(self, d, obs):
        """List of (cluster, llr) for one observation."""
        K = self.K
        kind, key, val, conf = obs
        if kind == "f":
            if not self.listed(d, key):
                return []
            p, b = self.p_find(d, key), (self.q[(d, key)] if self.kind == "lr" else K.bg[key])
            if self.report_model:
                r = CHANNEL_RELIABILITY[K.src[key]]
                p, b = r * p + (1 - r) * (1 - p), r * b + (1 - r) * (1 - b)
            x = math.log(p / b) if val else math.log((1 - p) / (1 - b))
            return [(K.cluster[key], conf * x)]
        t, o = key, val
        if t not in K.dist[d]:
            return []
        if self.kind == "freq":
            de = self.dist_eff(d, t)
            p = de[o]
            if self.report_model:
                r = CHANNEL_RELIABILITY[K.ttype[t]]
                p = r * p + (1 - r) * K.bgt[t][o]
            return [(f"test:{t}", conf * math.log(p / K.bgt[t][o]))]
        out = []
        for (_, o2) in self.test_codes(d, t):  # binary finding "t:o2" present iff observed o == o2
            p, b = K.dist[d][t][o2], self.q[(d, (t, o2))]
            x = math.log(p / b) if o == o2 else math.log((1 - p) / (1 - b))
            out.append((f"test:{t}", conf * x))
        return out


# ---------------------------------------------------------------------------------- engine

@dataclass
class Cond:
    name: str
    rep: str            # 'freq' | 'lr'
    objective: str      # 'pair' | 'entropy'
    cantmiss_rule: bool
    report_model: bool = False  # v0.2.1 patch: belief uses the planner's report-noise model
    ref_sigma: float = 0.0      # v0.1 only: each LR's study used its own comparison group (log-odds SD)


CONDS = {
    "A_v01": Cond("A_v01", "lr", "entropy", True),
    "B_v02": Cond("B_v02", "freq", "pair", False),
    "C_v02_old_objective": Cond("C_v02_old_objective", "freq", "entropy", False),
    "D_v02_old_representation": Cond("D_v02_old_representation", "lr", "pair", False),
    "E_v02_cantmiss_rule": Cond("E_v02_cantmiss_rule", "freq", "pair", True),
    "F_v021_report_noise": Cond("F_v021_report_noise", "freq", "pair", False, True),
    "A_v01_ref0.5": Cond("A_v01_ref0.5", "lr", "entropy", True, ref_sigma=0.5),
    "A_v01_ref1.0": Cond("A_v01_ref1.0", "lr", "entropy", True, ref_sigma=1.0),
}


def posterior(K: World, R: Rep, obs: list):
    ds = K.diseases
    prior_sum = sum(K.prior.values())
    other_prior = min(max(1 - prior_sum, S.other_mass), 0.3)
    scale = (1 - other_prior) / prior_sum
    ls = {}
    for d in ds:
        pos, neg = {}, {}
        for o in obs:
            for cl, x in R.llr(d, o):
                if x >= 0:
                    pos[cl] = max(pos.get(cl, 0.0), x)
                else:
                    neg[cl] = min(neg.get(cl, 0.0), x)
        ls[d] = math.log(K.prior[d] * scale) + sum(pos.values()) + sum(neg.values())
    lo = math.log(other_prior)
    m = max(max(ls.values()), lo)
    z = sum(math.exp(v - m) for v in ls.values()) + math.exp(lo - m)
    b = {d: math.exp(v - m) / z for d, v in ls.items()}
    return b, math.exp(lo - m) / z


def objective_value(cond: Cond, K: World, ids, probs, other):
    if cond.objective == "pair":
        tot = 0.0
        for i in range(len(probs)):
            for j in range(i + 1, len(probs)):
                tot += math.sqrt(probs[i] * probs[j])
            tot += math.sqrt(probs[i] * other)
        return tot
    return sum((1 + math.log(V01_HARM[d])) * h(p) for d, p in zip(ids, probs)) + h(other)


def score(cond, K, R, ids, probs, other, action):
    """action = ('f', [codes], type) | ('t', test_id, type). Returns (voi, gain)."""
    base_err = 1 - max(probs)
    base_o = objective_value(cond, K, ids, probs, other)
    kind = action[0]
    outcomes = []  # (liks per disease, lik other)
    if kind == "f":
        codes, typ = action[1], action[2]
        r = CHANNEL_RELIABILITY[typ]
        nz = lambda p: r * p + (1 - r) * (1 - p)
        pd = {c: [nz(R.p_find(d, c)) for d in ids] for c in codes}
        po = {c: nz(K.bg[c]) for c in codes}
        for pat in itertools.product([True, False], repeat=len(codes)):
            liks = [1.0] * len(ids)
            lo = 1.0
            for c, pres in zip(codes, pat):
                liks = [l * (pd[c][k] if pres else 1 - pd[c][k]) for k, l in enumerate(liks)]
                lo *= po[c] if pres else 1 - po[c]
            outcomes.append((liks, lo))
        cost, risk = (0.0 if typ == "history" else 5.0), 0.0
    else:
        t, typ = action[1], action[2]
        r = CHANNEL_RELIABILITY[typ]
        cost, risk = K.tcost[t], K.trisk[t]
        if R.kind == "freq":
            dists = [R.dist_eff(d, t) for d in ids]
            for o in K.outcomes[t]:
                outcomes.append(([r * dd[o] + (1 - r) * K.bgt[t][o] for dd in dists], K.bgt[t][o]))
        else:  # v0.1: the test's listed binary codes, treated as INDEPENDENT binary findings
            codes = sorted({c for d in ids for c in R.test_codes(d, t)})
            if not codes:
                return -1.0, 0.0
            def pc(d, c):
                return K.dist[d][t][c[1]] if c in R.test_codes(d, t) else R.q_plan.get(c, K.bgt[t][c[1]])
            codes = sorted(codes, key=lambda c: -sum(p * (pc(d, c) - K.bgt[t][c[1]]) ** 2 for d, p in zip(ids, probs)))[:3]
            nz = lambda p: r * p + (1 - r) * (1 - p)
            pd = {c: [nz(pc(d, c)) for d in ids] for c in codes}
            po = {c: nz(K.bgt[t][c[1]]) for c in codes}
            for pat in itertools.product([True, False], repeat=len(codes)):
                liks = [1.0] * len(ids)
                lo = 1.0
                for c, pres in zip(codes, pat):
                    liks = [l * (pd[c][k] if pres else 1 - pd[c][k]) for k, l in enumerate(liks)]
                    lo *= po[c] if pres else 1 - po[c]
                outcomes.append((liks, lo))
    exp_err = exp_o = 0.0
    for liks, lo in outcomes:
        joint = [p * l for p, l in zip(probs, liks)]
        jo = other * lo
        z = sum(joint) + jo
        if z <= 0:
            continue
        post = [j / z for j in joint]
        exp_err += z * (1 - max(post))
        exp_o += z * objective_value(cond, K, ids, post, jo / z)
    gain = S.misdiagnosis_loss * (base_err - exp_err) + S.pair_weight * (base_o - exp_o)
    return gain - S.cost_weight * cost - risk, gain


def target_ok(K, R, t, b):
    gate = MIN_TARGET_PROB.get(K.ttype[t])
    if gate is None:
        return True
    best = 0.0
    for d in K.diseases:
        if t not in K.dist[d]:
            continue
        if R.kind == "freq":
            de = R.dist_eff(d, t)
            ok = any(de[o] >= 2 * K.bgt[t][o] for o in de)
        else:
            ok = any(K.dist[d][t][o] / K.bgt[t][o] >= 2 for _, o in R.test_codes(d, t))
        if ok:
            best = max(best, b[d])
    return best >= gate


def run_case(cond: Cond, K: World, case: Case) -> dict:
    R = Rep(K, cond.rep, report_model=cond.report_model, ref_sigma=cond.ref_sigma)
    obs, asked, done_tests = [], set(), set()
    turn, cost, stall = 0, 0.0, 0
    n_proc = n_adv = 0
    reason = ""
    for t_i, codes in case.intake:
        turn = t_i
        for f in codes:
            obs.append(("f", f, case.finding_report[f], REPORT_CONF[K.src[f]]))
            asked.add(f)
        if t_i == 3:
            cost += 5.0
    while True:
        b, other = posterior(K, R, obs)
        ranked = sorted(b, key=b.get, reverse=True)
        ids = ranked
        probs = [b[d] for d in ids]
        leader = probs[0]
        if turn >= S.max_turns - 1:
            reason = "budget"
            break
        # candidates
        acts = []
        for f in K.bg:
            if f in asked or not any(R.listed(d, f) for d in ids):
                continue
            acts.append(("f", [f], K.src[f]))
        for t in K.bgt:
            if t in done_tests or not any(t in K.dist[d] for d in ids):
                continue
            if not target_ok(K, R, t, b):
                continue
            acts.append(("t", t, K.ttype[t]))
        scored = sorted(((score(cond, K, R, ids, probs, other, a), a) for a in acts), key=lambda x: -x[0][0])
        # turn-level options
        options = []
        for group, types, lim in (("history", {"history"}, 3), ("exam", {"exam"}, 3), ("labs", {"poc", "lab"}, 3)):
            batch = [(v, a) for (v, g), a in scored if a[2] in types and v > 0][:lim]
            if batch:
                options.append((sum(v for v, _ in batch) - S.turn_cost, [a for _, a in batch]))
        for (v, g), a in scored:
            if a[2] in ("imaging", "advanced_imaging", "procedure"):
                options.append((v - S.turn_cost, [a]))
        if not options and scored:
            options.append((scored[0][0][0] - S.turn_cost, [scored[0][1]]))
        options.sort(key=lambda o: -o[0])
        best = options[0][0] if options else 0.0
        can = turn >= S.min_turns_before_commit
        cm_live = any(K.cant_miss[d] and b[d] >= S.exclusion_threshold for d in ids[1:])
        if can and leader >= S.commit_threshold and not (cond.cantmiss_rule and cm_live):
            reason = "threshold"
            break
        if can and best <= S.voi_epsilon and leader >= S.min_commit_prob:
            reason = "no_value"
            break
        if can and best <= S.voi_epsilon and stall + 1 >= S.stall_turns:
            reason = "stalled"
            break
        stall = stall + 1 if best <= S.voi_epsilon else 0
        turn += 1
        if not options:
            continue  # uninformative free question
        for a in options[0][1]:
            if a[0] == "f":
                f = a[1][0]
                obs.append(("f", f, case.finding_report[f], REPORT_CONF[K.src[f]]))
                asked.add(f)
                cost += 0.0 if a[2] == "history" else 5.0
            else:
                t = a[1]
                obs.append(("t", t, case.test_report[t], 0.97))
                done_tests.add(t)
                cost += K.tcost[t]
                n_proc += K.ttype[t] == "procedure"
                n_adv += K.ttype[t] == "advanced_imaging"
    b, other = posterior(K, R, obs)
    ranked = sorted(b, key=b.get, reverse=True)
    return {"cond": cond.name, "idx": case.idx, "truth": case.disease, "top1": ranked[0], "p_top1": b[ranked[0]],
            "p_truth": b[case.disease], "rank": ranked.index(case.disease) + 1, "turns": turn + 1, "cost": cost,
            "reason": reason, "procedures": n_proc, "advanced_imaging": n_adv, "tests": len(done_tests)}


# ---------------------------------------------------------------------------------- statistics

def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - w, c + w)


def mcnemar_exact(b01, b10):
    """Two-sided exact McNemar p-value on discordant pairs."""
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def boot_diff(x, y, reps=4000, seed=0):
    """Paired bootstrap of mean(x - y): (mean, 2.5%, 97.5%)."""
    import numpy as np
    d = np.asarray(x, float) - np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def ece(rows, bins=10):
    tot = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sel = [r for r in rows if lo <= r["p_top1"] < hi or (i == bins - 1 and r["p_top1"] == 1.0)]
        if sel:
            conf = sum(r["p_top1"] for r in sel) / len(sel)
            acc = sum(r["top1"] == r["truth"] for r in sel) / len(sel)
            tot += len(sel) / len(rows) * abs(conf - acc)
    return tot


def summarize(rows, world):
    n = len(rows)
    k = sum(r["top1"] == r["truth"] for r in rows)
    cm = [r for r in rows if world.cant_miss[r["truth"]]]
    return {
        "n": n,
        "top1": k / n, "top1_ci": wilson(k, n),
        "top3": sum(r["rank"] <= 3 for r in rows) / n,
        "turns_mean": sum(r["turns"] for r in rows) / n,
        "cost_mean": sum(r["cost"] for r in rows) / n,
        "brier": sum((r["p_top1"] - (r["top1"] == r["truth"])) ** 2 for r in rows) / n,
        "log_loss_truth": -sum(math.log(max(r["p_truth"], 1e-9)) for r in rows) / n,
        "ece": ece(rows),
        "cantmiss_n": len(cm), "cantmiss_missed": sum(r["top1"] != r["truth"] for r in cm) / max(1, len(cm)),
        "procedure_rate": sum(r["procedures"] > 0 for r in rows) / n,
        "adv_imaging_rate": sum(r["advanced_imaging"] > 0 for r in rows) / n,
        "reasons": {x: sum(r["reason"] == x for r in rows) / n for x in ("threshold", "no_value", "stalled", "budget")},
        "per_disease": {d: sum(r["top1"] == r["truth"] for r in rows if r["truth"] == d) / max(1, sum(r["truth"] == d for r in rows))
                        for d in world.diseases},
    }


def paired(rows_a, rows_b):
    a = {r["idx"]: r for r in rows_a}
    b = {r["idx"]: r for r in rows_b}
    ids = sorted(a)
    ca = [a[i]["top1"] == a[i]["truth"] for i in ids]
    cb = [b[i]["top1"] == b[i]["truth"] for i in ids]
    b01 = sum((not x) and y for x, y in zip(ca, cb))  # A wrong, B right
    b10 = sum(x and (not y) for x, y in zip(ca, cb))
    acc = boot_diff([float(y) for y in cb], [float(x) for x in ca])
    turns = boot_diff([b[i]["turns"] for i in ids], [a[i]["turns"] for i in ids])
    cost = boot_diff([b[i]["cost"] for i in ids], [a[i]["cost"] for i in ids])
    ll = boot_diff([-math.log(max(b[i]["p_truth"], 1e-9)) for i in ids], [-math.log(max(a[i]["p_truth"], 1e-9)) for i in ids])
    return {"acc_diff": acc, "b_right_a_wrong": b01, "a_right_b_wrong": b10, "mcnemar_p": mcnemar_exact(b01, b10),
            "turns_diff": turns, "cost_diff": cost, "logloss_diff": ll}


def simulate(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    truth = load_truth()
    cases = [sample_case(truth, i, args.seed) for i in range(args.cases)]
    for f in args.noise:
        K = noisy(truth, f, args.seed + int(f * 100))
        with open(out / f"rows_noise{f}.jsonl", "w") as fh:
            for c in args.conds:
                t0 = time.time()
                rows = [run_case(CONDS[c], K, case) for case in cases]
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                print(f"noise x{f}: {c} {time.time() - t0:.0f}s top1 {sum(r['top1'] == r['truth'] for r in rows) / len(rows):.3f}", flush=True)


def analyze(args):
    out = Path(args.out)
    truth = load_truth()
    results = {}
    for f in args.noise:
        rows: dict = {}
        for line in open(out / f"rows_noise{f}.jsonl"):
            r = json.loads(line)
            rows.setdefault(r["cond"], []).append(r)
        results[str(f)] = {
            "summary": {c: summarize(v, truth) for c, v in rows.items()},
            "paired_vs_A": {c: paired(rows["A_v01"], rows[c]) for c in rows if c != "A_v01"},
            "paired_vs_B": {c: paired(rows[c], rows["B_v02"]) for c in rows if c != "B_v02"},
        }
    (out / "summary.json").write_text(json.dumps(results, indent=1))
    print("wrote", out / "summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=1000)
    ap.add_argument("--noise", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--conds", nargs="+", default=list(CONDS))
    ap.add_argument("--out", default="runs/lr_vs_freq")
    ap.add_argument("--stage", choices=["simulate", "analyze", "all"], default="all")
    args = ap.parse_args()
    if args.stage in ("simulate", "all"):
        simulate(args)
    if args.stage in ("analyze", "all"):
        analyze(args)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------- cross-check with the package

def package_store(K: World, tmp: Path) -> CardStore:
    """Write world K as a v0.2 knowledge base (conf 0.9 everywhere) and load it with the shipped CardStore."""
    import yaml
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "_findings.yaml").write_text(yaml.safe_dump([
        {"code": f, "label": f, "source": K.src[f], "background": K.bg[f], "cluster": K.cluster[f]} for f in K.bg]))
    (tmp / "_tests.yaml").write_text(yaml.safe_dump([
        {"id": t, "name": t, "type": K.ttype[t], "outcomes": K.outcomes[t], "background": K.bgt[t],
         "cost": K.tcost[t], "risk": K.trisk[t]} for t in K.bgt]))
    course = {True: "lethal", False: "serious"}
    for d in K.diseases:
        (tmp / f"{d}.yaml").write_text(yaml.safe_dump({
            "id": d, "name": K.names[d], "harm": {"untreated_course": course[K.cant_miss[d]], "urgency": "hours"},
            "findings": [{"code": f, "p": p, "conf": 0.9} for f, p in K.p[d].items()],
            "tests": {t: {"dist": dd, "conf": 0.9} for t, dd in K.dist[d].items()}}))
    return CardStore(tmp, None)


def run_case_package(K: World, store: CardStore, case: Case) -> dict:
    from dxagent.reasoning.belief import compute_posterior
    from dxagent.reasoning.planner import decide
    hyps = [{"disease_id": d, "name": K.names[d], "prior": K.prior[d], "cant_miss": K.cant_miss[d]} for d in K.diseases]
    findings, tests, observed, done = [], [], set(), set()
    turn, cost, stall = 0, 0.0, 0

    def add_f(f, t):
        findings.append({"code": f, "label": f, "present": case.finding_report[f], "confidence": REPORT_CONF[K.src[f]],
                         "source": K.src[f], "evidence": "", "turn": t, "chief_complaint": False})
        observed.add(f)

    for t_i, codes in case.intake:
        turn = t_i
        for f in codes:
            add_f(f, t_i)
        if t_i == 3:
            cost += 5.0
    while True:
        res = compute_posterior(hyps, findings, store, S, tests)
        dec = decide(res["differential"], res["other_prob"], store, S, turn, observed, done, stall_count=stall)
        if dec.commit:
            break
        stall = stall + 1 if dec.best_voi <= S.voi_epsilon else 0
        turn += 1
        for sa in dec.actions:
            a = sa.action
            done.add(a.id)
            if a.test:
                tests.append({"test_id": a.test, "outcome": case.test_report[a.test], "confidence": 0.97, "turn": turn, "evidence": ""})
                observed.add(a.test)
                cost += a.cost
            else:
                for f in a.reveals:
                    add_f(f, turn)
                cost += a.cost
    res = compute_posterior(hyps, findings, store, S, tests)
    diff = res["differential"]
    return {"top1": diff[0]["disease_id"], "p_top1": diff[0]["prob"], "turns": turn + 1, "cost": cost}


def validate(n: int, f: float, seed: int = 7) -> dict:
    import tempfile
    truth = load_truth()
    K = noisy(truth, f, seed + int(f * 100))
    store = package_store(K, Path(tempfile.mkdtemp()))
    agree = {"top1": 0, "turns": 0, "cost": 0, "p_close": 0}
    for i in range(n):
        case = sample_case(truth, i, seed)
        a = run_case(CONDS["B_v02"], K, case)
        b = run_case_package(K, store, case)
        agree["top1"] += a["top1"] == b["top1"]
        agree["turns"] += a["turns"] == b["turns"]
        agree["cost"] += abs(a["cost"] - b["cost"]) < 1e-6
        agree["p_close"] += abs(a["p_top1"] - b["p_top1"]) < 1e-6
    return {k: v / n for k, v in agree.items()}


if __name__ == "__main__" and "--validate" in sys.argv:
    pass
