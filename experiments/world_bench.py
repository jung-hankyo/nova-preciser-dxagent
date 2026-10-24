"""World benchmark: diverse synthetic diseases, competitor mechanisms, harm term, live drafting.

What this measures
------------------
Reasoning MECHANISMS in one controlled world, not the real systems. MAI-DxO and AMIE are not publicly
runnable and every system here would need an LLM; each archetype below implements the decision
mechanism its paper describes, given best-case conditions (perfect extraction, the same knowledge,
exact Bayesian arithmetic wherever the paper claims Bayesian updating).

World
-----
F presenting-complaint families (default 12) x 5-8 diseases each. Per family: a pool of history/exam
findings with background rates, 3 categorical family tests (normal + 2-4 abnormal patterns), 2 cheap
labs, and one confirmatory test per disease (lab, advanced imaging, or procedure). Diseases come in
confusable sibling pairs that share characteristic findings and partly share confirmatory results.
Harm profiles, treatment classes, and priors are sampled (lethal diseases are rarer).
Several independent worlds are generated from different seeds.

Mechanisms (all see identical pre-sampled patients: paired design)
    v02            dxagent v0.2: Bayes over frequencies, pairwise objective, turn-level batching
    v02_clinical   v0.2 with harm-weighted pair weights (safety weight s) and the can't-miss commit rule
    maidxo         Bayesian differential; Test-Chooser picks up to 3 actions maximizing entropy reduction
                   among the TOP-3 hypotheses; Stewardship vetoes tests over $300 with low yield;
                   commits at confidence 0.8
    graphdx        quantized-typicality additive evidence scores (not Bayesian), cost-aware discriminative
                   coverage among the top-3, commit at normalized score 0.8
    amie           dialogue only: history questions chosen by the v0.2 planner (generous), exact Bayes,
                   no exam and no tests

Live drafting: a share of diseases is absent from the knowledge base. With drafting on, the hypothesis
generator proposes an unknown disease with recall r, and its card is drafted K times with LLM error
(log-odds SD sigma_d, entries omitted with probability 0.25), quantized to frequency levels, aggregated
by median, kept if produced by at least half the samples, confidence from agreement; grounding reduces
error for a share of entries. Error levels are swept because real LLM drafting error is unknown.

Run:  python experiments/world_bench.py --stage all --worlds 3 --cases 800
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dxagent.config import CHANNEL_RELIABILITY, FREQ, FREQ_LEVELS, MISS_HARM, OVERTREAT_HARM, URGENCY, get_settings  # noqa: E402
from lr_vs_freq import EPS, Rep, World, boot_diff, clamp, h, logit, mcnemar_exact, sigm, wilson  # noqa: E402

S = get_settings()
FLIP = {"history": 0.10, "exam": 0.05}
REPORT_CONF = {"history": 0.85, "exam": 0.90}
TEST_NOISE = 0.03
COSTS = {"poc": 20.0, "lab": 40.0, "imaging": 300.0, "advanced_imaging": 900.0, "procedure": 1500.0}
RISKS = {"poc": 0.0, "lab": 0.005, "imaging": 0.02, "advanced_imaging": 0.3, "procedure": 1.5}
PRIOR_BUCKETS = [0.25, 0.08, 0.02, 0.005, 0.001]


def norm(d):
    z = sum(d.values())
    return {k: v / z for k, v in d.items()}


# ============================================================================== world generation

@dataclass
class Disease:
    id: str
    family: int
    prior: float
    course: str
    urgency: str
    treatments: frozenset
    treat_risk: str
    p: dict                # listed findings -> P(f | d)
    dist: dict             # listed tests -> P(outcome | d)

    @property
    def cant_miss(self):
        return self.course == "lethal" and self.urgency in ("hours", "days")


@dataclass
class Family:
    idx: int
    diseases: list
    bg: dict
    src: dict
    cluster: dict
    bgt: dict
    outcomes: dict
    ttype: dict


def gen_world(seed: int, n_fam: int = 12) -> list[Family]:
    rng = random.Random(seed)
    fams = []
    for fi in range(n_fam):
        nd = rng.randint(5, 8)
        # findings pool
        nf = 20
        bg, src, cluster = {}, {}, {}
        for k in range(nf):
            code = f"F{fi}_f{k}"
            bg[code] = rng.choice([0.05, 0.1, 0.15, 0.2, 0.3])
            src[code] = "history" if k < 13 else "exam"
            cluster[code] = f"F{fi}_c{k // 2}" if rng.random() < 0.35 else code
        # tests: 3 family categorical, 2 cheap labs, 1 confirmatory per disease
        bgt, outcomes, ttype = {}, {}, {}
        for k in range(3):
            t = f"F{fi}_img{k}" if k == 0 else f"F{fi}_test{k}"
            outs = ["normal"] + [f"pattern{j}" for j in range(rng.randint(2, 4))]
            bgt[t] = norm({o: (0.55 if o == "normal" else 0.45 / (len(outs) - 1)) for o in outs})
            outcomes[t] = outs
            ttype[t] = "imaging" if k == 0 else rng.choice(["poc", "lab"])
        for k in range(2):
            t = f"F{fi}_lab{k}"
            outcomes[t] = ["normal", "abnormal"]
            bgt[t] = {"normal": 0.7, "abnormal": 0.3}
            ttype[t] = "lab"
        dids = [f"F{fi}_d{j}" for j in range(nd)]
        sib = {}
        for j in range(0, nd - 1, 2):
            sib[dids[j]], sib[dids[j + 1]] = dids[j + 1], dids[j]
        for d in dids:
            t = f"confirm_{d}"
            outcomes[t] = ["negative", "positive"]
            bgt[t] = {"negative": 0.95, "positive": 0.05}
            ttype[t] = rng.choice(["lab", "lab", "advanced_imaging", "advanced_imaging", "procedure"])
        treat_pool = [f"F{fi}_tx{k}" for k in range(4)]
        diseases = []
        char_sets = {}
        codes = list(bg)
        for d in dids:
            s = sib.get(d)
            if s in char_sets:  # sibling shares half of its characteristic findings
                base = rng.sample(sorted(char_sets[s]), 3)
                rest = rng.sample([c for c in codes if c not in char_sets[s]], 3)
                char = set(base + rest)
            else:
                char = set(rng.sample(codes, 6))
            char_sets[d] = char
        for d in dids:
            r = rng.random()
            course, urg = (("lethal", "hours") if r < 0.2 else ("lethal", "days") if r < 0.3
                           else ("serious", rng.choice(["days", "weeks"])) if r < 0.7 else ("self_limited", "weeks"))
            prior = rng.choice([0.01, 0.02, 0.03]) if course == "lethal" else rng.choice([0.04, 0.08, 0.15, 0.25])
            tx = frozenset(rng.sample(treat_pool, rng.randint(1, 2)))
            risk = rng.choice(["high", "moderate"]) if course == "lethal" else rng.choice(["moderate", "low", "low"])
            p = {c: rng.uniform(0.5, 0.92) for c in sorted(char_sets[d])}
            negs = rng.sample([c for c in codes if c not in char_sets[d] and bg[c] >= 0.15], 3)
            for c in negs:
                p[c] = rng.uniform(0.01, 0.05)
            dist = {}
            for t in [x for x in outcomes if x.startswith(f"F{fi}_")]:
                outs = outcomes[t]
                if rng.random() < 0.55:  # the disease has a signature pattern on this test
                    sig = rng.choice(outs[1:])
                    dd = {o: 0.05 for o in outs}
                    dd[sig] = rng.uniform(0.5, 0.85)
                    dd["normal"] = max(0.05, 1 - dd[sig] - 0.05 * (len(outs) - 2))
                    dist[t] = norm(dd)
            dist[f"confirm_{d}"] = {"negative": 0.1, "positive": 0.9}
            s = sib.get(d)
            if s:
                dist[f"confirm_{s}"] = {"negative": 0.8, "positive": 0.2}  # partial cross-reactivity
            diseases.append(Disease(d, fi, prior, course, urg, tx, risk, p, dist))
        fams.append(Family(fi, diseases, bg, src, cluster, bgt, outcomes, ttype))
    return fams


# ============================================================================== knowledge

def noisy_p(p, sd, rng):
    return clamp(sigm(logit(p) + rng.gauss(0, sd))) if sd > 0 else p


def noisy_dist(d, sd, rng):
    if sd <= 0:
        return dict(d)
    return norm({o: max(v, EPS) * math.exp(rng.gauss(0, sd)) for o, v in d.items()})


@dataclass
class Card:
    p: dict
    dist: dict
    conf_f: dict
    conf_t: dict


def known_card(dz: Disease, fam: Family, sd: float, rng) -> Card:
    return Card({f: noisy_p(v, sd, rng) for f, v in dz.p.items()},
                {t: noisy_dist(v, sd, rng) for t, v in dz.dist.items()},
                {f: 0.9 for f in dz.p}, {t: 0.9 for t in dz.dist})


def quantize(p):
    return min(FREQ_LEVELS, key=lambda l: abs(math.log(FREQ[l]) - math.log(max(p, 1e-4))))


def drafted_card(dz: Disease, fam: Family, sd: float, k: int, grounded: float, rng) -> Card:
    """LLM drafting model: K noisy samples, frequency levels, median aggregation, >= half kept."""
    samples = []
    for _ in range(k):
        sp, st = {}, {}
        for f, v in dz.p.items():
            if rng.random() < 0.25:
                continue
            e = sd / 3 if rng.random() < grounded else sd
            sp[f] = quantize(noisy_p(v, e, rng))
        if rng.random() < 0.3:  # hallucinated entry
            f = rng.choice(list(fam.bg))
            sp.setdefault(f, rng.choice(["common", "occasional"]))
        for t, dd in dz.dist.items():
            if rng.random() < 0.2:
                continue
            e = sd / 3 if rng.random() < grounded else sd
            st[t] = {o: FREQ[quantize(v)] for o, v in noisy_dist(dd, e, rng).items()}
        samples.append((sp, st))
    need = math.ceil(k / 2)
    p, conf_f, dist, conf_t = {}, {}, {}, {}
    for f in sorted({f for sp, _ in samples for f in sp}):
        levels = [sp[f] for sp, _ in samples if f in sp]
        if len(levels) < need:
            continue
        idx = sorted(FREQ_LEVELS.index(l) for l in levels)
        med = FREQ_LEVELS[statistics.median_low(idx)]
        agree = sum(abs(FREQ_LEVELS.index(l) - FREQ_LEVELS.index(med)) <= 1 for l in levels) / k
        p[f] = FREQ[med]
        conf_f[f] = 0.55 if k == 1 else 0.4 + 0.5 * agree
    for t in sorted({t for _, st in samples for t in st}):
        rows = [st[t] for _, st in samples if t in st]
        if len(rows) < need:
            continue
        dist[t] = norm({o: sum(r[o] for r in rows) / len(rows) for o in rows[0]})
        conf_t[t] = 0.55 if k == 1 else 0.4 + 0.5 * len(rows) / k
    return Card(p, dist, conf_f, conf_t)


def agent_world(fam: Family, cards: dict, priors: dict, bg_sd: float, rng_bg) -> World:
    """Assemble the agent's view as a World for the shared engine (Rep, posterior)."""
    ids = list(cards)
    return World(
        diseases=ids, prior=priors,
        p={d: cards[d].p for d in ids},
        bg=fam.bg, src=fam.src, cluster=fam.cluster,
        dist={d: {t: {o: dd.get(o, EPS) for o in fam.outcomes[t]} for t, dd in cards[d].dist.items()} for d in ids},
        bgt=fam.bgt, outcomes=fam.outcomes, ttype=fam.ttype,
        tcost={t: COSTS[fam.ttype[t]] for t in fam.outcomes}, trisk={t: RISKS[fam.ttype[t]] for t in fam.outcomes},
        cant_miss={}, names={d: d for d in ids},
    )


class ConfRep(Rep):
    """Rep (frequency kind) with per-entry confidence instead of a single shrink constant."""

    def __init__(self, K: World, cards: dict):
        super().__init__(K, "freq")
        self.cards = cards

    def p_find(self, d, f):
        K = self.K
        if f not in K.p[d]:
            return K.bg[f]
        c = self.cards[d].conf_f.get(f, 0.9)
        return clamp(c * K.p[d][f] + (1 - c) * K.bg[f])

    def dist_eff(self, d, t):
        K = self.K
        if t not in K.dist[d]:
            return K.bgt[t]
        c = self.cards[d].conf_t.get(t, 0.9)
        dd = K.dist[d][t]
        return {o: c * dd[o] + (1 - c) * K.bgt[t][o] for o in dd}


# ============================================================================== cases

@dataclass
class Case:
    idx: int
    fam: int
    disease: str
    frep: dict
    trep: dict
    intake: list
    propose_u: float  # uniform draw: unknown truth proposed by the generator iff < recall


def sample_case(fams: list[Family], idx: int, seed: int) -> Case:
    rng = random.Random(seed * 7919 + idx)
    fam = fams[rng.randrange(len(fams))]
    dz = rng.choices(fam.diseases, weights=[d.prior for d in fam.diseases])[0]
    frep = {}
    for f in fam.bg:
        true = rng.random() < dz.p.get(f, fam.bg[f])
        frep[f] = (not true) if rng.random() < FLIP[fam.src[f]] else true
    trep = {}
    for t in fam.outcomes:
        dd = dz.dist.get(t, fam.bgt[t])
        o = rng.choices(list(dd), weights=list(dd.values()))[0]
        if rng.random() < TEST_NOISE:
            o = rng.choices(list(fam.bgt[t]), weights=list(fam.bgt[t].values()))[0]
        trep[t] = o
    hist = [f for f in fam.bg if fam.src[f] == "history" and frep[f]]
    rng.shuffle(hist)
    exams = [f for f in fam.bg if fam.src[f] == "exam"]
    intake = [(1, hist[:3]), (2, hist[3:5]), (3, rng.sample(exams, 3))]
    return Case(idx, fam.idx, dz.id, frep, trep, intake, rng.random())


# ============================================================================== mechanisms

@dataclass
class Mech:
    name: str
    belief: str = "bayes"        # bayes | additive
    planner: str = "v02"         # v02 | maidxo | graphdx | amie
    safety_weight: float = 0.0   # >0: harm-weighted pair weights
    cantmiss_rule: bool = False
    history_only: bool = False


def harm_loss(di: Disease, dj: Disease) -> float:
    ov = len(di.treatments & dj.treatments) / len(di.treatments | dj.treatments)
    return MISS_HARM[dj.course] * URGENCY[dj.urgency] * (1 - ov) + OVERTREAT_HARM[di.treat_risk]


def pair_w(ids, dz_by_id, sw):
    n = len(ids)
    w = [[1.0] * n for _ in range(n)]
    if sw <= 0:
        return w
    for a in range(n):
        for b in range(a + 1, n):
            da, db = dz_by_id.get(ids[a]), dz_by_id.get(ids[b])
            if da and db:
                w[a][b] = w[b][a] = 1 + sw * 0.5 * (harm_loss(da, db) + harm_loss(db, da))
    return w


def confusion(probs, other, w):
    tot = 0.0
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            tot += w[i][j] * math.sqrt(probs[i] * probs[j])
        tot += math.sqrt(probs[i] * other)
    return tot


def ent(ps):
    z = sum(ps)
    return -sum((p / z) * math.log(p / z) for p in ps if p > 0) if z > 0 else 0.0


def posterior(K: World, R: Rep, obs):
    ds = K.diseases
    prior_sum = sum(K.prior[d] for d in ds)
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
    return {d: math.exp(v - m) / z for d, v in ls.items()}, math.exp(lo - m) / z


TYP = [(0.8, 3), (0.5, 2), (0.25, 1), (0.1, 0)]


def typ(p):
    for thr, v in TYP:
        if p >= thr:
            return v
    return -1


def additive_scores(K: World, obs):
    """GraphDx-style evidence scoring: quantized typicality, summed; softmax to a pseudo-probability."""
    sc = {}
    for d in K.diseases:
        s = math.log(K.prior[d])
        for kind, key, val, _ in obs:
            if kind == "f":
                if key in K.p[d]:
                    t = typ(K.p[d][key])
                    s += t if val else -0.5 * max(t, 0) + (1.0 if t < 0 else 0.0)
            else:
                if key in K.dist[d]:
                    s += typ(K.dist[d][key][val])
        sc[d] = s
    m = max(sc.values())
    z = sum(math.exp(v - m) for v in sc.values())
    return {d: math.exp(v - m) / z for d, v in sc.items()}, 0.0


def outcome_liks(K, R, ids, action):
    kind = action[0]
    out = []
    if kind == "f":
        codes, typ_ = action[1], action[2]
        r = CHANNEL_RELIABILITY[typ_]
        nz = lambda p: r * p + (1 - r) * (1 - p)
        pd = {c: [nz(R.p_find(d, c)) for d in ids] for c in codes}
        po = {c: nz(K.bg[c]) for c in codes}
        for pat in itertools.product([True, False], repeat=len(codes)):
            liks = [1.0] * len(ids)
            lo = 1.0
            for c, pres in zip(codes, pat):
                liks = [l * (pd[c][k] if pres else 1 - pd[c][k]) for k, l in enumerate(liks)]
                lo *= po[c] if pres else 1 - po[c]
            out.append((liks, lo))
    else:
        t, typ_ = action[1], action[2]
        r = CHANNEL_RELIABILITY[typ_]
        dists = [R.dist_eff(d, t) for d in ids]
        for o in K.outcomes[t]:
            out.append(([r * dd[o] + (1 - r) * K.bgt[t][o] for dd in dists], K.bgt[t][o]))
    return out


def action_cost(K, a):
    if a[0] == "f":
        return (0.0 if a[2] == "history" else 5.0), 0.0
    return K.tcost[a[1]], K.trisk[a[1]]


def value_v02(K, R, ids, probs, other, a, w):
    base_err, base_c = 1 - max(probs), confusion(probs, other, w)
    ee = ec = 0.0
    for liks, lo in outcome_liks(K, R, ids, a):
        joint = [p * l for p, l in zip(probs, liks)]
        jo = other * lo
        z = sum(joint) + jo
        if z <= 0:
            continue
        post = [j / z for j in joint]
        ee += z * (1 - max(post))
        ec += z * confusion(post, jo / z, w)
    cost, risk = action_cost(K, a)
    return S.misdiagnosis_loss * (base_err - ee) + S.pair_weight * (base_c - ec) - S.cost_weight * cost - risk


def value_mai(K, R, ids, probs, other, a):
    """Test-Chooser: expected entropy reduction among the top-3 (renormalized); Stewardship veto."""
    top = sorted(range(len(ids)), key=lambda i: -probs[i])[:3]
    base = ent([probs[i] for i in top])
    exp_h = 0.0
    for liks, lo in outcome_liks(K, R, ids, a):
        joint = [probs[i] * liks[i] for i in top]
        z = sum(joint)
        zall = sum(p * l for p, l in zip(probs, liks)) + other * lo
        if z <= 0:
            continue
        exp_h += zall * ent(joint)
    ig = base - exp_h
    cost, _ = action_cost(K, a)
    if cost > 300 and ig < 0.15:
        return -1.0  # Stewardship veto of low-yield expensive tests
    return ig - cost / 5000.0


def value_graphdx(K, ids, probs, a):
    """Cost-aware discriminative coverage of the top-3 by quantized typicality."""
    top = sorted(range(len(ids)), key=lambda i: -probs[i])[:3]
    cov = 0.0
    for i, j in itertools.combinations(top, 2):
        di, dj = ids[i], ids[j]
        if a[0] == "f":
            f = a[1][0]
            ti = typ(K.p[di][f]) if f in K.p[di] else typ(K.bg[f])
            tj = typ(K.p[dj][f]) if f in K.p[dj] else typ(K.bg[f])
            cov += abs(ti - tj)
        else:
            t = a[1]
            qi = K.dist[di].get(t, K.bgt[t])
            qj = K.dist[dj].get(t, K.bgt[t])
            cov += max(abs(typ(qi[o]) - typ(qj[o])) for o in K.outcomes[t])
    cost, risk = action_cost(K, a)
    return cov / (1.0 + cost / 200.0) - risk - 0.05


def run(mech: Mech, K: World, R: Rep, case: Case, dz_by_id: dict) -> dict:
    obs, asked, done = [], set(), set()
    turn, cost, stall = 0, 0.0, 0
    n_inv = n_adv = 0
    reason = ""
    for t_i, codes in case.intake:
        turn = t_i
        if mech.history_only and t_i == 3:
            continue
        for f in codes:
            obs.append(("f", f, case.frep[f], REPORT_CONF[K.src[f]]))
            asked.add(f)
        if t_i == 3:
            cost += 5.0
    while True:
        b, other = additive_scores(K, obs) if mech.belief == "additive" else posterior(K, R, obs)
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
            if mech.history_only and K.src[f] != "history":
                continue
            acts.append(("f", [f], K.src[f]))
        if not mech.history_only:
            for t in K.outcomes:
                if t in done or not any(t in K.dist[d] for d in ids):
                    continue
                gate = {"advanced_imaging": 0.05, "procedure": 0.25}.get(K.ttype[t])
                if gate and mech.planner in ("v02",):
                    tp = max([b[d] for d in ids if t in K.dist[d] and any(
                        R.dist_eff(d, t)[o] >= 2 * K.bgt[t][o] for o in K.outcomes[t])] or [0.0])
                    if tp < gate:
                        continue
                acts.append(("t", t, K.ttype[t]))
        if mech.planner in ("v02", "amie"):
            w = pair_w(ids, dz_by_id, mech.safety_weight)
            scored = sorted(((value_v02(K, R, ids, probs, other, a, w), a) for a in acts), key=lambda x: -x[0])
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
        elif mech.planner == "maidxo":
            scored = sorted(((value_mai(K, R, ids, probs, other, a), a) for a in acts), key=lambda x: -x[0])
            batch = [(v, a) for v, a in scored if v > 0][:3]
            options = [(sum(v for v, _ in batch), [a for _, a in batch])] if batch else []
        else:  # graphdx
            scored = sorted(((value_graphdx(K, ids, probs, a), a) for a in acts), key=lambda x: -x[0])
            options = [(scored[0][0], [scored[0][1]])] if scored and scored[0][0] > 0 else []
        best = options[0][0] if options else 0.0
        can = turn >= S.min_turns_before_commit
        cm_live = any(dz_by_id[d].cant_miss and b[d] >= S.exclusion_threshold for d in ids[1:] if d in dz_by_id)
        if can and leader >= S.commit_threshold and not (mech.cantmiss_rule and cm_live):
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
            continue
        for a in options[0][1]:
            if a[0] == "f":
                f = a[1][0]
                obs.append(("f", f, case.frep[f], REPORT_CONF[K.src[f]]))
                asked.add(f)
                cost += 0.0 if a[2] == "history" else 5.0
            else:
                t = a[1]
                obs.append(("t", t, case.trep[t], 0.97))
                done.add(t)
                cost += K.tcost[t]
                n_inv += K.ttype[t] == "procedure"
                n_adv += K.ttype[t] == "advanced_imaging"
    b, other = additive_scores(K, obs) if mech.belief == "additive" else posterior(K, R, obs)
    ranked = sorted(b, key=b.get, reverse=True)
    truth = case.disease
    top1 = ranked[0]
    dz_t, dz_1 = dz_by_id[truth], dz_by_id.get(top1)
    harm = 0.0 if top1 == truth else (harm_loss(dz_1, dz_t) if dz_1 else MISS_HARM[dz_t.course] * URGENCY[dz_t.urgency])
    return {"mech": mech.name, "idx": case.idx, "truth": truth, "top1": top1, "p_top1": b[top1],
            "p_truth": b.get(truth, 0.0), "in_list": truth in b, "rank": (ranked.index(truth) + 1) if truth in b else 99,
            "turns": turn + 1, "cost": cost, "reason": reason, "procedures": n_inv, "adv_imaging": n_adv,
            "cant_miss_truth": dz_t.cant_miss, "harm": harm}


# ============================================================================== experiment driver

def build_agent(fam: Family, known: dict, drafted: dict, case: Case, recall: float, draft_on: bool):
    """Hypotheses = known diseases of the family (+ an unknown one if drafting is on and the generator proposes it)."""
    cards = {d: c for d, c in known.items() if d.startswith(f"F{fam.idx}_")}
    priors = {}
    for dz in fam.diseases:
        if dz.id in cards:
            priors[dz.id] = min(PRIOR_BUCKETS, key=lambda b: abs(math.log(b) - math.log(dz.prior)))
    if draft_on:
        for dz in fam.diseases:
            if dz.id not in drafted:
                continue
            if dz.id == case.disease:
                proposed = case.propose_u < recall          # generator recall for the true unknown disease
            else:  # plausible unknown alternatives are proposed half the time
                proposed = random.Random(zlib.crc32(f"{case.idx}:{dz.id}".encode())).random() < 0.5
            if proposed:
                cards[dz.id] = drafted[dz.id]
                priors[dz.id] = min(PRIOR_BUCKETS, key=lambda b: abs(math.log(b) - math.log(dz.prior)))
    K = agent_world(fam, cards, priors, 0, None)
    return K, ConfRep(K, cards)


MECHS = {
    "v02": Mech("v02"),
    "maidxo": Mech("maidxo", planner="maidxo"),
    "graphdx": Mech("graphdx", belief="additive", planner="graphdx"),
    "amie": Mech("amie", planner="amie", history_only=True),
}
HARM_MECHS = {f"v02_s{s}{'_cm' if cm else ''}": Mech(f"v02_s{s}{'_cm' if cm else ''}", safety_weight=s, cantmiss_rule=cm)
              for s in (0, 0.5, 1, 2, 4) for cm in (False, True)}


def experiment(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for w in range(args.worlds):
        seed = 1000 + w
        fams = gen_world(seed)
        dz_by_id = {d.id: d for f in fams for d in f.diseases}
        rng_k = random.Random(seed + 1)
        all_ids = sorted(dz_by_id)
        unknown = set(random.Random(seed + 2).sample(all_ids, int(len(all_ids) * args.unknown_frac)))
        known = {d: known_card(dz_by_id[d], fams[dz_by_id[d].family], args.knowledge_sd, rng_k) for d in all_ids}
        known_only = {d: c for d, c in known.items() if d not in unknown}
        cases = [sample_case(fams, i, seed) for i in range(args.cases)]
        t0 = time.time()
        # 1) architectures and 2) harm sweep: full knowledge (every disease known)
        for name, mech in {**MECHS, **HARM_MECHS}.items():
            if args.stage == "drafting":
                break
            for c in cases:
                fam = fams[c.fam]
                K, R = build_agent(fam, known, {}, c, 1.0, False)
                r = run(mech, K, R, c, dz_by_id)
                r.update({"world": w, "exp": "arch" if name in MECHS else "harm"})
                rows.append(r)
        # 3) live drafting on the unknown diseases
        if args.stage in ("drafting", "all"):
            configs = [("no_drafting", None)] + [(f"draft_sd{sd}_k{k}_g{g}", (sd, k, g))
                                                 for sd in (0.5, 1.0, 1.5) for k in (1, 3) for g in (0.0, 0.5)]
            configs.append(("oracle_card", "oracle"))
            for name, cfg in configs:
                if cfg is None:
                    drafted, on = {}, False
                elif cfg == "oracle":
                    drafted, on = {d: known[d] for d in unknown}, True
                else:
                    sd, k, g = cfg
                    rng_d = random.Random(seed + zlib.crc32(name.encode()))
                    drafted = {d: drafted_card(dz_by_id[d], fams[dz_by_id[d].family], sd, k, g, rng_d) for d in unknown}
                    on = True
                for c in cases:
                    fam = fams[c.fam]
                    K, R = build_agent(fam, known_only, drafted, c, args.recall, on)
                    r = run(MECHS["v02"], K, R, c, dz_by_id)
                    r.update({"world": w, "exp": "draft", "mech": name, "unknown_truth": c.disease in unknown})
                    rows.append(r)
        print(f"world {w}: {len(rows)} rows so far, {time.time() - t0:.0f}s", flush=True)
    with open(out / f"rows_{args.stage}.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


# ============================================================================== analysis

def summarize(rs):
    n = len(rs)
    k = sum(r["top1"] == r["truth"] for r in rs)
    cm = [r for r in rs if r["cant_miss_truth"]]
    return {"n": n, "top1": k / n, "ci": wilson(k, n), "top3": sum(r["rank"] <= 3 for r in rs) / n,
            "turns": sum(r["turns"] for r in rs) / n, "cost": sum(r["cost"] for r in rs) / n,
            "brier": sum((r["p_top1"] - (r["top1"] == r["truth"])) ** 2 for r in rs) / n,
            "cm_n": len(cm), "cm_missed": sum(r["top1"] != r["truth"] for r in cm) / max(1, len(cm)),
            "harm": sum(r["harm"] for r in rs) / n, "invasive": sum(r["procedures"] > 0 for r in rs) / n,
            "adv": sum(r["adv_imaging"] > 0 for r in rs) / n,
            "commit_at_budget": sum(r["reason"] == "budget" for r in rs) / n}


def paired(ra, rb, key="top1"):
    a = {(r["world"], r["idx"]): r for r in ra}
    b = {(r["world"], r["idx"]): r for r in rb}
    ids = sorted(set(a) & set(b))
    ca = [a[i]["top1"] == a[i]["truth"] for i in ids]
    cb = [b[i]["top1"] == b[i]["truth"] for i in ids]
    b01 = sum((not x) and y for x, y in zip(ca, cb))
    b10 = sum(x and not y for x, y in zip(ca, cb))
    return {"acc": boot_diff([float(y) for y in cb], [float(x) for x in ca]), "b_better": b01, "a_better": b10,
            "p": mcnemar_exact(b01, b10),
            "cost": boot_diff([b[i]["cost"] for i in ids], [a[i]["cost"] for i in ids]),
            "turns": boot_diff([b[i]["turns"] for i in ids], [a[i]["turns"] for i in ids]),
            "harm": boot_diff([b[i]["harm"] for i in ids], [a[i]["harm"] for i in ids]),
            "cm_missed": boot_diff([float(b[i]["top1"] != b[i]["truth"]) for i in ids if a[i]["cant_miss_truth"]],
                                   [float(a[i]["top1"] != a[i]["truth"]) for i in ids if a[i]["cant_miss_truth"]])}


def analyze(args):
    out = Path(args.out)
    rows = []
    for p in sorted(out.glob("rows_*.jsonl")):
        rows += [json.loads(l) for l in open(p)]
    res = {"arch": {}, "harm": {}, "draft": {}}
    by = {}
    for r in rows:
        by.setdefault((r["exp"], r["mech"]), []).append(r)
    for (exp, m), rs in by.items():
        res[exp][m] = {"summary": summarize(rs)}
    for m in MECHS:
        if m != "v02" and ("arch", m) in by:
            res["arch"][m]["paired_vs_v02"] = paired(by[("arch", m)], by[("arch", "v02")])
    for m in HARM_MECHS:
        if m != "v02_s0" and ("harm", m) in by:
            res["harm"][m]["paired_vs_s0"] = paired(by[("harm", "v02_s0")], by[("harm", m)])
    if ("draft", "no_drafting") in by:
        for m in [k for (e, k) in by if e == "draft"]:
            rs = by[("draft", m)]
            unk = [r for r in rs if r["unknown_truth"]]
            kn = [r for r in rs if not r["unknown_truth"]]
            res["draft"][m]["unknown"] = summarize(unk)
            res["draft"][m]["known"] = summarize(kn)
            if m != "no_drafting":
                base = by[("draft", "no_drafting")]
                res["draft"][m]["paired_unknown"] = paired([r for r in base if r["unknown_truth"]], unk)
                res["draft"][m]["paired_known"] = paired([r for r in base if not r["unknown_truth"]], kn)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print("wrote", out / "summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["arch", "drafting", "all", "analyze"], default="all")
    ap.add_argument("--worlds", type=int, default=3)
    ap.add_argument("--cases", type=int, default=800)
    ap.add_argument("--knowledge-sd", type=float, default=0.5, help="log-odds SD of error in known cards")
    ap.add_argument("--unknown-frac", type=float, default=0.2)
    ap.add_argument("--recall", type=float, default=0.8, help="hypothesis-generator recall for unknown diseases")
    ap.add_argument("--out", default="runs/world_bench")
    args = ap.parse_args()
    if args.stage == "analyze":
        analyze(args)
    else:
        experiment(args)


if __name__ == "__main__":
    main()
