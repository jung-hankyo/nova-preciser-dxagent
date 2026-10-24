"""Observation-model experiment: unified report likelihood vs shipped v0.2, on persona patients.

Modes (belief AND planner use the same model within each mode where stated)
    temper   shipped v0.2: belief LLR = c * log(P(f|d)/P(f|bg)) (reports treated as noise-free, damped by
             extraction confidence c); planner predicts noisy reports with channel reliability r.
    double   v0.2.1 patch: channel noise r inside the belief likelihood AND the c damping (discounts twice).
    unified  one likelihood for belief and planner:
               binary finding:  P(report | d) = rho * p + (1 - rho) * (1 - p),  rho = 0.5 + (r - 0.5) * c
               test outcome:    P(o | d)      = lam * P(o|d) + (1 - lam) * bg(o), lam = r * c
             no separate damping; the planner uses the expected confidence per channel.

Patients (real / persona target): each case draws a persona that sets history error rates and how often
the patient hedges (hedging is likelier when the answer is wrong). Extraction confidence is low for hedged
answers and high otherwise; a small rate of confident extraction errors (e.g. missed negation) is added.
Exams have examiner error independent of persona; tests a 3% misreport rate. The agent does NOT know the
persona: it uses fixed channel reliabilities (history 0.85, exam 0.90, tests from config).

Acceptance (fixed before running): gain in accuracy or calibration with no metric worse, where "worse"
means the paired 95% CI excludes zero in the harmful direction. Metrics: top-1, top-3, Brier, ECE,
log loss on the true disease, can't-miss missed, turns, cost.

Run: python experiments/obs_model.py --worlds 3 --cases 1500 --knowledge-sd 0.5 1.0
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import world_bench as WB  # noqa: E402
from dxagent.config import CHANNEL_RELIABILITY, get_settings  # noqa: E402
from lr_vs_freq import boot_diff, mcnemar_exact, wilson  # noqa: E402

S = get_settings()
EXPECTED_CONF = {"history": 0.85, "exam": 0.90}  # the agent's planning assumption (as in v0.2)
TEST_CONF = 0.97

PERSONAS = {  # FN, FP for history answers; P(hedge | correct), P(hedge | wrong)
    "reliable": (0.03, 0.02, 0.05, 0.30),
    "vague": (0.15, 0.12, 0.35, 0.60),
    "minimizer": (0.35, 0.02, 0.10, 0.30),
    "over_reporter": (0.03, 0.25, 0.10, 0.30),
    "low_literacy": (0.20, 0.15, 0.30, 0.50),
}
PERSONA_MIX = {"reliable": 0.40, "vague": 0.15, "minimizer": 0.15, "over_reporter": 0.15, "low_literacy": 0.15}
EXAM_ERR = (0.08, 0.04, 0.10, 0.30)
EXTRACTION_ERR = 0.03  # confident misextraction (e.g. negation missed)


@dataclass
class PCase:
    idx: int
    fam: int
    disease: str
    persona: str
    frep: dict
    fconf: dict
    trep: dict
    intake: list
    ftrue: dict = None
    ttrue: dict = None


def sample_pcase(fams, idx: int, seed: int) -> PCase:
    rng = random.Random(seed * 7919 + idx)
    fam = fams[rng.randrange(len(fams))]
    dz = rng.choices(fam.diseases, weights=[d.prior for d in fam.diseases])[0]
    persona = rng.choices(list(PERSONA_MIX), weights=list(PERSONA_MIX.values()))[0]
    frep, fconf, ftrue, ttrue = {}, {}, {}, {}
    for f in fam.bg:
        true = rng.random() < dz.p.get(f, fam.bg[f])
        fn, fp, h_ok, h_err = PERSONAS[persona] if fam.src[f] == "history" else EXAM_ERR
        wrong = rng.random() < (fn if true else fp)
        rep = (not true) if wrong else true
        hedged = rng.random() < (h_err if wrong else h_ok)
        c = rng.uniform(0.35, 0.70) if hedged else rng.uniform(0.80, 0.98)
        if rng.random() < EXTRACTION_ERR:
            rep, c = (not rep), rng.uniform(0.80, 0.95)
        frep[f], fconf[f], ftrue[f] = rep, c, true
    trep = {}
    for t in fam.outcomes:
        dd = dz.dist.get(t, fam.bgt[t])
        o = rng.choices(list(dd), weights=list(dd.values()))[0]
        ttrue[t] = o
        if rng.random() < 0.03:
            o = rng.choices(list(fam.bgt[t]), weights=list(fam.bgt[t].values()))[0]
        trep[t] = o
    hist = [f for f in fam.bg if fam.src[f] == "history" and frep[f]]
    rng.shuffle(hist)
    exams = [f for f in fam.bg if fam.src[f] == "exam"]
    return PCase(idx, fam.idx, dz.id, persona, frep, fconf, trep, [(1, hist[:3]), (2, hist[3:5]), (3, rng.sample(exams, 3))],
                 ftrue, ttrue)


# ------------------------------------------------------------------------------ observation models

def rho_binary(r: float, c: float) -> float:
    return 0.5 + (r - 0.5) * c


THRESH = {}  # per-mode commit threshold overrides (frontier check)
CAL = {}  # calibrated unified parameters: {"r": {channel: r}, "ec": {channel: E[c]}}; empty = config values


def r_of(channel: str) -> float:
    return CAL.get("r", {}).get(channel, CHANNEL_RELIABILITY[channel])


def ec_of(channel: str) -> float:
    return CAL.get("ec", {}).get(channel, EXPECTED_CONF.get(channel, TEST_CONF))


def calibrate(seed: int = 2000, n: int = 1500) -> dict:
    """Maximum-likelihood channel reliability under the unified model, on a development world."""
    fams = WB.gen_world(seed)
    rows = {"history": [], "exam": []}
    ecs = {"history": [], "exam": []}
    trows = []
    for i in range(n):
        c = sample_pcase(fams, i, seed)
        fam = fams[c.fam]
        for f in fam.bg:
            ch = fam.src[f]
            rows[ch].append((c.frep[f] == c.ftrue[f], c.fconf[f]))
            ecs[ch].append(c.fconf[f])
        for t in fam.outcomes:
            trows.append((c.trep[t] == c.ttrue[t], fam.bgt[t][c.trep[t]]))
    grid = [0.5 + 0.005 * k for k in range(1, 100)]
    out = {"r": {}, "ec": {}}
    for ch, rs in rows.items():
        ll = lambda r: sum(math.log(rho_binary(r, cc) if ok else 1 - rho_binary(r, cc)) for ok, cc in rs)
        out["r"][ch] = max(grid, key=ll)
        out["ec"][ch] = sum(ecs[ch]) / len(ecs[ch])
    # tests: P(report = true outcome) = lam + (1 - lam) * bg(o_true); P(other o) = (1 - lam) * bg(o); lam = r_t * c_t
    llt = lambda r: sum(math.log((r * TEST_CONF + (1 - r * TEST_CONF) * bg) if ok else (1 - r * TEST_CONF) * bg) for ok, bg in trows)
    rt = max(grid, key=llt)
    for ch in ("poc", "lab", "imaging", "advanced_imaging", "procedure"):
        out["r"][ch] = rt
    return out


class ObsRep(WB.ConfRep):
    def __init__(self, K, cards, mode: str):
        super().__init__(K, cards)
        self.mode = mode

    def llr(self, d, obs):
        K = self.K
        kind, key, val, c = obs
        if kind == "f":
            if not self.listed(d, key):
                return []
            p, b = self.p_find(d, key), K.bg[key]
            r = r_of(K.src[key]) if self.mode == "unified" else CHANNEL_RELIABILITY[K.src[key]]
            if self.mode == "temper":
                x = math.log(p / b) if val else math.log((1 - p) / (1 - b))
                return [(K.cluster[key], c * x)]
            rr = r if self.mode == "double" else rho_binary(r, c)
            pr, br = rr * p + (1 - rr) * (1 - p), rr * b + (1 - rr) * (1 - b)
            x = math.log(pr / br) if val else math.log((1 - pr) / (1 - br))
            return [(K.cluster[key], (c if self.mode == "double" else 1.0) * x)]
        t, o = key, val
        if t not in K.dist[d]:
            return []
        p = self.dist_eff(d, t)[o]
        bg = K.bgt[t][o]
        r = r_of(K.ttype[t]) if self.mode == "unified" else CHANNEL_RELIABILITY[K.ttype[t]]
        if self.mode == "temper":
            return [(f"test:{t}", c * math.log(p / bg))]
        lam = r if self.mode == "double" else r * c
        return [(f"test:{t}", (c if self.mode == "double" else 1.0) * math.log((lam * p + (1 - lam) * bg) / bg))]


def make_outcome_liks(mode: str):
    """Planner observation model. temper/double: channel r (as shipped). unified: same form as belief, E[c]."""

    def outcome_liks(K, R, ids, action):
        out = []
        if action[0] == "f":
            codes, typ_ = action[1], action[2]
            r = r_of(typ_) if mode == "unified" else CHANNEL_RELIABILITY[typ_]
            rr = rho_binary(r, ec_of(typ_)) if mode == "unified" else r
            nz = lambda p: rr * p + (1 - rr) * (1 - p)
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
            r = r_of(typ_) if mode == "unified" else CHANNEL_RELIABILITY[typ_]
            lam = r * TEST_CONF if mode == "unified" else r
            dists = [R.dist_eff(d, t) for d in ids]
            for o in K.outcomes[t]:
                out.append(([lam * dd[o] + (1 - lam) * K.bgt[t][o] for dd in dists], K.bgt[t][o]))
        return out

    return outcome_liks


# ------------------------------------------------------------------------------ encounter loop

def run(mode: str, K, R, case: PCase, dz_by_id: dict, fixed_conf: bool = False) -> dict:
    """world_bench.run with per-observation confidence (fixed_conf=True reproduces world_bench exactly)."""
    WB.outcome_liks = make_outcome_liks(mode)
    conf = (lambda f: WB.REPORT_CONF[K.src[f]]) if fixed_conf else (lambda f: case.fconf[f])
    obs, asked, done = [], set(), set()
    turn, cost, stall = 0, 0.0, 0
    for t_i, codes in case.intake:
        turn = t_i
        for f in codes:
            obs.append(("f", f, case.frep[f], conf(f)))
            asked.add(f)
        if t_i == 3:
            cost += 5.0
    reason = ""
    while True:
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
        if can and leader >= THRESH.get(mode, S.commit_threshold):
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
                obs.append(("f", f, case.frep[f], conf(f)))
                asked.add(f)
                cost += 0.0 if a[2] == "history" else 5.0
            else:
                t = a[1]
                obs.append(("t", t, case.trep[t], TEST_CONF if not fixed_conf else 0.97))
                done.add(t)
                cost += K.tcost[t]
    b, _ = WB.posterior(K, R, obs)
    ranked = sorted(b, key=b.get, reverse=True)
    dz_t = dz_by_id[case.disease]
    return {"mode": mode, "idx": case.idx, "truth": case.disease, "top1": ranked[0], "p_top1": b[ranked[0]],
            "p_truth": b.get(case.disease, 0.0), "rank": ranked.index(case.disease) + 1 if case.disease in b else 99,
            "turns": turn + 1, "cost": cost, "reason": reason, "cm": dz_t.cant_miss, "persona": getattr(case, "persona", "")}


# ------------------------------------------------------------------------------ validation & stats

def validate(n: int = 200, seed: int = 1000) -> float:
    """mode 'temper' with fixed confidences must reproduce the validated world_bench v0.2 engine exactly."""
    fams = WB.gen_world(seed)
    dz_by_id = {d.id: d for f in fams for d in f.diseases}
    known = {d: WB.known_card(dz_by_id[d], fams[dz_by_id[d].family], 0.5, random.Random(seed + 1)) for d in sorted(dz_by_id)}
    orig = WB.outcome_liks
    agree = 0
    for i in range(n):
        c = WB.sample_case(fams, i, seed)
        K, R0 = WB.build_agent(fams[c.fam], known, {}, c, 1.0, False)
        WB.outcome_liks = orig
        a = WB.run(WB.MECHS["v02"], K, R0, c, dz_by_id)
        R = ObsRep(K, {d: known[d] for d in K.diseases}, "temper")
        pc = PCase(c.idx, c.fam, c.disease, "", c.frep, {}, c.trep, c.intake)
        b = run("temper", K, R, pc, dz_by_id, fixed_conf=True)
        agree += (a["top1"], a["turns"], round(a["cost"], 6)) == (b["top1"], b["turns"], round(b["cost"], 6))
    WB.outcome_liks = orig
    return agree / n


def ece(rows, bins=10):
    tot = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sel = [r for r in rows if lo <= r["p_top1"] < hi or (i == bins - 1 and r["p_top1"] >= 1.0)]
        if sel:
            tot += len(sel) / len(rows) * abs(sum(r["p_top1"] for r in sel) / len(sel) - sum(r["top1"] == r["truth"] for r in sel) / len(sel))
    return tot


def summarize(rs):
    n = len(rs)
    k = sum(r["top1"] == r["truth"] for r in rs)
    cm = [r for r in rs if r["cm"]]
    return {"n": n, "top1": k / n, "ci": wilson(k, n), "top3": sum(r["rank"] <= 3 for r in rs) / n,
            "brier": sum((r["p_top1"] - (r["top1"] == r["truth"])) ** 2 for r in rs) / n, "ece": ece(rs),
            "logloss": -sum(math.log(max(r["p_truth"], 1e-9)) for r in rs) / n,
            "cm_missed": sum(r["top1"] != r["truth"] for r in cm) / max(1, len(cm)),
            "turns": sum(r["turns"] for r in rs) / n, "cost": sum(r["cost"] for r in rs) / n}


def ece_boot(ra, rb, reps=1000, seed=0):
    rng = random.Random(seed)
    n = len(ra)
    diffs = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(ece([rb[i] for i in idx]) - ece([ra[i] for i in idx]))
    diffs.sort()
    return ece(rb) - ece(ra), diffs[int(0.025 * reps)], diffs[int(0.975 * reps)]


def paired(ra, rb):
    """Differences rb - ra with 95% CIs (paired over the same cases)."""
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
    return {
        "top1": f(lambda r: float(r["top1"] == r["truth"])), "mcnemar_p": mcnemar_exact(b01, b10), "b_better": b01, "a_better": b10,
        "top3": f(lambda r: float(r["rank"] <= 3)),
        "brier": f(lambda r: (r["p_top1"] - (r["top1"] == r["truth"])) ** 2),
        "logloss": f(lambda r: -math.log(max(r["p_truth"], 1e-9))),
        "ece": ece_boot(A, B),
        "cm_missed": boot_diff([float(r["top1"] != r["truth"]) for r in cmB], [float(r["top1"] != r["truth"]) for r in cmA]),
        "turns": f(lambda r: r["turns"]), "cost": f(lambda r: r["cost"]),
    }


HIGHER_IS_BETTER = {"top1": True, "top3": True, "brier": False, "logloss": False, "ece": False,
                    "cm_missed": False, "turns": False, "cost": False}


def verdict(p: dict) -> dict:
    """Apply the pre-registered criterion."""
    better, worse = [], []
    for m, hib in HIGHER_IS_BETTER.items():
        est, lo, hi = p[m][0], p[m][1], p[m][2]
        if (hib and lo > 0) or (not hib and hi < 0):
            better.append(m)
        if (hib and hi < 0) or (not hib and lo > 0):
            worse.append(m)
    gain = any(m in better for m in ("top1", "brier", "logloss", "ece"))
    return {"better": better, "worse": worse, "accept": gain and not worse}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=3)
    ap.add_argument("--cases", type=int, default=1500)
    ap.add_argument("--knowledge-sd", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--modes", nargs="+", default=["temper", "unified", "double"])
    ap.add_argument("--out", default="runs/obs_model")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="fit unified r on the dev world (seed 2000) first")
    ap.add_argument("--temper-threshold", type=float, default=None)
    args = ap.parse_args()
    if args.calibrate:
        CAL.update(calibrate())
        print("calibrated on dev world (seed 2000):", json.dumps(CAL))
    if args.temper_threshold:
        THRESH["temper"] = args.temper_threshold
    if args.validate:
        print("temper (fixed confidence) == validated v0.2 engine:", validate())
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    orig = WB.outcome_liks
    for sd in args.knowledge_sd:
        rows = {m: [] for m in args.modes}
        t0 = time.time()
        for w in range(args.worlds):
            seed = 1000 + w
            fams = WB.gen_world(seed)
            dz_by_id = {d.id: d for f in fams for d in f.diseases}
            known = {d: WB.known_card(dz_by_id[d], fams[dz_by_id[d].family], sd, random.Random(seed + 1)) for d in sorted(dz_by_id)}
            cases = [sample_pcase(fams, i, seed) for i in range(args.cases)]
            for m in args.modes:
                for c in cases:
                    fam = fams[c.fam]
                    cards = {d: known[d] for d in known if d.startswith(f"F{fam.idx}_")}
                    K = WB.agent_world(fam, cards, {d: min(WB.PRIOR_BUCKETS, key=lambda b: abs(math.log(b) - math.log(dz_by_id[d].prior))) for d in cards}, 0, None)
                    R = ObsRep(K, cards, m)
                    r = run(m, K, R, c, dz_by_id)
                    r["world"] = w
                    rows[m].append(r)
        WB.outcome_liks = orig
        res = {"summary": {m: summarize(v) for m, v in rows.items()},
               "by_persona": {m: {p: summarize([r for r in v if r["persona"] == p]) for p in PERSONA_MIX} for m, v in rows.items()},
               "paired_vs_temper": {}}
        for m in args.modes:
            if m != "temper":
                p = paired(rows["temper"], rows[m])
                p["verdict"] = verdict(p)
                res["paired_vs_temper"][m] = p
        results[str(sd)] = res
        print(f"knowledge sd {sd}: {time.time() - t0:.0f}s", flush=True)
    results["_config"] = {"calibration": CAL, "thresholds": THRESH}
    (out / "summary.json").write_text(json.dumps(results, indent=1))
    print("wrote", out / "summary.json")


if __name__ == "__main__":
    main()
