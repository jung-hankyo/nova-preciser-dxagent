"""v0.4 held-out check: the REAL v0.4 graph nodes on the SAME simulated worlds, patients and LLM roles as v03_sim.py.

Everything except knowledge is imported unchanged from v03_sim.py (world generator, case sampler, patient,
extraction, generator, prior assessor, critic, report), so a v0.4 case and a v0.3 case with the same (world, index)
see the same patient and the same answers to the same questions: a paired design across package versions.

Knowledge model (assumptions, stated and swept)
    v0.3 (for reference): 75% of diseases pre-built (log-odds error SD 0.5, confidence 0.9, every entry present);
    the other 25% drafted live (SD 0.7, 25% of entries dropped, a hallucinated entry 30%, synonym codes 30%).

    v0.4 compiled cards (this harness), for the diseases the sources cover:
      COV       which diseases have a compiled card: "known" = the same 75% as v0.3's pre-built set (the offline
                default with no source for the rest), "all" = every disease
      F_STATED  a finding entry exists only if some source states it (0.70); tests T_STATED (0.85)
      MIX       how each stated value was expressed: tier-1 numeric 45%, tier-1 frequency word 15%, tier-2 class 25%,
                tier-2 numeric 15%
      SD_SRC    a source's value differs from the true frequency (study population, spectrum): log-odds SD 0.3;
                frequency words and classes are additionally quantized to the compiler's fixed values
      MISATTR   residual verified-but-misattributed entries (a value that belongs to a sibling disease): 2%
      DIRECT    population mismatch or unknown on 30% of entries (directness 0.9)
      confidence: the compiler's tier table (sources.entry_conf), not agreement
      harm: stated by a source for 90% of cards; prevalence: never (the compiler extracts none)
      No hallucinated entries (verification) and no synonym codes (registry/ontology codes).
    Live fallback (arm "+fallback"): an uncovered disease reaching the top of the list gets a provisional card
      from abstracts: findings stated 40%, tests 50%, SD 0.5, tier 3, confidence capped at 0.5; never in the reverse
      index. Five extraction calls per compiled disease.
    Contrast cards: off in every arm, as in v03_sim. The ontology tree is inactive (the worlds have no hierarchy).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import v03_sim as V3  # noqa: E402  (world, patient and roles, unchanged)
from v03_sim import P, Case, SimWorld, build_world, fam_prior, metrics, noisy, sample_case, u  # noqa: E402

from dxagent.config import Settings  # noqa: E402
from dxagent.graph.nodes import Nodes  # noqa: E402
from dxagent.knowledge.cards import (CardStore, DiseaseCard, EntryProvenance, FindingDef, FindingEntry,  # noqa: E402
                                     HarmProfile, Provenance, TestDef, TestEntry, normalize)
from dxagent.knowledge.compiler.claims import HPO_FREQUENCY  # noqa: E402
from dxagent.knowledge.compiler.sources import entry_conf  # noqa: E402

import world_bench as WB  # noqa: E402

C = dict(F_STATED=0.70, T_STATED=0.85, SD_SRC=0.3, MISATTR=0.02, DIRECT_UNK=0.30, HARM_STATED=0.90,
         FB_F=0.40, FB_T=0.50, FB_SD=0.5, FB_CAP=0.5, FB_CALLS=5,
         PREV=0.0,  # ablation: 1 = compiled cards carry the true complaint-keyed prevalence (as v0.3 pre-built cards did)
         MIXNUM=0.0)  # ablation: 1 = every stated value is tier-1 numeric (no quantization)
MIX = [(1, "numeric", 0.45), (1, "qualitative", 0.15), (2, "class", 0.25), (2, "numeric", 0.15)]
QUAL_VALUES = [0.95, 0.80, 0.60, 0.30, 0.15, 0.05, 0.01]  # claims.QUALITATIVE values
CLASS_BINS = [(0.80, HPO_FREQUENCY["HP:0040281"]), (0.30, HPO_FREQUENCY["HP:0040282"]), (0.05, HPO_FREQUENCY["HP:0040283"]),
              (0.0, HPO_FREQUENCY["HP:0040284"])]


def _pick_mix(*key) -> tuple[int, str]:
    r, acc = u(*key), 0.0
    for tier, prec, share in MIX:
        acc += share
        if r <= acc:
            return tier, prec
    return MIX[-1][0], MIX[-1][1]


def _express(p: float, prec: str, *key) -> float:
    """The value the compiler would compute from a source that reports p with study error."""
    q = noisy(p, C["SD_SRC"], *key)
    if prec == "qualitative":
        return min(QUAL_VALUES, key=lambda v: abs(math.log(v / (1 - v)) - math.log(q / (1 - q))))
    if prec == "class":
        return next(v for lo, v in CLASS_BINS if q >= lo)
    return q


def compiled_card(W: SimWorld, d: str, status: str = "compiled", fallback: bool = False) -> DiseaseCard:
    D, F = W.dz[d], W.fam[d]
    sibs = [x.id.lower() for x in F.diseases if x.id.lower() != d]
    fs = []
    for c0, v in sorted(D.p.items()):
        if u(W.seed, "c04_stated", d, c0, fallback) >= (C["FB_F"] if fallback else C["F_STATED"]):
            continue
        if fallback:
            tier, prec = 3, "numeric"
            p = noisy(v, C["FB_SD"], W.seed, "c04_fb", d, c0)
        else:
            tier, prec = (1, "numeric") if C["MIXNUM"] else _pick_mix(W.seed, "c04_mix", d, c0)
            src = v
            if u(W.seed, "c04_misattr", d, c0) < C["MISATTR"] and sibs:
                o = sibs[int(u(W.seed, "c04_mpick", d, c0) * len(sibs))]
                src = W.dz[o].p.get(c0, F.bg[c0])
            p = _express(src, prec, W.seed, "c04_val", d, c0)
        direct = 0.9 if u(W.seed, "c04_dir", d, c0) < C["DIRECT_UNK"] else 1.0
        conf = entry_conf(tier, prec, direct)
        if fallback:
            conf = min(conf, C["FB_CAP"])
        sc = W.code[c0]
        fs.append(FindingEntry(code=sc, label=W.label[sc], source=F.src[c0], p=p, conf=conf,
                               prov=EntryProvenance(source_id="sim", tier=tier, precision=prec, directness=direct)))
    te = {}
    for t, dd in D.dist.items():
        if u(W.seed, "c04_tstated", d, t, fallback) >= (C["FB_T"] if fallback else C["T_STATED"]):
            continue
        sd = C["FB_SD"] if fallback else C["SD_SRC"]
        dist = normalize({o: max(val, 1e-3) * math.exp(sd * WB_norm(W.seed, "c04_t", d, t, o, fallback)) for o, val in dd.items()})
        conf = min(entry_conf(3, "numeric", 1.0), C["FB_CAP"]) if fallback else entry_conf(1, "numeric", 1.0)
        te[t.lower()] = TestEntry(dist=dist, conf=conf)
    harm = HarmProfile()
    if not fallback and u(W.seed, "c04_harm", d) < C["HARM_STATED"]:
        harm = HarmProfile(untreated_course=D.course, urgency=D.urgency, treatments=sorted(D.treatments),
                           treatment_risk=D.treat_risk, evidence="sim")
    prev = {}
    if C["PREV"] and not fallback:
        b = V3.bucket(fam_prior(W, d))
        prev = {"prevalence": b, "prevalence_by_complaint": {W.cc[F.idx]: b}}
    return DiseaseCard(id=d, name=d, harm=harm, findings=fs, tests=te, **prev,
                       provenance=Provenance(status=status, sources=[{"id": "sim", "kind": "abstract" if fallback else "guideline"}]))


def WB_norm(*key) -> float:
    import statistics

    return statistics.NormalDist().inv_cdf(min(max(u(*key), 1e-9), 1 - 1e-9))


def make_store_v04(W: SimWorld, cov: str) -> CardStore:
    st = CardStore(Path("/nonexistent_registry"), None)
    fdefs, tdefs = [], []
    for F in W.fams:  # registries exactly as in v03_sim.make_store (true backgrounds, curated)
        for c in F.bg:
            sc = W.code[c]
            fdefs.append(FindingDef(code=sc, label=W.label[sc], source=F.src[c], background=F.bg[c],
                                    cluster=W.code.get(F.cluster[c], F.cluster[c]) if F.cluster[c] in W.code else f"cl_{F.cluster[c]}"))
        fdefs.append(FindingDef(code=W.cc[F.idx], label=W.label[W.cc[F.idx]], source="history", background=0.99))
        for t, outs in F.outcomes.items():
            tdefs.append(TestDef(id=t.lower(), name=t.lower().replace("_", " "), type=F.ttype[t], outcomes=outs, background=F.bgt[t],
                                 cost=WB.COSTS[F.ttype[t]], risk=WB.RISKS[F.ttype[t]]))
    st._findings = {f.code: f for f in fdefs}
    st._tests = {t.id: t for t in tdefs}
    covered = set(W.dz) if cov == "all" else set(W.known)
    for d in sorted(covered):
        st._register(compiled_card(W, d))
    st._lexicon = None
    st._by_code = None
    st._fmaps.clear()
    return st


class SimFallback:
    """Live fallback stand-in: a provisional card from abstracts for a resolved disease without knowledge."""

    def __init__(self, W: SimWorld, store: CardStore, sim):
        self.W, self.store, self.sim = W, store, sim

    def compile_one(self, disease_id: str, curie: str):
        if disease_id not in self.W.dz or self.store.get(disease_id) is not None:
            return None
        self.sim.calls["fallback_extract"] += C["FB_CALLS"]
        return self.store.add(compiled_card(self.W, disease_id, status="provisional", fallback=True), persist=False)


def arm_settings(arm: str) -> Settings:
    s = Settings()  # v0.4 defaults = v0.3.1 shipped flags (A1-A3, C3, D2, E1, F1, P1 on)
    s.contrast_cards = False  # off in every arm, as in v03_sim
    s.live_fallback = "+fallback" in arm
    return s


def run_case(W: SimWorld, store: CardStore, case: Case, arm: str, s: Settings) -> dict:
    sim = V3.Sim(W, case, arm)
    sim.store = store
    llm = sim.llm()

    class Rt:
        pass

    rt = Rt()
    rt.settings, rt.doctor, rt.store, rt.contrast = s, llm, store, None
    rt.fallback = SimFallback(W, store, sim) if s.live_fallback else None
    n = Nodes(rt)
    if rt.fallback is not None:  # the worlds have no ontology: a world disease id stands in for its ontology id
        orig = n.load_knowledge

        def load_knowledge(state):
            for h in state.get("hypotheses", []):
                if h["disease_id"] in W.dz:
                    h.setdefault("ontology_id", h["disease_id"])
                    h["ontology_id"] = h["ontology_id"] or h["disease_id"]
            return orig(state)

        n.load_knowledge = load_knowledge
    state = {"presentation": "", "presentation_source": "referral", "turn": 0, "phase": "gathering", "findings": [],
             "test_results": [], "hypotheses": [], "last_actions": [], "context": {}, "transcript": []}
    state["last_reply"] = state["presentation"] = sim.presentation()
    node, steps = "perceive", 0
    nxt = {"perceive": "hypothesize", "hypothesize": "load_knowledge", "load_knowledge": "update_belief", "update_belief": "contrast"}
    while True:
        steps += 1
        if steps > 3000:
            raise RuntimeError("runaway loop")
        if node == "await_reply":
            state["last_reply"] = sim.respond(state.get("last_actions", []), state.get("turn", 0))
            node = "perceive"
            continue
        state.update(getattr(n, node)(state) or {})
        if node in nxt:
            node = nxt[node]
        elif node == "contrast":
            node = n.route_after_contrast(state)
        elif node == "critic":
            node = n.route_after_critic(state)
        elif node == "plan":
            node = n.route_after_plan(state)
        elif node == "report":
            node = n.route_after_report(state)
            if node == "end":
                break
        elif node == "interview":
            node = "await_reply"
    diff = state.get("differential", [])
    top = diff[0] if diff else {"disease_id": None, "prob": 0.0}
    p_truth = next((d["prob"] for d in diff if d["disease_id"] == case.d), 0.0)
    final = (state.get("final_report") or {}).get("final_diagnosis", top["disease_id"])
    return {
        "case": case.idx, "truth": case.d, "correct": final == case.d, "top_prob": top["prob"],
        "top_correct": top["disease_id"] == case.d, "p_truth": p_truth, "in_list": any(d["disease_id"] == case.d for d in diff),
        "known": case.d in W.known, "covered": store.get(case.d) is not None and store.get(case.d).provenance.status == "compiled",
        "cant_miss": W.dz[case.d].cant_miss, "turns": state.get("turn", 0), "cost": state.get("cost_spent", 0.0),
        "other": state.get("other_prob", 0.0), "reason": state.get("commit_reason", ""), "calls": dict(sim.calls),
        "coverage_misses": len(state.get("coverage_log", [])),
        "truth_missed_coverage": any(c["disease_id"] == case.d for c in state.get("coverage_log", [])),
        "truth_card": (store.get(case.d).provenance.status if store.get(case.d) else "none"),
        "dissents": len(state.get("report_dissent", [])),
    }


def run_arm(arm: str, seeds: list[int], n_cases: int, log=print) -> list[dict]:
    cov = "all" if "cov100" in arm else "known"
    out = []
    for seed in seeds:
        W = build_world(seed, int(P["N_FAM"]))
        if "v3know" in arm:
            # the v0.4 code on v0.3's exact pre-built cards (the KNOWN share, every entry, SD 0.5, confidence 0.9) and no
            # live drafting: isolates the code changes and the loss of live drafting from the compiled-card model
            store = V3.make_store(W)
            store._by_code = None
            store._fmaps.clear()
        else:
            store = make_store_v04(W, cov)
        s = arm_settings(arm)
        t0 = time.time()
        for i in range(n_cases):
            out.append({**run_case(W, store, sample_case(W, i), arm, s), "seed": seed})
        log(f"  {arm} world {seed}: {n_cases} cases in {time.time() - t0:.0f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["v0.4", "v0.4+fallback", "v0.4+cov100"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
    ap.add_argument("--cases", type=int, default=500)
    ap.add_argument("--set", nargs="*", default=[], help="override knowledge assumptions, e.g. F_STATED=0.5 SD_SRC=0.5")
    ap.add_argument("--out", default="experiments/results/v04_heldout")
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("=")
        (C if k in C else P)[k] = float(v)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "_".join(a.set) or "default"
    for arm in a.arms:
        rs = run_arm(arm, a.seeds, a.cases)
        (out / f"{arm}__{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in rs))
        m = metrics(rs)
        print(f"{arm:14s} top1 {m['top1']:.3f} in_list {m['in_list']:.3f} brier {m['brier']:.3f} ece {m['ece']:.3f} "
              f"logloss {m['logloss']:.3f} turns {m['turns']:.2f} cost {m['cost']:.0f} llm {m['llm_calls']:.1f} cm_missed {m['cm_missed']:.3f}")


if __name__ == "__main__":
    main()
