# NOTE (v0.4): the v0.3 arms exercise the v0.3.1 live card drafter (KnowledgeService), which v0.4 removed, so they
# run only against the v0.3.1 code. The world, patient and LLM-role models below import cleanly under v0.4 and are
# reused unchanged by v04_sim.py, so v0.4 is tested on the identical held-out cases.
"""v0.3 practical-gain check: the REAL dxagent graph nodes, driven by simulated LLM roles and patients.

What runs for real
    Every graph node (perceive, hypothesize, load_knowledge, update_belief, contrast/gap-fill, critic, plan,
    interview), the belief update, reverse index, planner, knowledge builder (self-consistency, canonical codes,
    background pass, gap fill), lexical retrieval. Node order and routing follow graph/build.py exactly.

What is simulated (with stated, swept assumptions)
    World        world_bench.gen_world: 12 complaint families x 5-8 diseases, confusable sibling pairs, family
                 finding pools with backgrounds and clusters, categorical tests, a confirmatory test per disease.
                 Findings get pseudo-word codes and labels, so lexical retrieval works on text, not on ids.
    Knowledge    75% of diseases have a pre-built card (log-odds error SD 0.5, confidence 0.9); 25% are drafted
                 live from K=2 noisy samples (SD 0.7, entries dropped 25%, a hallucinated entry 30%). Drafts name
                 a finding with a synonym code 30% of the time (similarity mix measured on real synonyms: 10% exact,
                 80% in the LLM-confirmation band, 10% unmatched).
    Patient      answers pre-sampled per case (paired design). Each revealed finding is phrased with the registry's
                 words with probability Q_LEX (held-out lexical recall, 0.79) and otherwise in words retrieval
                 cannot match. Clarity: clear 75% / hedged 20% / inferred 5%, with polarity error 4% / 20% / 35%.
                 A direct question goes unanswered 10% of the time. Presentations list the primary complaint plus
                 0-2 characteristic present findings.
    Extraction   codes a revealed finding correctly iff its code is offered; otherwise it invents a code from the
                 words it heard. Its free-form confidence is overconfident and compressed, as LLM verbal confidence
                 tends to be (clear 0.90, hedged 0.82, inferred 0.78, jitter 0.03).
    Generator    persistent blind spot: the true disease is proposed with probability R (0.75); every other family
                 disease with 0.6. Prior buckets: the true P(d | complaint) (primary-complaint prompt) or the true
                 P(d | complaint + presenting findings) (v0.2 prompt), log-noise SD 0.5, capped at 0.25.
                 Critic: one rescue with probability 0.3 when about to commit and the truth is missing. Veto model
                 (D2, off by default): a wrong leader is vetoed naming the truth w.p. VETO_TP or a wrong alternative
                 w.p. VETO_WRONG; a correct leader is vetoed w.p. VETO_FP; STUBBORN=1 repeats the same verdict.
                 Open-world re-prompt (C1): proposes the missing truth w.p. RHO_OW.
                 "Which diseases cause X?" (C3 discovery): proposes the truth, if it has X, w.p. RHO_DISC.
    Report       (E1) dissents from a wrong computed leader naming the truth w.p. REP_FIX, and from a correct
                 leader naming a wrong alternative w.p. REP_ERR.
    Contrast cards are off in every arm.

    Prior context (v0.3.1, P1, variant "+P1"): the opening statement lists the primary complaint (stated reason
                 for the visit) and the presenting findings (not stated). The S-only prior assessor returns the true
                 P(d | complaint) with log-odds noise: a shared component (SD 0.4, the same in every sample) plus
                 an independent one per sample (SD 0.3), so one sample matches PRIOR_SD = 0.5 and the K=3 median
                 removes only the independent part. Complaint matching returns "new" with acuity 3.
    Leakage (LEAK, default 0): the generator ignores the "primary complaint only" instruction to degree LEAK.
                 Its bucket is the log-odds mix (1 - LEAK) * logit P(d | complaint) + LEAK * logit
                 P(d | complaint + every finding reported so far). LEAK_KB=1 (default) computes the second term
                 with the AGENT'S OWN knowledge (its cards, confidence-shrunk; noisy drafts for diseases without a
                 card), so the same evidence with the same errors is counted twice: the double-counting case.
                 LEAK_KB=0 computes it with the world's true frequencies, i.e. the generator brings independent,
                 better knowledge; that is an oracle, not leakage, and is reported only as a bound.

Arms: all v0.3 flags off ("v0.2"), all on ("v0.3"), and all-on minus one fix at a time (marginal contribution).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import world_bench as WB  # noqa: E402
from dxagent.config import FREQ, FREQ_LEVELS, PRIOR_BUCKET, Settings  # noqa: E402
from dxagent.graph.nodes import Nodes  # noqa: E402
from dxagent.config import PRIOR_BUCKET as _PB  # noqa: E402
BUCKETS = list(_PB)
try:  # removed in v0.4: v04_sim.py imports this module's world, patient and roles without it
    from dxagent.knowledge.builder import KnowledgeService  # noqa: E402
except ImportError:
    KnowledgeService = None
from dxagent.knowledge.cards import (CardStore, DiseaseCard, FindingDef, FindingEntry, HarmProfile, Provenance,  # noqa: E402
                                     TestDef, TestEntry)
from dxagent.llm import ScriptedLLM  # noqa: E402
from dxagent.schemas import (ContrastDraft, CriticOutput,  # noqa: E402
                             ExtractedFinding, ExtractedTest, ExtractionOutput, FreeQuestionOutput, HypothesisOutput,
                             InterviewerOutput, OutcomeLevel, ProposedHypothesis, ReportOutput)

try:  # v0.3 drafting schemas, removed in v0.4
    from dxagent.schemas import CardDraft, DraftFinding, DraftHarm, DraftTest  # noqa: E402
except ImportError:
    CardDraft = DraftFinding = DraftHarm = DraftTest = None

try:  # v0.3.1 schemas (P1)
    from dxagent.schemas import ComplaintMatchOutput, ExtractedComplaint, PriorAssessment, PriorItem
except ImportError:
    ComplaintMatchOutput = ExtractedComplaint = PriorAssessment = PriorItem = None

try:  # v0.3 schemas; absent when this harness validates against the v0.2.1 package
    from dxagent.schemas import BackgroundDraft, BackgroundItem, CodeMatch, CodeMatchOutput
except ImportError:
    BackgroundDraft = BackgroundItem = CodeMatch = CodeMatchOutput = None

class _SerialExecutor:
    """Run the package's LLM-role thread pools serially. Concurrent card drafting makes the order in which drafts
    reach the shared store (and canonical-code merges) depend on thread timing, so identical runs differed in a few
    percent of cases. Serial execution makes every run, and every paired comparison, exactly reproducible."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, *its):
        return list(map(fn, *its))


for _mod in ("dxagent.graph.nodes", "dxagent.knowledge.builder", "dxagent.priors", "dxagent.knowledge.contrast"):
    try:
        setattr(__import__(_mod, fromlist=["x"]), "ThreadPoolExecutor", _SerialExecutor)
    except ImportError:  # dxagent.priors exists from v0.3.1
        pass

P = dict(OMIT_KNOWN=0.0, N_FAM=12, RHO_OW=0.3, VETO_TP=0.0, VETO_WRONG=0.0, VETO_FP=0.0, STUBBORN=0.0,
         REP_FIX=0.2, REP_ERR=0.05, Q_LEX=0.79, R=0.75, P_OTHER=0.6, CRITIC=0.3, RHO_DISC=0.5, KNOWN=0.75, SD_KNOWN=0.5, SD_DRAFT=0.7,
         FRAG=0.3, NONANSWER=0.10, PRIOR_SD=0.5, LEAK=0.0, LEAK_KB=1.0, ASSESS_SHARED_SD=0.4, ASSESS_IND_SD=0.3)
CLARITY = [("clear", 0.75, 0.04), ("hedged", 0.20, 0.20), ("inferred", 0.05, 0.35)]
FREE_CONF = {"clear": 0.90, "hedged": 0.82, "inferred": 0.78}
SYL = ["ka", "lo", "mi", "tu", "re", "po", "na", "vi", "zu", "be", "ga", "fo", "hi", "jo", "ku", "mo", "ni", "pu", "ra",
       "ti", "de", "sa", "we", "yo", "xi", "qe", "da", "fe", "go", "le"]


def u(*key) -> float:
    h = hashlib.sha256("|".join(map(str, key)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def word(*key) -> str:
    return "".join(SYL[int(u(*key, i) * len(SYL))] for i in range(3))


def level(p: float) -> str:
    return min(FREQ_LEVELS, key=lambda l: abs(math.log(FREQ[l]) - math.log(max(p, 1e-4))))


def bucket(p: float) -> str:
    return min(PRIOR_BUCKET, key=lambda b: abs(math.log(PRIOR_BUCKET[b]) - math.log(max(p, 1e-4))))


def noisy(p: float, sd: float, *key) -> float:
    z = statistics.NormalDist().inv_cdf(min(max(u(*key), 1e-9), 1 - 1e-9))
    lo = math.log(p / (1 - p)) + sd * z
    return min(max(1 / (1 + math.exp(-lo)), 1e-3), 1 - 1e-3)


# ============================================================================ world -> knowledge base


@dataclass
class SimWorld:
    seed: int
    fams: list
    code: dict  # wb finding code -> sim code
    wb: dict  # sim code -> wb code
    label: dict  # sim code -> label
    fam_of_code: dict
    dz: dict  # disease id -> WB Disease
    fam: dict  # disease id -> Family
    known: set
    cc: dict  # family idx -> complaint code
    t_orig: dict = field(default_factory=dict)  # lowercase test id -> world_bench test id
    alt_canon: dict = field(default_factory=dict)  # synonym code -> canonical code
    alt: dict = field(default_factory=dict)  # (disease, sim code) -> (alt code, alt label, canonical)


def build_world(seed: int, n_fam: int = 12) -> SimWorld:
    fams = WB.gen_world(seed, n_fam)
    code, wb, label, foc, dz, fam, cc = {}, {}, {}, {}, {}, {}, {}
    for F in fams:
        for c in F.bg:
            w1, w2, w3 = word(seed, c, 1), word(seed, c, 2), word(seed, c, 3)
            sc = f"{w1}_{w2}"
            code[c], wb[sc], label[sc], foc[sc] = sc, c, f"{w1} {w2} {w3}", F.idx
        ccode = f"{word(seed, 'cc', F.idx, 1)}_{word(seed, 'cc', F.idx, 2)}"
        cc[F.idx] = ccode
        label[ccode] = ccode.replace("_", " ")
        foc[ccode] = F.idx
        for d in F.diseases:
            dz[d.id.lower()], fam[d.id.lower()] = d, F
    known = {d for d in dz if u(seed, "known", d) < P["KNOWN"]}
    W = SimWorld(seed, fams, code, wb, label, foc, dz, fam, known, cc)
    W.t_orig = {t.lower(): t for F in fams for t in F.outcomes}
    for d in dz:
        for c in dz[d].p:
            sc = code[c]
            r = u(seed, "frag", d, c)
            if r < P["FRAG"]:
                kind = u(seed, "fragkind", d, c)
                w1, w2, w3 = label[sc].split()
                if kind < 0.1:
                    W.alt[(d, sc)] = (f"{w1}_{w2}_{w3}_x", f"{w1} {w2} {w3}", sc)
                elif kind < 0.9:
                    w4 = word(seed, "alt", d, c)
                    W.alt[(d, sc)] = (f"{w1}_{w2}_{w4}", f"{w1} {w2} {w4}", sc)
                else:
                    w5, w6 = word(seed, "alt5", d, c), word(seed, "alt6", d, c)
                    W.alt[(d, sc)] = (f"{w1}_{w5}_{w6}", f"{w1} {w5} {w6}", sc)
    W.alt_canon = {a[0]: a[2] for a in W.alt.values()}
    for a in W.alt.values():
        W.label[a[0]] = a[1]
    return W


def true_p(W: SimWorld, d: str, sc: str) -> float:
    c = W.wb[sc]
    return W.dz[d].p.get(c, W.fam[d].bg[c])


def true_dist(W: SimWorld, d: str, t: str) -> dict:
    t0 = W.t_orig.get(t, t)
    return W.dz[d].dist.get(t0, W.fam[d].bgt[t0])


def make_store(W: SimWorld) -> CardStore:
    st = CardStore(Path("/nonexistent_seed"), None)
    fdefs, tdefs = [], []
    for F in W.fams:
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
    for d in sorted(W.known):
        D, F = W.dz[d], W.fam[d]
        fe = [FindingEntry(code=W.code[c], label=W.label[W.code[c]], source=F.src[c],
                           p=noisy(v, P["SD_KNOWN"], W.seed, "known_p", d, c), conf=0.9) for c, v in D.p.items()
              if u(W.seed, "omit_known", d, c) >= P["OMIT_KNOWN"]]  # incomplete pre-built cards (scenario)
        te = {t: TestEntry(dist={o: max(v, 1e-3) * math.exp(P["SD_KNOWN"] * statistics.NormalDist().inv_cdf(
            min(max(u(W.seed, "known_t", d, t, o), 1e-9), 1 - 1e-9))) for o, v in dd.items()}, conf=0.9) for t, dd in D.dist.items()}
        te = {t.lower(): v for t, v in te.items()}
        st._register(DiseaseCard(id=d, name=d, harm=HarmProfile(untreated_course=D.course, urgency=D.urgency,
                                                                treatments=sorted(D.treatments), treatment_risk=D.treat_risk),
                                 findings=fe, tests=te, provenance=Provenance(status="seed"), prevalence=bucket(fam_prior(W, d)),
                                 prevalence_by_complaint={W.cc[F.idx]: bucket(fam_prior(W, d))}))
    st._lexicon = None
    return st


def fam_prior(W: SimWorld, d: str) -> float:
    F = W.fam[d]
    return W.dz[d].prior / sum(x.prior for x in F.diseases)


# ============================================================================ patients


@dataclass
class Case:
    idx: int
    d: str
    fam: int
    present: dict  # sim code -> bool (true)
    outcome: dict  # test -> outcome
    presenting: list


def sample_case(W: SimWorld, idx: int) -> Case:
    F = W.fams[int(u(W.seed, "fam", idx) * len(W.fams))]
    tot = sum(x.prior for x in F.diseases)
    r, acc, d = u(W.seed, "dz", idx) * tot, 0.0, F.diseases[-1].id.lower()
    for x in F.diseases:
        acc += x.prior
        if r <= acc:
            d = x.id.lower()
            break
    present = {W.code[c]: u(W.seed, "pres", idx, c) < W.dz[d].p.get(c, F.bg[c]) for c in F.bg}
    outcome = {}
    for t in F.outcomes:
        dd, r, acc = true_dist(W, d, t), u(W.seed, "tout", idx, t), 0.0
        outcome[t.lower()] = list(dd)[-1]
        for o, v in dd.items():
            acc += v
            if r <= acc:
                outcome[t.lower()] = o
                break
    chars = [W.code[c] for c in sorted(W.dz[d].p) if W.dz[d].p[c] >= 0.5 and present[W.code[c]] and F.src[c] == "history"]
    k = int(u(W.seed, "npres", idx) * 3)
    chars.sort(key=lambda c: u(W.seed, "presord", idx, c))
    return Case(idx, d, F.idx, present, outcome, chars[:k])


# ============================================================================ simulated LLM roles and patient


class Sim:
    """All simulated roles for one case; draws are keyed by (world, case, item) so arms see identical answers."""

    def __init__(self, W: SimWorld, case: Case, arm: str):
        self.W, self.c, self.arm = W, case, arm
        self.reveal: list[tuple[str, bool, str, bool]] = []  # (code, reported present, grade, lexically recoverable)
        self.reveal_tests: list[tuple[str, str]] = []
        self.revealed: set[str] = set()
        self.revealed_all: set[str] = set()
        self.calls: Counter = Counter()
        self.critic_used = False
        self.ow_round = 0
        self._lock = __import__("threading").Lock()
        self._per: Counter = Counter()
        self.reported: dict[str, bool] = {}  # every finding the patient has reported so far (for LEAK)
        self.store = None  # the agent's CardStore, for LEAK_KB

    def _next(self, *key) -> int:
        """Per-item sample index (1, 2, ...), independent of thread order and of other drafting in the arm."""
        with self._lock:
            self._per[key] += 1
            return self._per[key]

    # -- patient -------------------------------------------------------------

    def _say(self, sc: str, key: str, asked: str = "") -> tuple:
        W, c = self.W, self.c
        r, acc, grade, flip = u(W.seed, c.idx, "clar", sc, key), 0.0, "clear", 0.04
        for g, share, fl in CLARITY:
            acc += share
            if r <= acc:
                grade, flip = g, fl
                break
        truth = c.present[sc]
        reported = truth if u(W.seed, c.idx, "flip", sc, key) >= flip else not truth
        rec = u(W.seed, c.idx, "lex", sc) < P["Q_LEX"]
        return sc, reported, grade, rec, asked or sc

    def respond(self, actions: list[dict], turn: int) -> str:
        W, c, F = self.W, self.c, self.W.fams[self.c.fam]
        self.reveal, self.reveal_tests = [], []
        hist = [W.code[x] for x in F.bg if F.src[x] == "history"]
        exam = [W.code[x] for x in F.bg if F.src[x] == "exam"]
        for a in actions:
            if a.get("intake"):
                if a["id"] == "open_hpi":
                    cands = [(sc, 0.35) for sc in hist if c.present[sc]]
                elif a["id"] == "pmh_meds_social":
                    cands = [(sc, 0.25 if c.present[sc] else 0.10) for sc in hist]
                else:
                    cands = [(sc, 0.6) for sc in exam]
                for sc, pr in cands:
                    if sc not in self.revealed and u(W.seed, c.idx, "intake", a["id"], sc) < pr:
                        self.reveal.append(self._say(sc, a["id"]))
            elif a.get("test"):
                self.reveal_tests.append((a["test"], c.outcome[a["test"]]))
            elif a.get("free"):
                pool = [sc for sc in hist if c.present[sc] and sc not in self.revealed]
                if pool and u(W.seed, c.idx, "free", turn) < 0.5:
                    self.reveal.append(self._say(pool[int(u(W.seed, c.idx, "freepick", turn) * len(pool))], f"free{turn}"))
            else:
                for asked in a.get("reveals", []):
                    sc = W.alt_canon.get(asked, asked)  # a synonym code asks about the same finding
                    if sc not in W.wb or W.fam_of_code[sc] != c.fam:
                        continue
                    na = P["NONANSWER"] if F.src[W.wb[sc]] == "history" else 0.03
                    if u(W.seed, c.idx, "nonans", sc, turn) >= na:
                        self.reveal.append(self._say(sc, f"t{turn}", asked))
                vol = [sc for sc in hist if c.present[sc] and sc not in self.revealed]
                if vol and u(W.seed, c.idx, "vol", turn) < 0.15:
                    self.reveal.append(self._say(vol[int(u(W.seed, c.idx, "volpick", turn) * len(vol))], f"v{turn}"))
        self.revealed |= {r[0] for r in self.reveal}
        self.revealed_all |= {r[0] for r in self.reveal}
        self.reported.update({r[0]: r[1] for r in self.reveal})
        return self.text()

    def presentation(self) -> str:
        W, c = self.W, self.c
        cc = W.cc[c.fam]
        self.reveal = [(cc, True, "clear", True, cc)] + [(sc, True, "clear", True, sc) for sc in c.presenting]
        self.reveal_tests = []
        self.revealed |= {r[0] for r in self.reveal}
        self.reported.update({r[0]: r[1] for r in self.reveal})
        self.opening = True
        return self.text()

    def text(self) -> str:
        parts = []
        for sc, pres, grade, rec, _ in self.reveal:
            words = self.W.label[sc] if rec else f"{word('para', sc, 1)} {word('para', sc, 2)}"
            parts.append(("" if pres else "no ") + words)
        parts += [f"{t}: {o}" for t, o in self.reveal_tests]
        return "; ".join(parts) or "nothing to add"

    # -- LLM roles -------------------------------------------------------------

    def extract(self, system: str, user: str) -> ExtractionOutput:
        self.calls["extract"] += 1
        offered = set(re.findall(r"^- ([a-z0-9_]+): ", user, flags=re.M))
        out = []
        primary_done = False
        for sc, pres, grade, rec, asked in self.reveal:
            if asked in offered:  # coded as the finding the doctor asked about
                code, lab = asked, self.W.label.get(asked, self.W.label[sc])
            elif sc in offered:
                code, lab = sc, self.W.label[sc]
            elif rec:  # heard the registry's words: invents a code from them
                lab = self.W.label[sc]
                code = lab.replace(" ", "_")
            else:
                lab = f"{word('para', sc, 1)} {word('para', sc, 2)}"
                code = lab.replace(" ", "_")
            prim = sc == self.W.cc[self.c.fam] and not primary_done
            primary_done = primary_done or prim
            conf = min(0.99, FREE_CONF[grade] + 0.03 * (2 * u(self.W.seed, self.c.idx, "fc", sc, len(self.calls)) - 1))
            out.append(ExtractedFinding(code=code, label=lab, present=pres, confidence=conf, source="history",
                                        evidence="", primary_complaint=prim))
        tests = [ExtractedTest(test_id=t, outcome=o, confidence=0.95, evidence="") for t, o in self.reveal_tests]
        complaints = []
        if getattr(self, "opening", False) and ExtractedComplaint is not None:  # P1: opening statement only
            self.opening = False
            for (sc, pres, grade, rec, asked), f in zip(self.reveal, out):
                if pres:
                    complaints.append(ExtractedComplaint(label=f.label, quote=f.label, finding_code=f.code,
                                                         stated_reason=sc == self.W.cc[self.c.fam]))
        return ExtractionOutput(findings=out, test_results=tests, problem_representation=f"case {self.c.idx}",
                                complaints=complaints)

    def _prior(self, d: str, primary_prompt: bool) -> str:
        W, c = self.W, self.c
        if primary_prompt:
            p = fam_prior(W, d)
        else:  # v0.2 prompt: the prior conditions on the whole presentation (computed with TRUE knowledge)
            F = W.fams[c.fam]
            sc = {x.id.lower(): math.log(fam_prior(W, x.id.lower())) + sum(math.log(true_p(W, x.id.lower(), f) / F.bg[W.wb[f]])
                                                                          for f in c.presenting)
                  for x in F.diseases}
            m = max(sc.values())
            z = sum(math.exp(v - m) for v in sc.values())
            p = math.exp(sc[d] - m) / z
        if primary_prompt and P["LEAK"] > 0:  # the instruction is not obeyed: findings seen so far leak in
            F = W.fams[c.fam]
            rep = {f: v for f, v in self.reported.items() if f in W.wb and W.fam_of_code[f] == c.fam}
            sc = {}
            for x in F.diseases:
                xi = x.id.lower()
                s = math.log(fam_prior(W, xi))
                for f, v in rep.items():
                    if P["LEAK_KB"] and self.store is not None and self.store.get(xi) is not None:
                        q, bg = self.store.finding_p_eff(xi, f), self.store.background(f)  # the agent's own knowledge
                    elif P["LEAK_KB"]:
                        q, bg = noisy(true_p(W, xi, f), P["SD_DRAFT"], W.seed, "leak", xi, f), F.bg[W.wb[f]]
                    else:
                        q, bg = true_p(W, xi, f), F.bg[W.wb[f]]  # oracle bound
                    s += math.log(q / bg) if v else math.log((1 - q) / (1 - bg))
                sc[xi] = s
            m = max(sc.values())
            z = sum(math.exp(v - m) for v in sc.values())
            p_full = min(max(math.exp(sc[d] - m) / z, 1e-4), 0.999)
            lo = (1 - P["LEAK"]) * math.log(p / (1 - p)) + P["LEAK"] * math.log(p_full / (1 - p_full))
            p = 1 / (1 + math.exp(-lo))
        return bucket(noisy(min(p, 0.9), P["PRIOR_SD"], W.seed, c.idx, "prior", d, primary_prompt))

    def assess(self, system: str, user: str) -> "PriorAssessment":
        """P1 prior assessor: sees only S. True P(d | complaint), shared + per-sample log-odds noise."""
        self.calls["prior_assess"] += 1
        W, c = self.W, self.c
        k = self._next("assess")
        items = []
        for d in re.findall(r"^- ([a-z0-9_]+): ", user, flags=re.M):
            if d not in W.dz:
                items.append(PriorItem(disease_id=d, prior_bucket="uncommon"))
                continue
            p = noisy(min(fam_prior(W, d), 0.9), P["ASSESS_SHARED_SD"], W.seed, c.idx, "assess_shared", d)
            p = noisy(p, P["ASSESS_IND_SD"], W.seed, c.idx, "assess_ind", d, k)
            items.append(PriorItem(disease_id=d, prior_bucket=bucket(p)))
        return PriorAssessment(items=items)

    def _prop(self, d: str, primary: bool, src: str = "") -> ProposedHypothesis:
        D = self.W.dz[d]
        return ProposedHypothesis(name=d, disease_id=d, prior_bucket=self._prior(d, primary), cant_miss=D.cant_miss, rationale=src)

    def hypothesize(self, system: str, user: str) -> HypothesisOutput:
        W, c = self.W, self.c
        F = W.fams[c.fam]
        primary = "PRIMARY complaint is given separately" in system
        listed = set(re.findall(r"(f\d+_d\d+) \(", user))
        if "Unexplained finding:" in user:  # C3 discovery
            self.calls["discovery"] += 1
            lab = re.search(r"Unexplained finding: (.+)", user).group(1).strip()
            sc = next((k for k, v in W.label.items() if v == lab), None)
            out = []
            for x in F.diseases:
                xi = x.id.lower()
                if xi in listed or sc is None or sc not in W.wb or W.dz[xi].p.get(W.wb[sc], 0) < 0.5:
                    continue
                ok = u(W.seed, c.idx, "disc", xi, sc) < (P["RHO_DISC"] if xi == c.d else P["P_OTHER"])
                if ok:
                    out.append(self._prop(xi, primary, "discovery"))
            return HypothesisOutput(hypotheses=out[:3])
        if "organized by organ system" in system:  # C1 open-world re-prompt: rescues a missing truth w.p. RHO_OW
            self.calls["open_world"] += 1
            self.ow_round += 1
            out = [self._prop(x.id.lower(), primary, "open_world") for x in F.diseases
                   if x.id.lower() not in listed and (x.id.lower() != c.d or u(W.seed, c.idx, "ow", self.ow_round) < P["RHO_OW"])]
            return HypothesisOutput(hypotheses=out)
        self.calls["generator"] += 1
        out = []
        for x in F.diseases:
            xi = x.id.lower()
            p = P["R"] if xi == c.d else P["P_OTHER"]
            if u(W.seed, c.idx, "gen", xi) < p:
                out.append(self._prop(xi, primary))
        return HypothesisOutput(hypotheses=out)

    def critic(self, system: str, user: str) -> CriticOutput:
        """Rescue (as before) plus a veto model for D2: when the agent is about to commit, a wrong leader is vetoed
        naming the truth w.p. VETO_TP or a wrong alternative w.p. VETO_WRONG; a correct leader is vetoed (naming a
        wrong alternative) w.p. VETO_FP. STUBBORN=1 repeats the same decision on every call of the case."""
        self.calls["critic"] += 1
        order = re.findall(r"(f\d+_d\d+) p=", user)
        listed = set(order)
        out = CriticOutput()
        committing = "about to commit" in user
        if committing and self.c.d not in listed and not self.critic_used:
            self.critic_used = True
            if u(self.W.seed, self.c.idx, "critic") < P["CRITIC"]:
                out.missing_hypotheses = [self._prop(self.c.d, True, "critic")]
        if committing and order:
            self.veto_calls = getattr(self, "veto_calls", 0) + 1
            key = 0 if P["STUBBORN"] else self.veto_calls
            lead, alt = order[0], ""
            wrong = [x for x in order[1:] if x != self.c.d] or [x.id.lower() for x in self.W.fams[self.c.fam].diseases
                                                                if x.id.lower() not in (lead, self.c.d)]
            if lead != self.c.d:
                r = u(self.W.seed, self.c.idx, "veto", key)
                if r < P["VETO_TP"]:
                    alt = self.c.d
                elif r < P["VETO_TP"] + P["VETO_WRONG"] and wrong:
                    alt = wrong[0]
            elif u(self.W.seed, self.c.idx, "vetofp", key) < P["VETO_FP"] and wrong:
                alt = wrong[0]
            if alt:
                self.calls["veto"] += 1
                out.block_commit, out.veto_alternative, out.veto_reason = True, alt, "cant_miss_not_excluded"
        return out

    def report(self, system: str, user: str) -> ReportOutput:
        """E1: the report LLM names the truth instead of a wrong top w.p. REP_FIX, and replaces a correct top by the
        runner-up w.p. REP_ERR."""
        self.calls["report"] += 1
        self.rep_calls = getattr(self, "rep_calls", 0) + 1
        ids = re.findall(r"^\d+\. (\S+) p=", user, flags=re.M)
        top = ids[0] if ids else ""
        name, agrees = top, True
        if top != self.c.d:
            if u(self.W.seed, self.c.idx, "rep", self.rep_calls) < P["REP_FIX"]:
                name, agrees = self.c.d, False
        elif len(ids) > 1 and u(self.W.seed, self.c.idx, "reperr", self.rep_calls) < P["REP_ERR"]:
            name, agrees = ids[1], False
        return ReportOutput(final_diagnosis=name, rationale="x", agrees_with_top=agrees,
                            dissent_diagnosis="" if agrees else name, dissent_reason="" if agrees else "sim")

    def card(self, system: str, user: str) -> CardDraft:
        self.calls["card"] += 1
        W = self.W
        d = re.search(r"Disease: (\S+)", user).group(1)
        D, F = W.dz[d], W.fam[d]
        k = self._next("card", d)
        fs = []
        for c0, v in sorted(D.p.items()):
            if u(W.seed, "drop", d, c0, k) < 0.25:
                continue
            sc = W.code[c0]
            code, lab = sc, W.label[sc]
            if (d, sc) in W.alt:
                code, lab, _ = W.alt[(d, sc)]
            fs.append(DraftFinding(code=code, label=lab, source=F.src[c0], cluster=code,
                                   freq_in_disease=level(noisy(v, P["SD_DRAFT"], W.seed, "draft", d, c0, k)),
                                   background_freq=level(noisy(F.bg[c0], 0.8, W.seed, "draftbg", d, c0, k))))
        if u(W.seed, "halluc", d, k) < 0.3:
            c0 = sorted(F.bg)[int(u(W.seed, "hpick", d, k) * len(F.bg))]
            fs.append(DraftFinding(code=W.code[c0], label=W.label[W.code[c0]], source=F.src[c0], cluster=W.code[c0],
                                   freq_in_disease="common", background_freq=level(F.bg[c0])))
        ts = []
        for t, dd in D.dist.items():
            if u(W.seed, "tdrop", d, t, k) < 0.2:
                continue
            ts.append(DraftTest(test_id=t.lower(), name=t.lower(), type=F.ttype[t],
                                outcomes=[OutcomeLevel(outcome=o, level=level(noisy(min(max(v, 0.01), 0.99), P["SD_DRAFT"], W.seed, "dt", d, t, o, k)))
                                          for o, v in dd.items()]))
        return CardDraft(name=d, harm=DraftHarm(untreated_course=D.course, urgency=D.urgency, treatments=sorted(D.treatments),
                                                treatment_risk=D.treat_risk),
                         findings=fs, tests=ts, prevalence=bucket(noisy(fam_prior(W, d), 0.5, W.seed, "prev", d, k)))

    def background(self, system: str, user: str) -> BackgroundDraft:
        self.calls["background"] += 1
        W = self.W
        k = self._next("background", user)
        items = []
        for code in re.findall(r"^- ([a-z0-9_]+): ", user, flags=re.M):
            canon = next((a[2] for a in W.alt.values() if a[0] == code), code)
            if canon in W.wb:
                bg = W.fams[W.fam_of_code[canon]].bg[W.wb[canon]]
                items.append(BackgroundItem(code=code, background_freq=level(noisy(bg, 0.5, W.seed, "bgpass", code, k))))
        return BackgroundDraft(items=items)

    def code_match(self, system: str, user: str) -> CodeMatchOutput:
        self.calls["code_match"] += 1
        W = self.W
        out = []
        for m in re.finditer(r"^- new ([a-z0-9_]+) \(.*?\) \| candidates: (.+)$", user, flags=re.M):
            code, cands = m.group(1), re.findall(r"([a-z0-9_]+) \(", m.group(2))
            canon = next((a[2] for a in W.alt.values() if a[0] == code), None)
            ok = canon in cands and u(W.seed, "confirm", code) < 0.95
            out.append(CodeMatch(new_code=code, same_as=canon if ok else ""))
        return CodeMatchOutput(matches=out)

    def llm(self) -> ScriptedLLM:
        def fail(s, u_):
            raise RuntimeError("not simulated")
        h = {
            ExtractionOutput: self.extract, HypothesisOutput: self.hypothesize, CriticOutput: self.critic,
            InterviewerOutput: lambda s, u_: InterviewerOutput(message="..."),
            FreeQuestionOutput: lambda s, u_: FreeQuestionOutput(question="anything else?", target="open"),
            ReportOutput: self.report,
            ContrastDraft: fail,
        }
        if CardDraft is not None:
            h[CardDraft] = self.card
        if BackgroundDraft is not None:
            h.update({BackgroundDraft: self.background, CodeMatchOutput: self.code_match})
        if PriorAssessment is not None:
            h.update({PriorAssessment: self.assess,
                      ComplaintMatchOutput: lambda s, u_: ComplaintMatchOutput(same_as="", acuity=3)})
        return ScriptedLLM(h)


# ============================================================================ driver (same topology as graph/build.py)


# Shipped v0.3 fixes (switched off together in the "v0.2" arm, one at a time in "no-X" arms).
FLAGS = {"A1": "primary_complaint_only", "A2": "lexicon_vocab", "A3": "canonical_codes", "C3": "reverse_prevalence_prior",
         "D2": "structured_veto",
         "E1": ["deterministic_final", "dissent_reevaluate"], "F1": "strict_quorum"}
# Options tested and not shipped (off by default); "v0.3+B3" etc. switches one on.
VARIANTS = {
    "B3": {"background_pass": True},
    "C3a": {"reverse_evidence_rule": True},
    "C3b": {"cardless_discovery": True},
    "C1": {"open_world_enforce": True},  # no measurable gain on development worlds, +1.9 LLM calls per case
    "D1": {"conditional_batches": True},  # no measurable gain on development worlds
    "final": {},  # name used for the first held-out round (equal to the shipped defaults)
    "final2pct": {"reverse_prevalence_prior": False},
    "P1": {"prior_context": True},  # v0.3.1: S-only prior assessor, complaint rule, complaint-keyed prevalence
}


def arm_settings(arm: str) -> Settings:
    s = Settings()
    s.draft_missing_cards, s.live_retrieval, s.max_contrasts_per_turn = True, False, 0
    s.drafted_card_dir = None
    base, _, variant = arm.partition("+")
    on = base != "v0.2"
    off = set(base[3:].split(",")) if base.startswith("no-") else set()  # e.g. no-A2,A3
    s.background_pass = s.reverse_evidence_rule = s.cardless_discovery = False
    s.open_world_enforce = s.conditional_batches = False
    s.prior_context = False  # v0.3.1 P1 is a variant ("+P1"), so every earlier arm is reproduced exactly
    for key, fl in FLAGS.items():
        val = on and key not in off
        for f in (fl if isinstance(fl, list) else [fl]):
            setattr(s, f, val)
    s.live_samples = 3 if s.strict_quorum else 2  # F1: K=3 with a strict majority; v0.2: K=2, either sample
    for part in variant.split("+") if variant else []:
        if "=" in part:  # e.g. veto_ceiling=2
            k, v = part.split("=")
            cur = getattr(s, k)
            setattr(s, k, type(cur)(float(v)) if not isinstance(cur, bool) else v in ("1", "True", "true"))
        else:
            for k, v in VARIANTS.get(part, {}).items():
                setattr(s, k, v)
    return s


def run_case(W: SimWorld, store: CardStore, case: Case, arm: str, s: Settings) -> dict:
    sim = Sim(W, case, arm)
    sim.store = store
    llm = sim.llm()

    class Rt:
        pass

    rt = Rt()
    rt.settings, rt.doctor, rt.store = s, llm, store
    rt.knowledge = KnowledgeService(llm, store, s)
    n = Nodes(rt)
    state = {"presentation": "", "presentation_source": "referral", "turn": 0, "phase": "gathering", "findings": [],
             "test_results": [], "hypotheses": [], "last_actions": [], "context": {}, "transcript": []}
    state["last_reply"] = state["presentation"] = sim.presentation()
    node = "perceive"
    steps = 0
    while True:
        steps += 1
        if steps > 3000:
            raise RuntimeError("runaway loop")
        if node == "await_reply":
            reply = sim.respond(state.get("last_actions", []), state.get("turn", 0))
            state["last_reply"] = reply
            node = "perceive"
            continue
        out = getattr(n, node)(state) or {}
        state.update(out)
        if node == "perceive":
            node = "hypothesize"
        elif node == "hypothesize":
            node = "load_knowledge"
        elif node == "load_knowledge":
            node = "update_belief"
        elif node == "update_belief":
            node = "contrast"
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
    vetoes = state.get("vetoes", [])
    return {
        "case": case.idx, "truth": case.d, "correct": final == case.d, "top_prob": top["prob"],
        "top_correct": top["disease_id"] == case.d, "final_is_top": final == top["disease_id"],
        "vetoes": len(vetoes), "vetoes_honored": sum(v["honored"] for v in vetoes) if vetoes else int(state.get("critic_block", False)),
        "unresolved": bool(state.get("unresolved_objection")), "dissents": len(state.get("report_dissent", [])),
        "dissent_outcomes": [e.get("outcome", "") for e in state.get("report_dissent", [])],
        "p_truth": p_truth, "in_list": any(d["disease_id"] == case.d for d in diff), "known": case.d in W.known,
        "cant_miss": W.dz[case.d].cant_miss, "turns": state.get("turn", 0), "cost": state.get("cost_spent", 0.0),
        "other": state.get("other_prob", 0.0), "reason": state.get("commit_reason", ""), "calls": dict(sim.calls),
        "n_findings": len(state.get("findings", [])), "list_effect": state.get("list_effect_log", []),
        "n_hyps": len(state.get("hypotheses", [])),
        "prior_sens": (state.get("final_report") or {}).get("prior_sensitivity", {}),
        "prior_routes": dict(Counter(e["route"] for e in state.get("prior_log", []))),
        "prior_shift": [abs(BUCKETS.index(e["assessor"]) - BUCKETS.index(e["proposed"])) for e in state.get("prior_log", [])
                        if e.get("assessor") and e.get("proposed")],
        "revealed": len(sim.revealed_all), "usable": sum(1 for f in state.get("findings", [])
                                                          if not f.get("chief_complaint") and f["code"] in W.wb),
    }


# ============================================================================ statistics


def ece(rs: list[dict]) -> float:
    tot = 0.0
    for lo in [0.0, 0.2, 0.4, 0.6, 0.8]:
        sel = [r for r in rs if lo <= r["top_prob"] < lo + 0.2 or (lo == 0.8 and r["top_prob"] >= 1.0)]
        if sel:
            tot += len(sel) * abs(sum(r["top_prob"] for r in sel) / len(sel) - sum(r["correct"] for r in sel) / len(sel))
    return tot / len(rs)


def metrics(rs: list[dict]) -> dict:
    n = len(rs)
    calls = Counter()
    for r in rs:
        calls.update(r["calls"])
    return {
        "n": n, "top1": sum(r["correct"] for r in rs) / n, "in_list": sum(r["in_list"] for r in rs) / n,
        "brier": sum((r["top_prob"] - r["correct"]) ** 2 for r in rs) / n, "ece": ece(rs),
        "logloss": -sum(math.log(max(r["p_truth"], 1e-3)) for r in rs) / n,
        "turns": sum(r["turns"] for r in rs) / n, "cost": sum(r["cost"] for r in rs) / n,
        "cm_missed": (sum(1 for r in rs if r["cant_miss"] and not r["correct"]) / max(1, sum(r["cant_miss"] for r in rs))),
        "llm_calls": sum(calls.values()) / n, "calls": {k: round(v / n, 2) for k, v in calls.items()},
    }


def paired(a: list[dict], b: list[dict], key, reps: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    """Mean of key(b) - key(a) over paired cases, with a 95% bootstrap CI."""
    diffs = [key(y) - key(x) for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return sum(diffs) / n, boots[int(0.025 * reps)], boots[int(0.975 * reps)]


def mcnemar(a: list[dict], b: list[dict]) -> float:
    b01 = sum(1 for x, y in zip(a, b) if not x["correct"] and y["correct"])
    b10 = sum(1 for x, y in zip(a, b) if x["correct"] and not y["correct"])
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


# ============================================================================ main


def run_arm(arm: str, seeds: list[int], n_cases: int, log=print) -> list[dict]:
    out = []
    for seed in seeds:
        W = build_world(seed, int(P["N_FAM"]))
        store = make_store(W)
        s = arm_settings(arm)
        t0 = time.time()
        for i in range(n_cases):
            out.append({**run_case(W, store, sample_case(W, i), arm, s), "seed": seed})
        log(f"  {arm:6s} world {seed}: {n_cases} cases in {time.time() - t0:.0f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["v0.2", "v0.3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 12])
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--set", nargs="*", default=[], help="override assumptions, e.g. RHO_OW=0.6 R=0.9")
    ap.add_argument("--out", default="experiments/results/v03_sim")
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("=")
        P[k] = float(v)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "_".join(a.set) or "default"
    for arm in a.arms:
        rs = run_arm(arm, a.seeds, a.cases)
        (out / f"{arm}__{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in rs))
        m = metrics(rs)
        print(f"{arm:6s} top1 {m['top1']:.3f} in_list {m['in_list']:.3f} brier {m['brier']:.3f} ece {m['ece']:.3f} "
              f"logloss {m['logloss']:.3f} turns {m['turns']:.1f} cost {m['cost']:.0f} llm {m['llm_calls']:.1f}")


if __name__ == "__main__":
    main()
