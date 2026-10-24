"""Knowledge model: per-disease frequencies, run-time likelihood ratios.

A disease card stores what the disease looks like ON ITS OWN TERMS:

    findings   P(finding | disease)                 e.g. fever in pneumonia: common (0.65)
    tests      P(outcome | disease) per test        e.g. chest X-ray in pneumonia: consolidation 0.80, normal 0.10 ...
    harm       untreated course, urgency, treatments, treatment risk

Two registries hold what belongs to the finding or test, not to any disease:

    findings registry   background rate P(finding | patients presenting like this)
    tests registry      outcomes, background outcome distribution, cost, risk, numeric rules

A contrast card (A vs B) stores comparatively elicited frequencies for the findings and
tests that separate two confusable diseases. It fills gaps in either disease card.

Likelihood ratios are never stored. At run time the belief update compares P(obs | d)
with the background, and the planner compares diseases in the differential against each
other, so every ratio is between the diseases actually under consideration.

Every entry carries a confidence in [0, 1]. Low-confidence entries are shrunk toward the
background, so weakly supported knowledge moves beliefs less.

v0.4 (knowledge compiler). Where knowledge comes from and what the store loads:

    registry/          curated infrastructure: finding and test registries, complaint registry,
                       local ontology extension terms (shipped with the package)
    kb/                compiled knowledge (`dxagent kb compile`): cards.jsonl, findings.jsonl, tests.jsonl,
                       ontology indexes. Every entry carries provenance (source, tier, edition, locator, quote).
    cache/             run-time additions: LLM contrast cards, on-demand cards from the local documents,
                       provisional online-fallback cards, learned aliases
    fixtures/          plumbing fixtures for tests and `--offline` only; never loaded by default

Disease identity is an ontology id (MONDO, or a local DXA extension term) when one is known. A disease without
its own card inherits entries from its nearest carded ancestor (`lineage`), with a confidence decay per level,
and a card's own entries always override inherited ones.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import threading
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence

import yaml
from pydantic import BaseModel, Field, model_validator

from dxagent.config import DEFAULT_ACTION_COST, DEFAULT_ACTION_RISK, FREQ
from dxagent.knowledge.ontology import OntologyIndex, OntologySet, card_id_to_curie, curie_to_card_id

ActionType = Literal["history", "exam", "poc", "lab", "imaging", "advanced_imaging", "procedure"]
Level = Literal["very_common", "common", "occasional", "uncommon", "rare", "very_rare"]
EPS = 1e-3


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower().strip()).strip("_")


def norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def clamp(p: float) -> float:
    return min(max(float(p), EPS), 1 - EPS)


def normalize(dist: dict[str, float]) -> dict[str, float]:
    z = sum(max(v, 0.0) for v in dist.values())
    if z <= 0:
        return {k: 1.0 / len(dist) for k in dist} if dist else {}
    return {k: max(v, 0.0) / z for k, v in dist.items()}


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


class NumericRule(BaseModel):
    """Maps a numeric lab value to a test outcome deterministically."""

    analyte: str
    op: Literal["gt", "ge", "lt", "le"]
    value: float
    unit: str = ""
    outcome: str
    else_outcome: str = ""

    def evaluate(self, value: float) -> Optional[str]:
        hit = {"gt": value > self.value, "ge": value >= self.value, "lt": value < self.value, "le": value <= self.value}[self.op]
        return self.outcome if hit else (self.else_outcome or None)


class FindingDef(BaseModel):
    code: str
    label: str = ""
    source: ActionType = "history"
    background: float = 0.15  # P(finding | a patient presenting with this complaint)
    cluster: str = ""
    aliases: list[str] = Field(default_factory=list)  # v0.3 (A3): other codes that mean this finding
    # v0.3.1 (step 5): what kind of fact this is, so lost context can be measured by kind, and deterministic rules
    # that map a numeric vital sign to this finding (outcome "present" / else_outcome "absent").
    category: Optional[Literal["symptom", "sign", "vital", "risk_factor", "exposure", "medication", "family_history",
                               "lab", "imaging"]] = None
    rules: list[NumericRule] = Field(default_factory=list)
    # v0.4: ontology cross-references (e.g. "HP:0001945") and where the background rate came from:
    # registry (curated), claim (1 - specificity from a cited source), derived (mixture over compiled cards),
    # default (nothing known: 0.15).
    xrefs: list[str] = Field(default_factory=list)
    background_source: Literal["registry", "claim", "derived", "default", "contrast"] = "registry"
    synonyms: list[str] = Field(default_factory=list)  # v0.4: free-text names (ontology synonyms) for lexical retrieval
    # v0.5 (N.O.V.A, Korean encounters): lay Korean terms for lexical retrieval over Korean replies, and ONE Korean
    # utterance that reveals this finding (a question of at most 30 characters for history findings, a single
    # maneuver request for exam findings). Empty: the runtime phrases it with the LLM.
    ko: list[str] = Field(default_factory=list)
    ask_ko: str = ""

    @model_validator(mode="after")
    def _fill(self) -> "FindingDef":
        self.background = clamp(self.background)
        self.label = self.label or self.code.replace("_", " ")
        self.cluster = self.cluster or self.code
        return self


class TestDef(BaseModel):
    id: str
    name: str
    type: ActionType
    outcomes: list[str]
    background: dict[str, float] = Field(default_factory=dict)
    cost: Optional[float] = None
    risk: Optional[float] = None
    rules: list[NumericRule] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)  # v0.4: names used to verify that a cited quote is about this test
    ko: str = ""  # v0.5: Korean name of the test (plan section of the SOAP note)
    ask_ko: str = ""  # v0.5: one Korean sentence ordering this single test (final round TEST action)

    @model_validator(mode="after")
    def _fill(self) -> "TestDef":
        self.outcomes = [slug(o) for o in self.outcomes if slug(o)]
        bg = {slug(k): v for k, v in self.background.items()}
        self.background = normalize({o: bg.get(o, 0.0) or 1.0 / len(self.outcomes) for o in self.outcomes})
        if self.cost is None:
            self.cost = DEFAULT_ACTION_COST[self.type]
        if self.risk is None:
            self.risk = DEFAULT_ACTION_RISK[self.type]
        return self


# ---------------------------------------------------------------------------
# Disease and contrast cards
# ---------------------------------------------------------------------------


class EntryProvenance(BaseModel):
    """v0.4: where one card value came from. Written by the compiler; never by an LLM."""

    source_id: str  # manifest source id
    kind: str = ""  # guideline, systematic_review, rce, reference, curated_db, primary, abstract
    tier: int = 3
    precision: Literal["numeric", "class", "qualitative"] = "qualitative"
    edition: str = ""
    locator: str = ""  # page, section, or record (e.g. "p.12", "ORPHA:558 PMID:123")
    quote: str = ""  # verified verbatim span (truncated to 30 words)
    population: str = "unknown"
    n: Optional[int] = None  # sample size behind the value, when stated
    directness: float = 1.0
    conflicts: list[dict[str, Any]] = Field(default_factory=list)  # other sources that disagree (never averaged)


class FindingEntry(BaseModel):
    code: str
    label: str = ""
    source: ActionType = "history"
    p: Optional[float] = None  # P(finding | disease)
    freq: Optional[Level] = None  # alternative to p
    conf: float = 0.7
    cluster: str = ""
    evidence: str = ""  # human-readable citation
    prov: Optional[EntryProvenance] = None

    @model_validator(mode="after")
    def _fill(self) -> "FindingEntry":
        if self.p is None:
            self.p = FREQ[self.freq] if self.freq else 0.5
        self.p = clamp(self.p)
        self.conf = min(max(self.conf, 0.0), 1.0)
        return self


class TestEntry(BaseModel):
    dist: dict[str, float]  # P(outcome | disease)
    conf: float = 0.7
    evidence: str = ""
    prov: Optional[EntryProvenance] = None

    @model_validator(mode="after")
    def _fill(self) -> "TestEntry":
        self.dist = normalize({slug(k): v for k, v in self.dist.items()})
        return self


class HarmProfile(BaseModel):
    untreated_course: Literal["lethal", "serious", "self_limited"] = "serious"
    urgency: Literal["hours", "days", "weeks"] = "days"
    treatments: list[str] = Field(default_factory=list)
    treatment_risk: Literal["high", "moderate", "low"] = "low"
    evidence: str = ""  # v0.4: citation; empty means the defaults above (no source stated a harm profile)


class Provenance(BaseModel):
    # compiled: from the offline compiler. on_demand: compiled mid-case from the downloaded sources by the local
    # fallback (same pipeline and confidence). provisional: online fallback (capped confidence, not in the reverse index).
    # fixture: test and --offline plumbing only. llm_contrast: an LLM-elicited contrast card.
    # seed / grounded / consistent / llm_draft: pre-v0.4 statuses, accepted only so old files still load.
    status: Literal["compiled", "on_demand", "provisional", "fixture", "llm_contrast", "seed", "grounded", "consistent", "llm_draft"] = "compiled"
    source: str = ""
    version: str = "0.4"
    sources: list[dict[str, str]] = Field(default_factory=list)  # [{id, edition, kind}]
    compiled_at: str = ""


class DiseaseCard(BaseModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    harm: HarmProfile = Field(default_factory=HarmProfile)
    findings: list[FindingEntry] = Field(default_factory=list)
    tests: dict[str, TestEntry] = Field(default_factory=dict)
    mimics: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)
    # v0.3 (C3): pretest bucket among patients presenting with this kind of complaint (PRIOR_BUCKET key).
    prevalence: Optional[Literal["very_common", "common", "uncommon", "rare", "very_rare"]] = None
    # v0.3.1 (P1): the same bucket, keyed by canonical complaint (slug). A prevalence is a property of a disease
    # WITHIN a presenting population, so a card drafted for one complaint must not lend its bucket to another.
    prevalence_by_complaint: dict[str, Literal["very_common", "common", "uncommon", "rare", "very_rare"]] = Field(
        default_factory=dict)
    # v0.4: ontology identity and a complaint-independent floor bucket (e.g. very_rare for Orphanet diseases),
    # used by the reverse index when no complaint-keyed prevalence exists.
    ontology_id: str = ""  # "MONDO:0005249" or a local extension "DXA:0000001"
    parents: list[str] = Field(default_factory=list)  # ontology ids of direct parents
    base_prevalence: Optional[Literal["very_common", "common", "uncommon", "rare", "very_rare"]] = None

    def prevalence_for(self, complaint: str) -> Optional[str]:
        """Bucket for a canonical complaint key, or None. The single legacy `prevalence` is deliberately NOT a
        fallback: it was drafted in whatever context first triggered the card (or "general adult presentation"
        for `kb build`), which is not a presenting population. Without a keyed entry the assessor decides."""
        return self.prevalence_by_complaint.get(complaint) if complaint else None

    @property
    def cant_miss(self) -> bool:
        return self.harm.untreated_course == "lethal" and self.harm.urgency in ("hours", "days")

    @property
    def finding_map(self) -> dict[str, FindingEntry]:
        return {f.code: f for f in self.findings}


class ContrastFinding(BaseModel):
    code: str
    label: str = ""
    source: ActionType = "history"
    p_a: float
    p_b: float
    conf: float = 0.6
    cluster: str = ""


class ContrastTest(BaseModel):
    dist_a: dict[str, float]
    dist_b: dict[str, float]
    conf: float = 0.6


class ContrastCard(BaseModel):
    a: str
    b: str
    findings: list[ContrastFinding] = Field(default_factory=list)
    tests: dict[str, ContrastTest] = Field(default_factory=dict)
    management_differs: bool = True
    provenance: Provenance = Field(default_factory=Provenance)

    @property
    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.a, self.b)))  # type: ignore[return-value]


class ActionSpec(BaseModel):
    """One thing the doctor can do: ask or examine for findings, or order a test."""

    id: str
    name: str
    type: ActionType
    reveals: list[str] = Field(default_factory=list)  # finding codes (history/exam actions)
    test: Optional[str] = None  # test id (test actions)
    cost: float = 0.0
    risk: float = 0.0


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _dump(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)


def _read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


_PREC = {"numeric": "n", "class": "c", "qualitative": "q"}
_PREC_R = {v: k for k, v in _PREC.items()}


def _prov_row(pv: Optional[EntryProvenance], src_index: dict[str, int]) -> list:
    if pv is None:
        return []
    row = [src_index.get(pv.source_id, -1), pv.tier, _PREC[pv.precision], pv.locator, pv.n, pv.directness, pv.quote,
           pv.population, pv.conflicts]
    defaults = [None, None, None, "", None, 1.0, "", "unknown", []]
    while len(row) > 3 and row[-1] == defaults[len(row) - 1]:
        row.pop()
    return row


def card_to_row(card: DiseaseCard) -> dict[str, Any]:
    """Compact JSON row for kb/cards.jsonl.gz (v0.4). Per-entry labels, channels and clusters live in the finding
    registry; source kind and edition live once per card."""
    srcs = card.provenance.sources or [{"id": card.provenance.source or "unknown", "edition": "", "kind": ""}]
    idx = {x["id"]: i for i, x in enumerate(srcs)}
    row: dict[str, Any] = {"id": card.id, "name": card.name}
    if card.ontology_id:
        row["oid"] = card.ontology_id
    if card.aliases:
        row["al"] = card.aliases
    if card.parents:
        row["par"] = card.parents
    if card.base_prevalence:
        row["bp"] = card.base_prevalence
    if card.prevalence:
        row["prev"] = card.prevalence
    if card.prevalence_by_complaint:
        row["pv"] = card.prevalence_by_complaint
    if card.harm != HarmProfile():
        row["harm"] = card.harm.model_dump(exclude_defaults=True)
    if card.mimics:
        row["mim"] = card.mimics
    row["prov"] = {"st": card.provenance.status, "src": [[x.get("id", ""), x.get("edition", ""), x.get("kind", "")] for x in srcs],
                   "at": card.provenance.compiled_at, "v": card.provenance.version}
    row["f"] = [[f.code, round(f.p, 5), round(f.conf, 4), _prov_row(f.prov, idx)] for f in card.findings]
    if card.tests:
        row["t"] = {tid: [{o: round(v, 5) for o, v in e.dist.items()}, round(e.conf, 4), _prov_row(e.prov, idx)]
                    for tid, e in card.tests.items()}
    return row


def _expand_prov(pr: list, srcs: list[list[str]]) -> Optional[EntryProvenance]:
    if not pr:
        return None
    pr = list(pr) + [None] * (9 - len(pr))
    si, tier, prec, loc, n, d, quote, pop, conflicts = pr
    sid, edition, kind = (srcs[si] if si is not None and 0 <= si < len(srcs) else ["unknown", "", ""])
    return EntryProvenance(source_id=sid, kind=kind, tier=tier or 3, precision=_PREC_R.get(prec, "qualitative"), edition=edition,
                           locator=loc or "", quote=quote or "", population=pop or "unknown", n=n,
                           directness=1.0 if d is None else d, conflicts=conflicts or [])


def row_to_card(row: dict[str, Any], findings: dict[str, "FindingDef"]) -> DiseaseCard:
    pv = row.get("prov", {})
    srcs = pv.get("src", [])
    fes = []
    for code, p, conf, pr in row.get("f", []):
        fd = findings.get(code)
        prov = _expand_prov(pr, srcs)
        fes.append(FindingEntry(code=code, label=fd.label if fd else code.replace("_", " "), source=fd.source if fd else "history",
                                p=p, conf=conf, cluster=fd.cluster if fd else code, prov=prov,
                                evidence=" ".join(x for x in (prov.source_id, prov.edition, prov.locator) if x) if prov else ""))
    tests = {}
    for tid, (dist, conf, pr) in row.get("t", {}).items():
        prov = _expand_prov(pr, srcs)
        tests[tid] = TestEntry(dist=dist, conf=conf, prov=prov,
                               evidence=" ".join(x for x in (prov.source_id, prov.edition, prov.locator) if x) if prov else "")
    return DiseaseCard(
        id=row["id"], name=row["name"], aliases=row.get("al", []), harm=HarmProfile(**row.get("harm", {})), findings=fes,
        tests=tests, mimics=row.get("mim", []), ontology_id=row.get("oid", ""), parents=row.get("par", []),
        base_prevalence=row.get("bp"), prevalence=row.get("prev"), prevalence_by_complaint=row.get("pv", {}),
        provenance=Provenance(status=pv.get("st", "compiled"), source=", ".join(x[0] for x in srcs), version=pv.get("v", "0.4"),
                              sources=[{"id": a, "edition": b, "kind": c} for a, b, c in srcs], compiled_at=pv.get("at", "")))


class CardMeta:
    """What the reverse index needs from a card, readable without parsing its entries."""

    __slots__ = ("id", "name", "status", "prevalence", "prevalence_by_complaint", "base_prevalence", "cant_miss")

    def __init__(self, id, name, status, prevalence, prevalence_by_complaint, base_prevalence, cant_miss):
        self.id, self.name, self.status = id, name, status
        self.prevalence, self.prevalence_by_complaint = prevalence, prevalence_by_complaint or {}
        self.base_prevalence, self.cant_miss = base_prevalence, cant_miss

    def prevalence_for(self, complaint: str) -> Optional[str]:
        return self.prevalence_by_complaint.get(complaint) if complaint else None


# Statuses whose cards the reverse index may admit. Provisional (live fallback) and LLM contrast knowledge never
# pulls a disease onto the differential by itself.
INDEX_STATUSES = {"compiled", "on_demand", "fixture", "seed"}


class CardStore:
    """All knowledge, with run-time likelihood queries. Thread-safe for writes.

    registry_dir   curated finding/test registries (always loaded)
    cache_dir      run-time additions (contrast cards, provisional cards, learned aliases); also where they persist
    card_dirs      extra YAML card directories (fixtures for tests and --offline)
    kb_dir         compiled knowledge base (cards.jsonl, findings.jsonl, tests.jsonl, ontology/)
    ontology       disease (and phenotype) indexes; loaded from kb_dir/ontology when not given
    inherit_depth  how many ontology levels a card-less disease may climb to inherit a carded ancestor (0: off)
    """

    def __init__(self, registry_dir: Path, cache_dir: Optional[Path] = None, perturb: float = 1.0, perturb_seed: int = 0,
                 *, card_dirs: Sequence[Path] = (), kb_dir: Optional[Path] = None, ontology: Optional[OntologySet] = None,
                 inherit_depth: int = 3, inherit_decay: float = 0.9, local_terms: Optional[Path] = None):
        self.registry_dir = Path(registry_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.kb_dir = Path(kb_dir) if kb_dir else None
        self.perturb = perturb
        self.perturb_seed = perturb_seed
        self.inherit_depth = inherit_depth
        self.inherit_decay = inherit_decay
        self._lock = threading.RLock()
        self._cards: dict[str, DiseaseCard] = {}
        self._raw: dict[str, dict[str, Any]] = {}  # compiled cards not parsed yet (parsed on first use)
        self._alias: dict[str, str] = {}
        self._findings: dict[str, FindingDef] = {}
        self._tests: dict[str, TestDef] = {}
        self._contrasts: dict[tuple[str, str], ContrastCard] = {}
        self._cache_findings: dict[str, FindingDef] = {}  # run-time registry additions, persisted to cache_dir
        self._cache_tests: dict[str, TestDef] = {}
        self._canon: dict[str, str] = {}  # alias code -> canonical code (A3)
        self._lexicon = None
        self._by_code: Optional[dict[str, set[str]]] = None  # inverted index: finding code / test id -> card ids
        self._lineage_cache: dict[str, list[tuple[str, int]]] = {}
        self._fmaps: dict[str, dict[str, tuple[float, float]]] = {}
        self.kb_info: dict[str, Any] = {}

        self._load_registry(self.registry_dir)
        if self.kb_dir and self.kb_dir.exists():
            self._load_kb(self.kb_dir)
        for d in card_dirs:
            if d and Path(d).exists():
                self._load_cards_dir(Path(d))
        if self.cache_dir and self.cache_dir.exists():
            self._load_registry(self.cache_dir, cache=True)
            self._load_cards_dir(self.cache_dir / "on_demand")
            self._load_cards_dir(self.cache_dir / "provisional")
            self._load_contrasts(self.cache_dir / "contrasts")
        if ontology is None and self.kb_dir and (self.kb_dir / "ontology").exists():
            ontology = OntologySet.load_dir(self.kb_dir / "ontology")
        self.ontology: OntologySet = ontology or OntologySet()
        lt = local_terms if local_terms is not None else self.registry_dir / "local_terms.yaml"
        if self.ontology.disease is not None and lt and Path(lt).exists():
            extend_with_local_terms(self.ontology.disease, Path(lt))
        self._rebuild_canon()

    # -- loading -------------------------------------------------------------

    def _load_registry(self, d: Path, cache: bool = False) -> None:
        fpath, tpath = d / "_findings.yaml", d / "_tests.yaml"
        if fpath.exists():
            for row in yaml.safe_load(fpath.read_text(encoding="utf-8")) or []:
                self._add_finding_def(FindingDef.model_validate(row), cache)
        if tpath.exists():
            for row in yaml.safe_load(tpath.read_text(encoding="utf-8")) or []:
                td = TestDef.model_validate(row)
                self._tests.setdefault(td.id, td)
                if cache:
                    self._cache_tests[td.id] = self._tests[td.id]

    def _add_finding_def(self, fd: FindingDef, cache: bool = False) -> None:
        have = self._findings.get(fd.code)
        if have is not None:
            if fd.aliases:  # a later alias list extends the definition (A3)
                have.aliases = list(dict.fromkeys(have.aliases + fd.aliases))
            if fd.xrefs:
                have.xrefs = list(dict.fromkeys(have.xrefs + fd.xrefs))
        else:
            self._findings[fd.code] = fd
        if cache:
            self._cache_findings[fd.code] = self._findings[fd.code]

    def _load_kb(self, d: Path) -> None:
        if (d / "findings.jsonl").exists():
            for row in _read_jsonl(d / "findings.jsonl"):
                self._add_finding_def(FindingDef.model_validate(row))
        if (d / "tests.jsonl").exists():
            for row in _read_jsonl(d / "tests.jsonl"):
                td = TestDef.model_validate(row)
                self._tests.setdefault(td.id, td)
        for name in ("cards.jsonl.gz", "cards.jsonl"):
            if (d / name).exists():
                for row in _read_jsonl(d / name):
                    cid = row["id"]
                    self._raw[cid] = row
                    for key in [cid, row.get("name", ""), *row.get("al", []), row.get("oid", "")]:
                        if key:
                            self._alias.setdefault(norm_key(key), cid)
                break
        if (d / "build.json").exists():
            try:
                self.kb_info = json.loads((d / "build.json").read_text(encoding="utf-8"))
            except Exception:  # an unreadable build record must not block loading the cards
                self.kb_info = {}

    def _load_cards_dir(self, d: Path) -> None:
        if not d.exists():
            return
        for path in sorted(d.glob("*.yaml")):
            if path.name.startswith("_"):
                continue
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if data:
                self._register(DiseaseCard.model_validate(data))
        self._load_contrasts(d / "contrasts")

    def _load_contrasts(self, cdir: Path) -> None:
        if cdir.exists():
            for path in sorted(cdir.glob("*.yaml")):
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if data:
                    cc = ContrastCard.model_validate(data)
                    self._contrasts[cc.key] = cc

    def _register(self, card: DiseaseCard) -> None:
        with self._lock:
            self._remap_card(card)
            self._cards[card.id] = card
            for key in [card.id, card.name, *card.aliases, card.ontology_id]:
                if key:
                    self._alias.setdefault(norm_key(key), card.id)
            for f in card.findings:
                if f.code not in self._findings:
                    self._findings[f.code] = FindingDef(code=f.code, label=f.label, source=f.source, cluster=f.cluster,
                                                        background_source="default")
                    self._lexicon = None
            self._by_code = None
            self._lineage_cache.clear()
            self._fmaps.pop(card.id, None)

    def _has(self, cid: str) -> bool:
        return cid in self._cards or cid in self._raw

    def _fmap(self, cid: str) -> dict[str, tuple[float, float]]:
        """code -> (p, conf) for a card, WITHOUT parsing a compiled row (the reverse index scans thousands)."""
        hit = self._fmaps.get(cid)
        if hit is not None:
            return hit
        card = self._cards.get(cid)
        if card is not None:
            m = {f.code: (f.p, f.conf) for f in card.findings}
        else:
            row = self._raw.get(cid) or {}
            m = {}
            for code, p, conf, _ in row.get("f", []):
                c = self.canonical(code)
                if c not in m or conf > m[c][1]:
                    m[c] = (clamp(p), conf)
        self._fmaps[cid] = m
        return m

    def _tmap(self, cid: str) -> dict[str, tuple[dict[str, float], float]]:
        card = self._cards.get(cid)
        if card is not None:
            return {t: (e.dist, e.conf) for t, e in card.tests.items()}
        row = self._raw.get(cid) or {}
        return {t: (normalize({slug(k): v for k, v in d.items()}), conf) for t, (d, conf, _) in row.get("t", {}).items()}

    def card_meta(self, cid: str) -> Optional["CardMeta"]:
        """Name, status, prevalence and harm of a card without parsing its entries."""
        card = self._cards.get(cid)
        if card is not None:
            return CardMeta(card.id, card.name, card.provenance.status, card.prevalence, card.prevalence_by_complaint,
                            card.base_prevalence, card.cant_miss)
        row = self._raw.get(cid)
        if row is None:
            return None
        harm = HarmProfile(**row.get("harm", {}))
        cm = harm.untreated_course == "lethal" and harm.urgency in ("hours", "days")
        return CardMeta(cid, row.get("name", cid), row.get("prov", {}).get("st", "compiled"), row.get("prev"), row.get("pv", {}),
                        row.get("bp"), cm)

    def _card(self, cid: str) -> Optional[DiseaseCard]:
        card = self._cards.get(cid)
        if card is not None:
            return card
        with self._lock:
            row = self._raw.pop(cid, None)
            if row is None:
                return self._cards.get(cid)
            card = row_to_card(row, self._findings)
            self._remap_card(card)
            self._cards[cid] = card
            return card

    # -- canonical finding codes (v0.3, A3) --------------------------------------

    def _rebuild_canon(self) -> None:
        with self._lock:
            for fd in self._findings.values():
                for a in fd.aliases:
                    if a != fd.code:
                        self._canon[a] = fd.code
            for card in self._cards.values():
                self._remap_card(card)
            for cc in self._contrasts.values():
                seen: dict[str, ContrastFinding] = {}
                for f in cc.findings:
                    f.code = self.canonical(f.code)
                    if f.code not in seen or f.conf > seen[f.code].conf:
                        seen[f.code] = f
                cc.findings = list(seen.values())
            self._lexicon = None
            self._by_code = None
            self._fmaps.clear()

    def _remap_card(self, card: DiseaseCard) -> None:
        if not self._canon:
            return
        kept: dict[str, FindingEntry] = {}
        for f in card.findings:
            f.code = self.canonical(f.code)
            if f.code not in kept or f.conf > kept[f.code].conf:
                kept[f.code] = f
        card.findings = list(kept.values())

    def canonical(self, code: str) -> str:
        seen = set()
        while code in self._canon and code not in seen:
            seen.add(code)
            code = self._canon[code]
        return code

    def add_alias(self, alias: str, canonical: str, persist: bool = True) -> None:
        """Declare `alias` a synonym of `canonical`; cards and contrasts are rewritten to the canonical code."""
        canonical = self.canonical(canonical)
        if not alias or alias == canonical:
            return
        with self._lock:
            self._canon[alias] = canonical
            fd = self._findings.get(canonical)
            if fd is not None and alias not in fd.aliases:
                fd.aliases.append(alias)
                self._cache_findings.setdefault(canonical, fd)
            self._rebuild_canon()
            if persist:
                self._persist_registry()

    def lexicon(self):
        from dxagent.knowledge.lexicon import Lexicon

        with self._lock:
            if self._lexicon is None:
                self._lexicon = Lexicon(
                    (c, fd.label, list(fd.aliases) + list(fd.synonyms) + list(fd.ko))
                    for c, fd in self._findings.items() if c not in self._canon
                )
            return self._lexicon

    def _persist_registry(self) -> None:
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "_findings.yaml").write_text(
            _dump([f.model_dump(mode="json", exclude_defaults=True) | {"code": f.code} for f in self._cache_findings.values()]),
            encoding="utf-8")
        if self._cache_tests:
            (self.cache_dir / "_tests.yaml").write_text(
                _dump([t.model_dump(mode="json", exclude_none=True) for t in self._cache_tests.values()]), encoding="utf-8")

    def add_registry(self, findings: Iterable[FindingDef] = (), tests: Iterable[TestDef] = (), persist: bool = True) -> None:
        with self._lock:
            changed = False
            for fd in findings:
                if fd.code not in self._findings and fd.code not in self._canon:
                    self._findings[fd.code] = fd
                    self._cache_findings[fd.code] = fd
                    changed = True
                    self._lexicon = None
            for td in tests:
                if td.id not in self._tests:
                    self._tests[td.id] = td
                    self._cache_tests[td.id] = td
                    changed = True
            if changed and persist:
                self._persist_registry()

    def add(self, card: DiseaseCard, persist: bool = True) -> DiseaseCard:
        """Add a run-time card (on_demand from the local fallback, provisional from the online fallback). Compiled
        cards come only from `kb/`. Persisted under cache/<status>/."""
        with self._lock:
            existing = self._card(card.id) or self.get(card.name)
            if existing:
                return existing
            self._register(card)
            if persist and self.cache_dir:
                d = self.cache_dir / ("on_demand" if card.provenance.status == "on_demand" else "provisional")
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{card.id}.yaml").write_text(_dump(card.model_dump(mode="json", exclude_none=True)), encoding="utf-8")
            return card

    def add_contrast(self, cc: ContrastCard, persist: bool = True) -> None:
        with self._lock:
            self._contrasts[cc.key] = cc
            for f in cc.findings:
                if f.code not in self._findings:
                    self._findings[f.code] = FindingDef(code=f.code, label=f.label, source=f.source, cluster=f.cluster,
                                                        background_source="contrast")
            self._lineage_cache.clear()
            if persist and self.cache_dir:
                cdir = self.cache_dir / "contrasts"
                cdir.mkdir(parents=True, exist_ok=True)
                (cdir / f"{cc.key[0]}__{cc.key[1]}.yaml").write_text(_dump(cc.model_dump(mode="json")), encoding="utf-8")

    # -- lookup ----------------------------------------------------------------

    def get(self, key: str) -> Optional[DiseaseCard]:
        if not key:
            return None
        if self._has(key):
            return self._card(key)
        cid = self._alias.get(norm_key(key))
        return self._card(cid) if cid else None

    def resolve(self, name: str, proposed_id: str = "") -> Optional[str]:
        """Card id for a name or id, or None (cards only; see `resolve_disease` for ontology-level identity)."""
        card = self.get(proposed_id) or self.get(name)
        return card.id if card else None

    def resolve_disease(self, name: str, proposed_id: str = "") -> tuple[Optional[str], str, str]:
        """(disease id, ontology id, how). A card match wins; otherwise the disease ontology resolves the name
        deterministically and the id is the ontology-derived card id even if no card exists yet (it may inherit
        from a carded ancestor). how in {card, exact, synonym, token, id, ambiguous, none}."""
        card = self.get(proposed_id) or self.get(name)
        if card:
            return card.id, card.ontology_id, "card"
        idx = self.ontology.disease
        if idx is None:
            return None, "", "none"
        queries = []
        if proposed_id:
            curie = card_id_to_curie(proposed_id)
            queries.append(curie if curie else proposed_id.replace("_", " "))
        if name:
            queries.append(name)
        for q in queries:
            tid, how, _ = idx.resolve(q)
            if tid and how != "ambiguous":
                cid = curie_to_card_id(tid)
                return cid, tid, how
        return None, "", "none"

    def __len__(self) -> int:
        return len(self._cards) + len(self._raw)

    def all_cards(self) -> list[DiseaseCard]:
        """Every card, parsed. Slow on a large compiled base; prefer `card_ids`, `get`, or `cards_speaking_to`."""
        for cid in list(self._raw):
            self._card(cid)
        with self._lock:
            return list(self._cards.values())

    def card_ids(self) -> list[str]:
        return list(self._cards) + list(self._raw)

    def known_diseases(self) -> list[tuple[str, str]]:
        return [(c.id, c.name) for c in self._cards.values()] + [(cid, r.get("name", cid)) for cid, r in self._raw.items()]

    def has_contrast(self, a: str, b: str) -> bool:
        return tuple(sorted((a, b))) in self._contrasts

    def contrasts_for(self, disease_id: str) -> list[tuple[ContrastCard, str]]:
        """Contrast cards involving this disease, with the side ("a" or "b") it occupies."""
        with self._lock:
            out = []
            for cc in self._contrasts.values():
                if cc.a == disease_id:
                    out.append((cc, "a"))
                elif cc.b == disease_id:
                    out.append((cc, "b"))
            return out

    def finding_def(self, code: str) -> Optional[FindingDef]:
        return self._findings.get(code)

    def test_def(self, test_id: str) -> Optional[TestDef]:
        return self._tests.get(test_id)

    def tests(self) -> list[TestDef]:
        return list(self._tests.values())

    def background(self, code: str) -> float:
        fd = self._findings.get(code)
        return fd.background if fd else 0.15

    def cluster(self, code: str) -> str:
        fd = self._findings.get(code)
        return fd.cluster if fd else code

    # -- ontology: identity, hierarchy, inheritance (v0.4) ------------------------

    def ontology_id(self, disease_id: str) -> str:
        row = self._raw.get(disease_id)
        if row is not None:
            return row.get("oid", "") or card_id_to_curie(disease_id)
        card = self._cards.get(disease_id)
        if card and card.ontology_id:
            return card.ontology_id
        return card_id_to_curie(disease_id)

    def ancestors(self, disease_id: str) -> dict[str, int]:
        """{ancestor disease id: is_a distance}, in card-id form. Uses the ontology when the disease has an
        ontology id; otherwise the card's declared parents (one level)."""
        idx = self.ontology.disease
        curie = self.ontology_id(disease_id)
        if idx is not None and curie and curie in idx:
            return {curie_to_card_id(a): d for a, d in idx.ancestors(curie).items()}
        card = self._card(disease_id) if self._has(disease_id) else None
        out: dict[str, int] = {}
        for p in (card.parents if card else []):
            pid = curie_to_card_id(p) if ":" in p else p
            out[pid] = 1
        return out

    def is_ancestor(self, anc: str, disease_id: str) -> bool:
        return anc != disease_id and anc in self.ancestors(disease_id)

    def lineage(self, disease_id: str) -> list[tuple[str, int]]:
        """[(card id, distance)] to consult for this disease: its own card (distance 0, if any), then carded
        ancestors within `inherit_depth`, nearest first."""
        hit = self._lineage_cache.get(disease_id)
        if hit is not None:
            return hit
        out: list[tuple[str, int]] = []
        if self._has(disease_id):
            out.append((disease_id, 0))
        if self.inherit_depth > 0:
            anc = sorted(((a, d) for a, d in self.ancestors(disease_id).items() if d <= self.inherit_depth and self._has(a)),
                         key=lambda x: (x[1], x[0]))
            out.extend(anc)
        self._lineage_cache[disease_id] = out
        return out

    def has_knowledge(self, disease_id: str) -> bool:
        """True if any card (own or inherited) or contrast card speaks for this disease."""
        return bool(self.lineage(disease_id)) or bool(self.contrasts_for(disease_id))

    def inherited_from(self, disease_id: str) -> Optional[str]:
        """The ancestor card a card-less disease borrows from, or None."""
        lin = self.lineage(disease_id)
        return lin[0][0] if lin and lin[0][1] > 0 else None

    def index_eligible(self, card) -> bool:
        """Accepts a DiseaseCard or a CardMeta."""
        status = card.status if isinstance(card, CardMeta) else card.provenance.status
        return status in INDEX_STATUSES

    def cards_speaking_to(self, codes: Iterable[str], test_ids: Iterable[str]) -> set[str]:
        """Card ids with an entry for any of these finding codes or tests (inverted index, rebuilt lazily)."""
        with self._lock:
            if self._by_code is None:
                idx: dict[str, set[str]] = {}
                for c in self._cards.values():
                    for f in c.findings:
                        idx.setdefault(f.code, set()).add(c.id)
                    for t in c.tests:
                        idx.setdefault(f"test:{t}", set()).add(c.id)
                for cid, row in self._raw.items():
                    for f in row.get("f", []):
                        idx.setdefault(self.canonical(f[0]), set()).add(cid)
                    for t in row.get("t", {}):
                        idx.setdefault(f"test:{t}", set()).add(cid)
                self._by_code = idx
            out: set[str] = set()
            for c in codes:
                out |= self._by_code.get(c, set())
            for t in test_ids:
                out |= self._by_code.get(f"test:{t}", set())
            return out

    # -- run-time likelihoods ---------------------------------------------------

    def _noise(self, *key: str) -> float:
        """Deterministic multiplicative log-odds perturbation for robustness checks."""
        if self.perturb <= 1.0:
            return 0.0
        h = hashlib.sha256(("|".join(key) + f"|{self.perturb_seed}").encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2**64 * 2 - 1  # uniform in [-1, 1)
        return u * math.log(self.perturb)

    def _perturb_p(self, p: float, *key: str) -> float:
        n = self._noise(*key)
        if n == 0.0:
            return p
        lo = math.log(p / (1 - p)) + n
        return clamp(1 / (1 + math.exp(-lo)))

    def finding_p(self, disease_id: str, code: str) -> Optional[tuple[float, float]]:
        """(P(finding | disease), confidence): the disease's own card, else the nearest carded ancestor (confidence
        decayed per level), else a contrast card. An own entry always overrides an inherited one."""
        for cid, dist in self.lineage(disease_id):
            e = self._fmap(cid).get(code)
            if e is not None:
                return self._perturb_p(e[0], disease_id, code), e[1] * (self.inherit_decay ** dist)
        best: Optional[tuple[float, float]] = None
        for cc, side in self.contrasts_for(disease_id):
            for f in cc.findings:
                if f.code == code:
                    p = clamp(f.p_a if side == "a" else f.p_b)
                    if best is None or f.conf > best[1]:
                        best = (self._perturb_p(p, disease_id, code), f.conf)
        return best

    def finding_p_eff(self, disease_id: Optional[str], code: str) -> float:
        """Confidence-shrunk P(finding | disease); background if the disease has no entry."""
        bg = self.background(code)
        if disease_id is None:
            return bg
        hit = self.finding_p(disease_id, code)
        if hit is None:
            return bg
        p, conf = hit
        return clamp(conf * p + (1 - conf) * bg)

    def test_dist(self, disease_id: str, test_id: str) -> Optional[tuple[dict[str, float], float]]:
        td = self._tests.get(test_id)
        if td is None:
            return None
        raw: Optional[tuple[dict[str, float], float]] = None
        for cid, dist in self.lineage(disease_id):
            e = self._tmap(cid).get(test_id)
            if e is not None:
                raw = (e[0], e[1] * (self.inherit_decay ** dist))
                break
        if raw is None:
            for cc, side in self.contrasts_for(disease_id):
                ct = cc.tests.get(test_id)
                if ct is not None and (raw is None or ct.conf > raw[1]):
                    raw = (normalize(ct.dist_a if side == "a" else ct.dist_b), ct.conf)
        if raw is None:
            return None
        dist = {o: raw[0].get(o, 0.0) for o in td.outcomes}
        if self.perturb > 1.0:
            dist = {o: math.exp(math.log(max(v, EPS)) + self._noise(disease_id, test_id, o)) for o, v in dist.items()}
        return normalize(dist), raw[1]

    def test_dist_eff(self, disease_id: Optional[str], test_id: str) -> dict[str, float]:
        td = self._tests[test_id]
        if disease_id is None:
            return td.background
        hit = self.test_dist(disease_id, test_id)
        if hit is None:
            return td.background
        dist, conf = hit
        return normalize({o: conf * dist[o] + (1 - conf) * td.background[o] for o in td.outcomes})

    def entry_source(self, disease_id: str, code: str = "", test_id: str = "") -> dict[str, Any]:
        """Where the value used for (disease, finding code | test) comes from (v0.4.1, diagnostics only):
        {via: own | inherited | contrast | none, card, status, tier, precision, source_id}."""
        for cid, dist in self.lineage(disease_id):
            has = (code and code in self._fmap(cid)) or (test_id and test_id in self._tmap(cid))
            if not has:
                continue
            out: dict[str, Any] = {"via": "own" if dist == 0 else "inherited", "card": cid}
            card = self._card(cid)
            out["status"] = card.provenance.status
            entry = card.finding_map.get(code) if code else card.tests.get(test_id)
            if entry is not None and entry.prov is not None:
                out.update({"tier": entry.prov.tier, "precision": entry.prov.precision, "source_id": entry.prov.source_id})
            return out
        for cc, _ in self.contrasts_for(disease_id):
            if (code and any(f.code == code for f in cc.findings)) or (test_id and test_id in cc.tests):
                return {"via": "contrast", "card": f"{cc.a}__{cc.b}", "status": cc.provenance.status}
        return {"via": "none"}

    # -- derived views ------------------------------------------------------------

    def codes_for(self, disease_id: str) -> set[str]:
        codes: set[str] = set()
        for cid, _ in self.lineage(disease_id):
            codes |= set(self._fmap(cid))
        for cc, _ in self.contrasts_for(disease_id):
            codes |= {f.code for f in cc.findings}
        return codes

    def tests_for(self, disease_id: str) -> set[str]:
        ids: set[str] = set()
        for cid, _ in self.lineage(disease_id):
            ids |= set(self._tmap(cid))
        for cc, _ in self.contrasts_for(disease_id):
            ids |= set(cc.tests)
        return {t for t in ids if t in self._tests}

    def vocabulary(self, ids: Optional[list[str]] = None) -> dict[str, dict[str, str]]:
        codes = set(self._findings) if not ids else set().union(*(self.codes_for(i) for i in ids)) if ids else set()
        codes = {c for c in codes if c not in self._canon}  # aliases are never offered (A3)
        return {c: {"label": self._findings[c].label, "source": self._findings[c].source} for c in sorted(codes) if c in self._findings}

    def _ranked_codes(self, disease_id: str, limit: int) -> list[str]:
        """The disease's most informative finding codes: largest |log P(f|d) - log bg| first (v0.4: compiled cards
        can carry hundreds of phenotype entries; the planner only needs the discriminating ones)."""
        codes = self.codes_for(disease_id)
        if limit <= 0 or len(codes) <= limit:
            return sorted(codes)

        def strength(c: str) -> float:
            p, bg = self.finding_p_eff(disease_id, c), self.background(c)
            return max(abs(math.log(p / bg)), abs(math.log((1 - p) / (1 - bg))))

        return sorted(codes, key=lambda c: (-strength(c), c))[:limit]

    def actions_for(self, ids: list[str], max_codes_per_disease: int = 0) -> list[ActionSpec]:
        actions: dict[str, ActionSpec] = {}
        for cid in ids:
            for code in self._ranked_codes(cid, max_codes_per_disease):
                fd = self._findings.get(code)
                if fd is None:
                    continue
                prefix = {"history": "ask", "exam": "exam"}.get(fd.source, "obs")
                aid = f"{prefix}_{code}"
                if aid not in actions:
                    actions[aid] = ActionSpec(
                        id=aid, name=fd.label, type=fd.source, reveals=[code],
                        cost=DEFAULT_ACTION_COST[fd.source], risk=DEFAULT_ACTION_RISK[fd.source],
                    )
            for tid in self.tests_for(cid):
                td = self._tests[tid]
                actions.setdefault(tid, ActionSpec(id=tid, name=td.name, type=td.type, test=tid, cost=float(td.cost), risk=float(td.risk)))
        return list(actions.values())

    def finding_rules(self) -> list[tuple[str, NumericRule]]:
        """(finding code, rule) for numeric vital-sign rules (v0.3.1)."""
        return [(f.code, r) for f in self._findings.values() for r in f.rules]

    def numeric_rules(self) -> list[tuple[str, NumericRule]]:
        return [(t.id, r) for t in self._tests.values() for r in t.rules]


# ---------------------------------------------------------------------------
# Local ontology extension terms (v0.4)
# ---------------------------------------------------------------------------


class LocalTerm(BaseModel):
    """A clinical entity MONDO lacks (e.g. 'aortic dissection', 'COPD exacerbation'), anchored under MONDO parents.

    These are identity, not knowledge: a label, synonyms, and where the term sits in the hierarchy. Their cards
    are compiled from sources like any other disease."""

    id: str  # "DXA:0000001"
    name: str
    synonyms: list[str] = Field(default_factory=list)
    parents: list[str] = Field(default_factory=list)  # MONDO (or DXA) ids
    xrefs: dict[str, list[str]] = Field(default_factory=dict)
    note: str = ""


def load_local_terms(path: Path) -> list[LocalTerm]:
    return [LocalTerm.model_validate(r) for r in (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or [])]


def extend_with_local_terms(index: OntologyIndex, path: Path) -> int:
    """Add local terms to a loaded disease index (in memory). Returns the number added."""
    from dxagent.knowledge.ontology import norm_name, tokens

    n = 0
    for t in load_local_terms(path):
        if t.id in index.terms:
            continue
        parents = [p for p in t.parents if p in index.terms]
        index.terms[t.id] = [t.name, list(t.synonyms), [], parents, dict(t.xrefs), "", []]
        for p in parents:
            index._children[p].append(t.id)
        for name in [t.name, *t.synonyms]:
            key = norm_name(name)
            if key:
                index._exact[key].append(t.id)
            if index._tok_index is not None:
                for tok in tokens(name):
                    index._tok_index[tok].add(t.id)
        index._anc_cache.clear()
        n += 1
    return n
