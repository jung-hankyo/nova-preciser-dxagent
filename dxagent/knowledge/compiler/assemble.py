"""Claims -> cards (v0.4). Deterministic.

Reconciliation, per (disease, key):
    rank = (tier, precision, -directness, -edition year, -sample size)   lower is better
    the best-ranked claim is the entry; any other claim whose value differs by more than a factor of 3 in odds is
    recorded under `prov.conflicts` (never averaged), and a conflict at the same or a better tier multiplies the
    entry's confidence by 0.9.

Vocabulary: a finding maps onto (1) a registry code linked to the same HPO term, (2) an existing code with a
near-identical name (lexicon similarity >= dedup_auto), (3) an HPO term by exact label or synonym, or (4) a new
code. New HPO-derived codes carry the HPO id, the HPO synonyms (for lexical retrieval), a channel guessed from
the label, and a cluster at the nearest specific HPO parent (so a term and its propagated ancestor are not
counted twice by the belief update).

Background rates for new codes, in order: 1 - specificity stated by a cited source; else a mixture over the
compiled cards, bg(f) = sum_d w_d P(f|d) / sum_d w_d, with w_d the disease's prevalence bucket and the rare-disease
(HPO-only) cards together capped at `rare_share` of the total weight; clamped to [bg_min, bg_max]. Curated registry
backgrounds are never overwritten.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Iterable, Optional

from dxagent.config import PRIOR_BUCKET
from dxagent.knowledge.cards import (CardStore, DiseaseCard, EntryProvenance, FindingDef, FindingEntry, HarmProfile,
                                     Provenance, TestEntry, normalize, slug)
from dxagent.knowledge.compiler.claims import Claim, truncate_words
from dxagent.knowledge.compiler.hpo import channel_for
from dxagent.knowledge.compiler.sources import PRECISION_RANK, entry_conf
from dxagent.knowledge.ontology import OntologyIndex, curie_to_card_id

CONFLICT_LOGODDS = math.log(3.0)
CONFLICT_FACTOR = 0.9
CHANNEL_BY_CATEGORY = {"symptom": "history", "risk_factor": "history", "exposure": "history", "medication": "history",
                       "family_history": "history", "sign": "exam", "vital": "exam", "lab": "lab", "imaging": "imaging"}


def _year(edition: str) -> int:
    m = re.search(r"(19|20)\d{2}", edition or "")
    return int(m.group(0)) if m else 0


def rank(c: Claim) -> tuple:
    return (c.tier, PRECISION_RANK.get(c.precision, 2), -c.directness, -_year(c.edition), -(c.n or 0), c.source_id, c.locator)


def _logit(p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def reconcile(claims: Iterable[Claim]) -> dict[tuple, tuple[Claim, list[Claim]]]:
    """{key: (chosen claim, conflicting claims)}."""
    groups: dict[tuple, list[Claim]] = defaultdict(list)
    for c in claims:
        groups[c.key()].append(c)
    out = {}
    for key, cs in groups.items():
        cs = sorted(cs, key=rank)
        best = cs[0]
        conflicts = []
        if best.p is not None:
            for c in cs[1:]:
                if c.p is None or (c.source_id, c.locator) == (best.source_id, best.locator):
                    continue
                if abs(_logit(c.p) - _logit(best.p)) > CONFLICT_LOGODDS:
                    conflicts.append(c)
        out[key] = (best, conflicts)
    return out


class VocabularyMapper:
    """Maps claim findings onto finding codes; collects the new finding definitions a compile creates."""

    def __init__(self, store: CardStore, phenotype: Optional[OntologyIndex], auto: float = 0.85, min_term_depth: int = 3):
        self.store = store
        self.ph = phenotype
        self.auto = auto
        self.min_term_depth = min_term_depth
        self.new_defs: dict[str, FindingDef] = {}
        self.links: dict[str, list[str]] = defaultdict(list)  # registry code -> HPO ids linked by this compile
        self.hp_to_code: dict[str, str] = {}
        for code, fd in store._findings.items():
            for x in fd.xrefs:
                if x.startswith("HP:"):
                    self.hp_to_code.setdefault(x, code)
        if self.ph is not None:  # link curated registry codes to HPO by exact label or synonym
            for code, fd in list(store._findings.items()):
                if any(x.startswith("HP:") for x in fd.xrefs):
                    continue
                for name in (fd.label, code.replace("_", " ")):
                    tid, how, _ = self.ph.resolve(name)
                    if tid and how in ("exact", "synonym") and tid not in self.hp_to_code:
                        self.hp_to_code[tid] = code
                        self.links[code].append(tid)
                        break

    def _unique(self, code: str, hp: str = "") -> str:
        if code in self.new_defs and hp and hp not in self.new_defs[code].xrefs:
            return f"{code}_{hp.split(':')[-1]}"
        return code

    def for_hpo(self, hp: str) -> str:
        if hp in self.hp_to_code:
            return self.hp_to_code[hp]
        label = self.ph.label(hp) if self.ph is not None else hp
        code = slug(label) or slug(hp)
        existing = self.store.finding_def(self.store.canonical(code))
        if existing is not None:  # same name already curated: same finding
            self.hp_to_code[hp] = existing.code
            self.links[existing.code].append(hp)
            return existing.code
        code = self._unique(code, hp)
        channel, category = channel_for(label)
        cluster = code
        if self.ph is not None:
            parents = [p for p in self.ph.parents(hp) if self.ph.depth(p) >= self.min_term_depth + 1]
            if parents:
                cluster = "hpo_" + slug(self.ph.label(sorted(parents)[0]))
        self.new_defs[code] = FindingDef(code=code, label=label, source=channel, category=category, cluster=cluster,
                                         xrefs=[hp], synonyms=(self.ph.synonyms(hp)[:6] if self.ph is not None else []),
                                         background=0.15, background_source="default")
        self.hp_to_code[hp] = code
        return code

    def for_label(self, label: str, category: str = "symptom") -> str:
        code = slug(label)
        if not code:
            return ""
        canon = self.store.canonical(code)
        if self.store.finding_def(canon) is not None:
            return canon
        if code in self.new_defs:
            return code
        sims = self.store.lexicon().similar(code, label, k=1)
        if sims and sims[0][1] >= self.auto:
            return self.store.canonical(sims[0][0])
        if self.ph is not None:
            tid, how, _ = self.ph.resolve(label)
            if tid and how in ("exact", "synonym"):
                return self.for_hpo(tid)
        self.new_defs[code] = FindingDef(code=code, label=label, source=CHANNEL_BY_CATEGORY.get(category, "history"),
                                         category=category if category in CHANNEL_BY_CATEGORY else None, cluster=code,
                                         background=0.15, background_source="default")
        return code

    def finding_def(self, code: str) -> Optional[FindingDef]:
        return self.new_defs.get(code) or self.store.finding_def(code)


def _prov(c: Claim, conflicts: list[Claim]) -> EntryProvenance:
    return EntryProvenance(
        source_id=c.source_id, kind=c.source_kind, tier=c.tier, precision=c.precision, edition=c.edition, locator=c.locator,
        quote=truncate_words(c.quote, 30), population=c.population, n=c.n, directness=round(c.directness, 4),
        conflicts=[{"source_id": x.source_id, "edition": x.edition, "locator": x.locator, "value": round(x.p or 0.0, 4),
                    "tier": x.tier} for x in conflicts])


def _conf(c: Claim, conflicts: list[Claim], cap: Optional[float]) -> float:
    conf = entry_conf(c.tier, c.precision, c.directness)
    if any(x.tier <= c.tier for x in conflicts):
        conf *= CONFLICT_FACTOR
    if cap is not None:
        conf = min(conf, cap)
    return round(conf, 4)


def build_test_entries(chosen: dict[tuple, tuple[Claim, list[Claim]]], store: CardStore,
                       cap: Optional[float]) -> dict[str, TestEntry]:
    """Combine per-outcome-set test claims into one outcome distribution per test. Stated outcome sets get their
    stated mass (split within a set by background proportions); the remaining mass is spread over unstated outcomes
    by background proportions. Overlapping sets: the better-ranked claim wins."""
    by_test: dict[str, list[tuple[Claim, list[Claim]]]] = defaultdict(list)
    for key, (c, conf) in chosen.items():
        if key[0] == "test":
            by_test[c.test_id].append((c, conf))
    out: dict[str, TestEntry] = {}
    for tid, items in by_test.items():
        td = store.test_def(tid)
        if td is None:
            continue
        items.sort(key=lambda x: rank(x[0]))
        dist: dict[str, float] = {}
        used: list[tuple[Claim, list[Claim]]] = []
        for c, conf in items:
            outs = [o for o in c.positive if o in td.outcomes]
            if not outs or any(o in dist for o in outs) or c.p is None:
                continue
            bg = {o: td.background[o] for o in outs}
            z = sum(bg.values()) or 1.0
            for o in outs:
                dist[o] = c.p * bg[o] / z
            used.append((c, conf))
        if not used:
            continue
        stated = sum(dist.values())
        if stated > 1.0:
            dist = {o: v / stated for o, v in dist.items()}
            stated = 1.0
        rest = [o for o in td.outcomes if o not in dist]
        zr = sum(td.background[o] for o in rest) or 1.0
        for o in rest:
            dist[o] = (1.0 - stated) * td.background[o] / zr
        lead, lead_conf = used[0]
        conf = min(_conf(c, cf, cap) for c, cf in used)
        out[tid] = TestEntry(dist=normalize(dist), conf=conf, evidence=lead.citation(), prov=_prov(lead, lead_conf))
    return out


def build_card(curie: str, disease: Optional[OntologyIndex], claims: list[Claim], mapper: VocabularyMapper, store: CardStore,
               compiled_at: str, status: str = "compiled", conf_cap: Optional[float] = None,
               base_prevalence: Optional[str] = None, label: str = "") -> tuple[Optional[DiseaseCard], dict[str, Any]]:
    """Build one card from its claims. Returns (card or None, notes for the compile report)."""
    notes: dict[str, Any] = {"claims": len(claims)}
    for c in claims:
        if c.kind == "finding" and not c.code:
            c.code = mapper.for_hpo(c.hpo_id) if c.hpo_id else mapper.for_label(c.label, c.category)
    claims = [c for c in claims if c.kind != "finding" or c.code]
    chosen = reconcile(claims)
    findings: list[FindingEntry] = []
    for key, (c, conflicts) in chosen.items():
        if key[0] != "finding":
            continue
        fd = mapper.finding_def(c.code)
        findings.append(FindingEntry(code=c.code, label=fd.label if fd else c.label, source=fd.source if fd else "history",
                                     p=c.p, conf=_conf(c, conflicts, conf_cap), cluster=fd.cluster if fd else c.code,
                                     evidence=c.citation(), prov=_prov(c, conflicts)))
        if fd is not None and c.background is not None and fd.code in mapper.new_defs and fd.background_source != "claim":
            fd.background, fd.background_source = min(max(c.background, 1e-3), 1 - 1e-3), "claim"
    tests = build_test_entries(chosen, store, conf_cap)
    harm = HarmProfile()
    hc = chosen.get(("harm",))
    if hc:
        c = hc[0]
        x = c.extra
        harm = HarmProfile(untreated_course=x.get("untreated_course") or "serious", urgency=x.get("urgency") or "days",
                           treatments=sorted({slug(t) for t in x.get("treatments", []) if slug(t)}),
                           treatment_risk=x.get("treatment_risk") or "low", evidence=c.citation())
    mimics = sorted({c.extra.get("name", "") for k, (c, _) in chosen.items() if k[0] == "mimic"} - {""})
    if not findings and not tests:
        notes["empty"] = True
        return None, notes
    name = label or (disease.label(curie) if disease is not None else curie)
    aliases: list[str] = []
    parents: list[str] = []
    if disease is not None and curie in disease:
        aliases = [n for n in disease.synonyms(curie)[:11] if n.lower() != name.lower()]
        parents = disease.parents(curie)
    srcs: dict[str, dict[str, str]] = {}
    for c in claims:
        srcs.setdefault(c.source_id, {"id": c.source_id, "edition": c.edition, "kind": c.source_kind})
    card = DiseaseCard(
        id=curie_to_card_id(curie), name=name, aliases=aliases, harm=harm, findings=sorted(findings, key=lambda f: f.code),
        tests=tests, mimics=mimics, ontology_id=curie, parents=parents, base_prevalence=base_prevalence,
        provenance=Provenance(status=status, source=", ".join(sorted(srcs)), version="0.4", sources=sorted(srcs.values(), key=lambda x: x["id"]),
                              compiled_at=compiled_at))
    notes.update({"findings": len(findings), "tests": len(tests), "conflicts": sum(len(v[1]) for v in chosen.values()),
                  "harm_sourced": bool(hc)})
    return card, notes


def derive_backgrounds(cards: list[DiseaseCard], defs: dict[str, FindingDef], rare_share: float = 0.05,
                       bg_min: float = 0.02, bg_max: float = 0.5) -> dict[str, float]:
    """Fill `default` backgrounds of new finding definitions with the prevalence-weighted card mixture."""
    targets = {c for c, fd in defs.items() if fd.background_source == "default"}
    if not targets or not cards:
        return {}
    rare_ids = {c.id for c in cards if {s.get("kind") for s in c.provenance.sources} <= {"curated_db"}}
    rare = [c for c in cards if c.id in rare_ids]
    other = [c for c in cards if c.id not in rare_ids]
    w: dict[str, float] = {c.id: PRIOR_BUCKET.get(c.base_prevalence or "uncommon", PRIOR_BUCKET["uncommon"]) for c in other}
    w_other = sum(w.values())
    if rare:
        total_rare = (rare_share / (1 - rare_share)) * w_other if w_other > 0 else 1.0
        for c in rare:
            w[c.id] = total_rare / len(rare)
    w_total = sum(w.values()) or 1.0
    num: dict[str, float] = defaultdict(float)
    for c in cards:
        for f in c.findings:
            if f.code in targets:
                num[f.code] += w[c.id] * f.p
    out = {}
    for code in targets:
        bg = min(max(num.get(code, 0.0) / w_total, bg_min), bg_max)
        defs[code].background, defs[code].background_source = bg, "derived"
        out[code] = bg
    return out
