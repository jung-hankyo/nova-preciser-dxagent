"""HPO disease annotations -> finding claims (v0.4). Deterministic, no LLM.

`phenotype.hpoa` (HPO release) lists, for OMIM and Orphanet diseases, which phenotypes occur and how often:
an HPO frequency class (Orphanet curation), a count n/m (case reports and series), or a percentage. Rows are
joined onto MONDO diseases through MONDO's OMIM/Orphanet cross-references.

    aspect P only (phenotypic abnormality); inheritance, onset and modifier rows are skipped
    NOT qualifier       -> an excluded feature: p = 0.01 (a pertinent negative), never propagated
    class / percentage  -> claims.HPO_FREQUENCY / percent
    n of m              -> numeric when m >= 10, else treated as a frequency class (small series are not precise)
    no frequency        -> p = 0.35 with an extra 0.85 confidence factor (observed, frequency not stated)

True-path propagation: a disease annotated with a specific term also has every ancestor term at least that often,
so an ancestor within `propagate_depth` levels that is itself specific enough (at least `min_term_depth` below
'Phenotypic abnormality') gets the maximum of its annotated descendants, with confidence x0.9. A patient finding
coded at a coarser level than the annotation can then still meet the card.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from dxagent.knowledge.compiler.claims import HPO_FREQUENCY, Claim
from dxagent.knowledge.compiler.sources import SourceDoc
from dxagent.knowledge.ontology import OntologyIndex

UNKNOWN_FREQ_P = 0.35
UNKNOWN_FREQ_FACTOR = 0.85
PROPAGATED_FACTOR = 0.9
SMALL_SERIES = 10


def read_hpoa(path: Path) -> tuple[str, Iterable[dict[str, str]]]:
    """(version, row iterator). Rows are dicts keyed by the header line; the file is streamed."""
    path = Path(path)
    version = ""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if line.startswith("#version:"):
                version = line.split(":", 1)[1].strip()

    def rows() -> Iterable[dict[str, str]]:
        with open(path, encoding="utf-8") as fh:
            header: Optional[list[str]] = None
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if header is None:
                    header = parts
                    continue
                yield dict(zip(header, parts))

    return version, rows()


def parse_frequency(freq: str) -> tuple[Optional[float], str, Optional[int]]:
    """(p, precision, sample size) for an hpoa frequency field."""
    f = (freq or "").strip()
    if not f:
        return None, "unknown", None
    if f in HPO_FREQUENCY:
        return HPO_FREQUENCY[f], "class", None
    m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", f)
    if m:
        n, total = int(m.group(1)), int(m.group(2))
        if total <= 0 or n > total:
            return None, "unknown", None
        p = (n + 0.5) / (total + 1)
        return p, ("numeric" if total >= SMALL_SERIES else "class"), total
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", f)
    if m:
        return min(float(m.group(1)) / 100.0, 1.0), "numeric", None
    return None, "unknown", None


def disease_map(disease: OntologyIndex) -> dict[str, list[str]]:
    """hpoa database_id ('OMIM:154700', 'ORPHA:558') -> MONDO ids."""
    out: dict[str, list[str]] = {}
    for ref, ids in disease.xref_map("OMIM").items():
        out[ref] = ids
    for ref, ids in disease.xref_map("Orphanet").items():
        out["ORPHA:" + ref.split(":", 1)[1]] = ids
    return out


def hpo_claims(hpoa_path: Path, source: SourceDoc, disease: OntologyIndex, phenotype: Optional[OntologyIndex],
               scope: Optional[set[str]] = None, min_annotations: int = 3, propagate_depth: int = 2,
               min_term_depth: int = 3) -> tuple[dict[str, list[Claim]], dict[str, int]]:
    """Return ({MONDO id: claims}, stats). `scope`: only these MONDO ids (None: every mapped disease)."""
    version, rows = read_hpoa(hpoa_path)
    edition = source.edition or version
    dmap = disease_map(disease)
    stats = defaultdict(int)
    per: dict[str, dict[str, list[Claim]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        stats["rows"] += 1
        if r.get("aspect") != "P":
            continue
        dbid = r.get("database_id", "")
        mondo_ids = dmap.get(dbid)
        if not mondo_ids:
            stats["unmapped_rows"] += 1
            continue
        hp = r.get("hpo_id", "")
        if phenotype is not None:
            hp = phenotype.canonical_id(hp)
            if hp not in phenotype.terms:
                stats["unknown_terms"] += 1
                continue
        negated = r.get("qualifier", "") == "NOT"
        if negated:
            p, precision, n, factor = 0.01, "class", None, 1.0
        else:
            p, precision, n = parse_frequency(r.get("frequency", ""))
            factor = 1.0
            if p is None:
                p, precision, factor = UNKNOWN_FREQ_P, "qualitative", UNKNOWN_FREQ_FACTOR
        for mid in mondo_ids:
            if scope is not None and mid not in scope:
                continue
            label = phenotype.label(hp) if phenotype is not None else hp
            per[mid][hp].append(Claim(
                disease=mid, kind="finding", source_id=source.id, source_kind=source.kind, tier=source.tier or 2,
                edition=edition, precision=precision, locator=f"{dbid} {r.get('reference', '')}".strip(),
                quote="", population="unknown", directness=factor, n=n, label=label, hpo_id=hp, p=p,
                extra={"negated": negated, "frequency": r.get("frequency", ""), "evidence_code": r.get("evidence", "")}))
            stats["claims"] += 1
    out: dict[str, list[Claim]] = {}
    for mid, terms in per.items():
        if len(terms) < min_annotations:
            stats["below_min_annotations"] += 1
            continue
        claims = [c for cs in terms.values() for c in cs]
        if phenotype is not None and propagate_depth > 0:
            claims += _propagate(mid, terms, phenotype, source, edition, propagate_depth, min_term_depth)
        out[mid] = claims
    stats["diseases"] = len(out)
    return out, dict(stats)


def _propagate(mid: str, terms: dict[str, list[Claim]], phenotype: OntologyIndex, source: SourceDoc, edition: str,
               depth: int, min_term_depth: int) -> list[Claim]:
    best: dict[str, Claim] = {}
    for hp, cs in terms.items():
        positive = [c for c in cs if not c.extra.get("negated")]
        if not positive:
            continue
        top = max(positive, key=lambda c: c.p or 0.0)
        for anc, d in phenotype.ancestors(hp).items():
            if d > depth or anc in terms or phenotype.depth(anc) < min_term_depth:
                continue
            cur = best.get(anc)
            if cur is None or (top.p or 0) > (cur.p or 0):
                best[anc] = Claim(
                    disease=mid, kind="finding", source_id=source.id, source_kind=source.kind, tier=source.tier or 2,
                    edition=edition, precision=top.precision, locator=f"propagated from {hp} ({top.locator})",
                    population="unknown", directness=top.directness * PROPAGATED_FACTOR, n=top.n,
                    label=phenotype.label(anc), hpo_id=anc, p=top.p, extra={"propagated_from": hp})
    return list(best.values())


SYMPTOM_WORDS = re.compile(r"\b(symptom|pain|ache|headache|fatigue|malaise|nausea|vomiting|diarrh|constipation|dyspnea|cough|fever|"
                           r"pruritus|itch|palpitation|syncope|dizziness|vertigo|weight loss|anorexia|insomnia|anxiety|"
                           r"depress|hallucination|paresthesia|numbness|weakness|dysphagia|hemoptysis|hematuria|"
                           r"polyuria|polydipsia|chills|sweat|photophobia|tinnitus|seizure|tremor|confusion)", re.I)
LAB_WORDS = re.compile(r"\b(concentration|level|count|serum|plasma|urine|urinary|circulating|activity|titer|antibod|"
                       r"blood|ratio|clearance|excretion)\b", re.I)
IMAGING_WORDS = re.compile(r"\b(on mri|on ct|radiograph|imaging|cortical|white matter|hypoplasia|atrophy of the|"
                           r"agenesis|dysplasia|cyst|calcification)\b", re.I)


def channel_for(label: str) -> tuple[str, str]:
    """(action channel, category) for an HPO phenotype label. Heuristic; HPO does not record how a phenotype is
    observed. Symptoms are asked, laboratory phenotypes are measured, structural phenotypes need imaging, the rest
    are examined."""
    if LAB_WORDS.search(label):
        return "lab", "lab"
    if SYMPTOM_WORDS.search(label):
        return "history", "symptom"
    if IMAGING_WORDS.search(label):
        return "imaging", "imaging"
    return "exam", "sign"
