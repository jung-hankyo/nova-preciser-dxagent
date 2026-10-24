"""Source manifest and evidence tiers (v0.4).

Every value in a compiled card is traced to one entry of `sources/manifest.yaml`. The manifest records what a
source is, where its file lives, which edition it is, under which licence it is held, and which diseases it
covers. The tier comes from the kind of source, and the entry confidence comes from the tier and how the value
was stated, by a fixed table: no LLM judges how much to trust a value.

    tier 1   guideline, systematic_review (incl. diagnostic-accuracy reviews), rce (JAMA Rational Clinical Exam)
    tier 2   curated_db (HPO / Orphanet annotations), reference (StatPearls, GeneReviews, NCI PDQ, textbooks)
    tier 3   primary (single studies), abstract (online fallback: abstracts only)

    precision  numeric      a number stated for this disease (percent, n of m, sensitivity, likelihood ratios)
               class        a bounded frequency class (HPO/Orphanet: very frequent 80-99%, frequent 30-79% ...)
               qualitative  a frequency word ("usually", "rarely"), mapped to a value by a fixed lexicon

    confidence = CONF_BASE[tier][precision] x directness (population match 1.0, unknown 0.9, mismatch 0.8)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

SourceKind = Literal["guideline", "systematic_review", "rce", "reference", "curated_db", "primary", "abstract"]
TIER: dict[str, int] = {"guideline": 1, "systematic_review": 1, "rce": 1, "curated_db": 2, "reference": 2,
                        "primary": 3, "abstract": 3}
CONF_BASE: dict[int, dict[str, float]] = {
    1: {"numeric": 0.95, "class": 0.85, "qualitative": 0.80},
    2: {"numeric": 0.85, "class": 0.75, "qualitative": 0.70},
    3: {"numeric": 0.70, "class": 0.60, "qualitative": 0.55},
}
DIRECTNESS = {"match": 1.0, "unknown": 0.9, "mismatch": 0.8}
PRECISION_RANK = {"numeric": 0, "class": 1, "qualitative": 2}


def entry_conf(tier: int, precision: str, directness: float) -> float:
    return round(CONF_BASE[min(max(tier, 1), 3)][precision] * directness, 4)


def population_directness(claim_pop: str, source_pop: str, target: str) -> tuple[str, float]:
    """('match' | 'unknown' | 'mismatch', factor). The claim's own population wins over the source's."""
    pop = claim_pop if claim_pop and claim_pop != "unknown" else (source_pop or "unknown")
    if pop == "unknown":
        return "unknown", DIRECTNESS["unknown"]
    if pop == target or pop == "mixed" or target in ("", "any"):
        return "match", DIRECTNESS["match"]
    return "mismatch", DIRECTNESS["mismatch"]


class SourceDoc(BaseModel):
    id: str
    kind: SourceKind
    title: str = ""
    publisher: str = ""
    edition: str = ""  # year or date of the edition used; the newer edition wins within a tier
    licence: str = ""  # e.g. "CC BY 4.0", "institutional subscription", "public domain"
    access: str = ""  # open | subscription | institutional
    path: str = ""  # relative to the sources directory (pdf, html, txt, md; or hpoa for curated_db)
    url: str = ""
    diseases: list[str] = Field(default_factory=list)  # names or ontology ids this source is compiled for
    population: Literal["adult", "pediatric", "mixed", "unknown"] = "unknown"
    region: str = ""
    tier: Optional[int] = None  # override; normally derived from kind
    sha256: str = ""  # filled by `kb sources --check`
    notes: str = ""

    @model_validator(mode="after")
    def _fill(self) -> "SourceDoc":
        if self.tier is None:
            self.tier = TIER[self.kind]
        return self


class OntologySource(BaseModel):
    id: str  # mondo | hpo | hpo_annotations
    path: str
    url: str = ""
    version: str = ""
    sha256: str = ""
    fetched_at: str = ""


class Manifest(BaseModel):
    version: int = 1
    ontologies: list[OntologySource] = Field(default_factory=list)
    sources: list[SourceDoc] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(MANIFEST_HEADER + yaml.safe_dump(self.model_dump(mode="json", exclude_defaults=True), sort_keys=False,
                                                          allow_unicode=True), encoding="utf-8")

    def source(self, sid: str) -> Optional[SourceDoc]:
        return next((s for s in self.sources if s.id == sid), None)

    def ontology(self, oid: str) -> Optional[OntologySource]:
        return next((o for o in self.ontologies if o.id == oid), None)

    def upsert_ontology(self, o: OntologySource) -> None:
        self.ontologies = [x for x in self.ontologies if x.id != o.id] + [o]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


MANIFEST_HEADER = """# dxagent source manifest (v0.4). Every compiled value cites one of these entries.
# kind: guideline | systematic_review | rce | reference | curated_db | primary | abstract
# Files live under the sources directory, which is gitignored: keep subscription material out of the repository.
# Example document entry:
#   - id: esc_pe_2019
#     kind: guideline
#     title: 2019 ESC Guidelines for the diagnosis and management of acute pulmonary embolism
#     publisher: European Society of Cardiology
#     edition: "2019"
#     licence: "publisher copyright; held for private research use"
#     access: open
#     path: docs/esc_pe_2019.pdf
#     diseases: [pulmonary embolism]
#     population: adult
"""
