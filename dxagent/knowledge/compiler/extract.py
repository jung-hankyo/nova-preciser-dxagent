"""Document passages -> verified claims (v0.4).

The LLM is used once per (source edition, disease) to point at statements: it copies quotes and the numbers they
contain. Its output is cached by the source file's hash, so a recompile is deterministic and costs nothing. Every
claim then passes the deterministic checks in verify.py; the value is computed there, never taken from the LLM.

A source covering several diseases must name the disease inside the quote (a figure in a comparison sentence may
belong to the other disease); a source devoted to one disease need not repeat its name.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from dxagent.knowledge.cards import CardStore, slug
from dxagent.knowledge.compiler.claims import Claim
from dxagent.knowledge.compiler.documents import Passage
from dxagent.knowledge.compiler.sources import SourceDoc, population_directness
from dxagent.knowledge.compiler.verify import derive, span_found, subject_mentioned, verify_claim
from dxagent.llm import LLMLike
from dxagent.prompts import EXTRACT_CLAIMS_SYSTEM, extract_claims_user
from dxagent.schemas import ClaimExtraction

log = logging.getLogger(__name__)
EXTRACTOR_VERSION = "1"
SMALL_SERIES = 10


class ClaimExtractor:
    def __init__(self, llm: Optional[LLMLike], cache_dir: Optional[Path] = None):
        self.llm = llm
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _cache_path(self, sha: str, curie: str) -> Optional[Path]:
        if not self.cache_dir or not sha:
            return None
        return self.cache_dir / f"{sha[:16]}__{curie.replace(':', '_')}__v{EXTRACTOR_VERSION}.json"

    def extract(self, source: SourceDoc, sha: str, curie: str, disease: str, synonyms: list[str], passages: list[Passage],
                test_lines: list[str]) -> tuple[Optional[ClaimExtraction], str]:
        """(extraction or None, how) with how in {cache, llm, no_llm, no_passages, error}."""
        if not passages:
            return None, "no_passages"
        path = self._cache_path(sha, curie)
        if path and path.exists():
            return ClaimExtraction.model_validate_json(path.read_text(encoding="utf-8")), "cache"
        if self.llm is None:
            return None, "no_llm"
        try:
            out = self.llm.structured(ClaimExtraction, EXTRACT_CLAIMS_SYSTEM,
                                      extract_claims_user(disease, synonyms, [(p.locator, p.text) for p in passages], test_lines))
        except Exception as exc:
            log.warning("Extraction failed for %s / %s: %s", source.id, curie, exc)
            return None, "error"
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(out.model_dump_json(indent=1), encoding="utf-8")
        return out, "llm"


def verify_extraction(ex: ClaimExtraction, source: SourceDoc, curie: str, disease_names: list[str], source_norm: str,
                      store: CardStore, target_population: str, require_disease_mention: bool,
                      edition: str = "") -> tuple[list[Claim], list[dict[str, Any]]]:
    claims: list[Claim] = []
    rejected: list[dict[str, Any]] = []
    tier = source.tier or 3
    edition = edition or source.edition

    def reject(kind: str, item: str, why: str) -> None:
        rejected.append({"disease": curie, "source": source.id, "kind": kind, "item": item[:80], "reason": why})

    def base(kind: str, quote: str, locator: str, population: str) -> dict[str, Any]:
        _, d = population_directness(population, source.population, target_population)
        return dict(disease=curie, kind=kind, source_id=source.id, source_kind=source.kind, tier=tier, edition=edition,
                    locator=locator, quote=quote, population=population if population != "unknown" else source.population,
                    directness=d)

    def about_disease(quote: str) -> bool:
        return not require_disease_mention or subject_mentioned(quote, disease_names)

    for f in ex.findings:
        nums = {"percent": f.percent, "n": f.n, "m": f.m, "sensitivity": f.sensitivity, "specificity": f.specificity,
                "lr_positive": f.lr_positive, "lr_negative": f.lr_negative}
        why = verify_claim(f.quote, source_norm, [f.finding], nums)
        if why is None and not about_disease(f.quote):
            why = "quote does not name the disease (multi-disease source)"
        if why:
            reject("finding", f.finding, why)
            continue
        p, bg, precision, size, how = derive(f.quote, f.percent, f.n, f.m, f.sensitivity, f.specificity, f.lr_positive, f.lr_negative,
                                             subject_names=[f.finding])
        if p is None:
            reject("finding", f.finding, how)
            continue
        if precision == "numeric" and size is not None and size < SMALL_SERIES:
            precision = "class"
        claims.append(Claim(**base("finding", f.quote, f.locator, f.population), precision=precision, n=size,
                            label=f.finding, category=f.category, p=p, background=bg, extra={"derived_from": how}))

    for t in ex.tests:
        td = store.test_def(slug(t.test_id))
        if td is None:
            reject("test", t.test_id, "test not in the registry")
            continue
        positive = [slug(o) for o in t.positive_outcomes if slug(o) in td.outcomes]
        if not positive:
            reject("test", t.test_id, "no listed outcome named as positive")
            continue
        nums = {"percent": t.percent, "sensitivity": t.sensitivity, "specificity": t.specificity,
                "lr_positive": t.lr_positive, "lr_negative": t.lr_negative}
        names = [td.name, td.id.replace("_", " ").replace("lab ", "").replace("img ", "").replace("test ", ""), *td.aliases]
        why = verify_claim(t.quote, source_norm, names, nums)
        if why is None and not about_disease(t.quote):
            why = "quote does not name the disease (multi-disease source)"
        if why:
            reject("test", t.test_id, why)
            continue
        p, bg, precision, size, how = derive(t.quote, t.percent, None, None, t.sensitivity, t.specificity, t.lr_positive, t.lr_negative,
                                             subject_names=names)
        if p is None:
            reject("test", t.test_id, how)
            continue
        claims.append(Claim(**base("test", t.quote, t.locator, t.population), precision=precision, n=size, test_id=td.id,
                            positive=positive, p=p, background=bg, extra={"derived_from": how}))

    if ex.harm is not None and (ex.harm.untreated_course or ex.harm.urgency or ex.harm.treatments):
        if span_found(ex.harm.quote, source_norm):
            claims.append(Claim(**base("harm", ex.harm.quote, ex.harm.locator, "unknown"), precision="qualitative",
                                extra=ex.harm.model_dump(exclude={"quote", "locator"})))
        else:
            reject("harm", "harm profile", "quote not found verbatim in the source")

    for m in ex.mimics:
        if m.quote and not span_found(m.quote, source_norm):
            reject("mimic", m.name, "quote not found verbatim in the source")
            continue
        if not m.quote:
            reject("mimic", m.name, "no supporting quote")
            continue
        claims.append(Claim(**base("mimic", m.quote, m.locator, "unknown"), precision="qualitative", extra={"name": m.name}))
    return claims, rejected


def dump_rejections(rejected: list[dict[str, Any]], limit: int = 2000) -> str:
    return json.dumps(rejected[:limit], indent=1)
