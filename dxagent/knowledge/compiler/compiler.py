"""`dxagent kb compile`: sources -> kb/ (v0.4).

    1. load the ontology indexes (kb/ontology, from `kb fetch`) and the local extension terms
    2. resolve every disease named in the manifest (and in an optional list) to an ontology id
    3. claims: HPO annotations (structured) and documents (LLM extraction, cached, then verified)
    4. per disease: reconcile claims and build the card; new finding codes get derived backgrounds
    5. write kb/cards.jsonl, kb/findings.jsonl, kb/tests.jsonl, kb/build.json, kb/compile_report.json

The build record lists every source with its kind, tier, edition, licence and SHA-256, and the ontology versions,
so a compiled knowledge base can always be traced back to exactly what it was built from.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from dxagent.config import Settings
from dxagent.knowledge.cards import CardStore, DiseaseCard, card_to_row
from dxagent.knowledge.compiler.assemble import VocabularyMapper, build_card, derive_backgrounds
from dxagent.knowledge.compiler.claims import Claim
from dxagent.knowledge.compiler.documents import Passage, full_text, load_passages, select_passages
from dxagent.knowledge.compiler.extract import ClaimExtractor, verify_extraction
from dxagent.knowledge.compiler.hpo import hpo_claims
from dxagent.knowledge.compiler.sources import Manifest, SourceDoc, sha256_file
from dxagent.knowledge.compiler.verify import norm_text
from dxagent.knowledge.ontology import OntologyIndex, OntologySet, iter_names
from dxagent.llm import LLMLike

COMPILER_VERSION = "0.4.0"
DOC_SUFFIXES = {".pdf", ".html", ".htm", ".txt", ".md"}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _write_jsonl(path: Path, rows) -> int:
    n = 0
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


class KnowledgeCompiler:
    def __init__(self, settings: Settings, llm: Optional[LLMLike] = None, progress=print):
        self.s = settings
        self.llm = llm
        self.progress = progress

    # -- setup -------------------------------------------------------------------------------------------------

    def load_ontology(self) -> OntologySet:
        onto = OntologySet.load_dir(self.s.kb_dir / "ontology")
        if onto.disease is None:
            raise RuntimeError(f"No disease ontology index in {self.s.kb_dir / 'ontology'}. Run `dxagent kb fetch` first.")
        return onto

    def vocabulary_store(self, onto: OntologySet) -> CardStore:
        """Registry only (no compiled cards): the curated vocabulary a compile maps findings onto."""
        return CardStore(self.s.registry_dir, ontology=onto)

    @staticmethod
    def resolve_names(names: list[str], index: OntologyIndex) -> tuple[dict[str, str], list[dict[str, str]]]:
        ok: dict[str, str] = {}
        bad: list[dict[str, str]] = []
        for n in names:
            tid, how, score = index.resolve(n)
            if tid and how != "ambiguous":
                ok[n] = tid
            else:
                bad.append({"name": n, "how": how, "candidate": tid or "", "score": round(score, 3)})
        return ok, bad

    def disease_names(self, index: OntologyIndex, curie: str) -> list[str]:
        return list(iter_names(index, curie, 12))

    # -- documents --------------------------------------------------------------------------------------------

    def document_claims(self, doc: SourceDoc, passages: list[Passage], sha: str, targets: list[str], onto: OntologySet,
                        store: CardStore, extractor: ClaimExtractor, report: dict[str, Any]) -> list[Claim]:
        source_norm = norm_text(full_text(passages))
        test_lines = [f"{t.id}: {', '.join(t.outcomes)}" + (f" (also: {', '.join(t.aliases)})" if t.aliases else "")
                      for t in store.tests()]
        multi = len(targets) > 1
        out: list[Claim] = []
        for curie in targets:
            names = self.disease_names(onto.disease, curie)
            sel = select_passages(passages, names, require_name=multi)
            ex, how = extractor.extract(doc, sha, curie, names[0] if names else curie, names[1:], sel, test_lines)
            report["extractions"].append({"source": doc.id, "disease": curie, "how": how, "passages": len(sel)})
            if ex is None:
                continue
            claims, rejected = verify_extraction(ex, doc, curie, names, source_norm, store, self.s.target_population,
                                                 require_disease_mention=multi)
            report["rejected"].extend(rejected)
            out.extend(claims)
        return out

    # -- compile ----------------------------------------------------------------------------------------------

    def compile(self, extra_diseases: Optional[list[str]] = None, hpo_scope: Optional[str] = None,
                write: bool = True) -> dict[str, Any]:
        s = self.s
        onto = self.load_ontology()
        manifest = Manifest.load(s.sources_dir / "manifest.yaml")
        store = self.vocabulary_store(onto)  # also extends the disease index with local terms
        mapper = VocabularyMapper(store, onto.phenotype, s.dedup_auto, s.hpo_min_term_depth)
        extractor = ClaimExtractor(self.llm, s.kb_dir / ".extract_cache")
        compiled_at = _now()
        report: dict[str, Any] = {"compiled_at": compiled_at, "unresolved": [], "missing_files": [], "extractions": [],
                                  "rejected": [], "cards": {}, "hpo": {}}
        hpo_scope = hpo_scope or s.hpo_scope

        docs = [d for d in manifest.sources if d.kind != "curated_db"]
        doc_targets: dict[str, list[str]] = {}
        for d in docs:
            ok, bad = self.resolve_names(d.diseases, onto.disease)
            doc_targets[d.id] = list(dict.fromkeys(ok.values()))
            report["unresolved"] += [{**b, "source": d.id} for b in bad]
        listed: set[str] = set()
        if extra_diseases:
            ok, bad = self.resolve_names(extra_diseases, onto.disease)
            listed = set(ok.values())
            report["unresolved"] += [{**b, "source": "disease_list"} for b in bad]

        claims_by: dict[str, list[Claim]] = defaultdict(list)
        hpo_src = manifest.source("hpo_annotations")
        if hpo_src is not None and hpo_scope != "none":
            path = s.sources_dir / hpo_src.path
            if path.exists():
                scope = None if hpo_scope == "all" else (listed | {c for t in doc_targets.values() for c in t})
                hc, stats = hpo_claims(path, hpo_src, onto.disease, onto.phenotype, scope, s.hpo_min_annotations,
                                       s.hpo_propagate_depth, s.hpo_min_term_depth)
                report["hpo"] = stats
                for k, v in hc.items():
                    claims_by[k].extend(v)
                self.progress(f"HPO annotations: {stats.get('diseases', 0)} diseases, {stats.get('claims', 0)} claims")
            else:
                report["missing_files"].append(str(path))

        for d in docs:
            path = s.sources_dir / d.path if d.path else None
            if path is None or not path.exists() or path.suffix.lower() not in DOC_SUFFIXES:
                report["missing_files"].append(str(path) if path else f"{d.id}: no path")
                continue
            if not doc_targets.get(d.id):
                continue
            sha = sha256_file(path)
            passages = load_passages(path)
            got = self.document_claims(d, passages, sha, doc_targets[d.id], onto, store, extractor, report)
            for c in got:
                claims_by[c.disease].append(c)
            self.progress(f"{d.id}: {len(passages)} passages, {len(got)} verified claims for {len(doc_targets[d.id])} disease(s)")

        cards: list[DiseaseCard] = []
        for curie in sorted(claims_by):
            cs = claims_by[curie]
            only_db = all(c.source_kind == "curated_db" for c in cs)
            card, notes = build_card(curie, onto.disease, cs, mapper, store, compiled_at, status="compiled",
                                     base_prevalence="very_rare" if only_db else None)
            report["cards"][curie] = notes
            if card is not None:
                cards.append(card)
        backgrounds = derive_backgrounds(cards, mapper.new_defs, s.derived_bg_rare_share, s.derived_bg_min, s.derived_bg_max)

        used_ids = {src["id"] for c in cards for src in c.provenance.sources}
        build = {
            "compiler": COMPILER_VERSION,
            "compiled_at": compiled_at,
            "target_population": s.target_population,
            "hpo_scope": hpo_scope,
            "ontologies": [o.model_dump() for o in manifest.ontologies],
            "ontology_index_versions": {"disease": onto.disease.version, "phenotype": onto.phenotype.version if onto.phenotype else ""},
            "sources": [{"id": d.id, "kind": d.kind, "tier": d.tier, "edition": d.edition, "licence": d.licence,
                         "sha256": sha256_file(s.sources_dir / d.path) if d.path and (s.sources_dir / d.path).exists() else "",
                         "used": d.id in used_ids} for d in manifest.sources],
            "counts": {"cards": len(cards), "new_findings": len(mapper.new_defs), "derived_backgrounds": len(backgrounds),
                       "registry_hpo_links": sum(len(v) for v in mapper.links.values()),
                       "rejected_claims": len(report["rejected"]), "unresolved_diseases": len(report["unresolved"])},
        }
        if write:
            kb = s.kb_dir
            kb.mkdir(parents=True, exist_ok=True)
            for stale in ("cards.jsonl", "cards.jsonl.gz"):
                if (kb / stale).exists():
                    (kb / stale).unlink()
            _write_jsonl(kb / "cards.jsonl.gz", (card_to_row(c) for c in cards))
            rows = [fd.model_dump(mode="json", exclude_defaults=True) | {"code": fd.code} for fd in mapper.new_defs.values()]
            rows += [{"code": code, "xrefs": hps} for code, hps in mapper.links.items()]
            _write_jsonl(kb / "findings.jsonl", rows)
            _write_jsonl(kb / "tests.jsonl", [])
            (kb / "build.json").write_text(json.dumps(build, indent=1), encoding="utf-8")
            report_out = {**report, "rejected": report["rejected"][:5000]}
            (kb / "compile_report.json").write_text(json.dumps(report_out, indent=1), encoding="utf-8")
            self.progress(f"wrote {len(cards)} cards, {len(mapper.new_defs)} new finding codes -> {kb}")
        return {"build": build, "report": report, "cards": cards}
