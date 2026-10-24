"""Local-document retrieval (v0.4.1): the PRIMARY fallback for a disease with no knowledge.

When a hypothesis resolves in the ontology but has no card (own or inherited), the agent searches the documents
already downloaded into `sources/` (every manifest entry that is not a curated database), takes the passages that
name the disease, and runs the same extract -> verify -> assemble pipeline as `dxagent kb compile`:

    search      passages that name the disease (label and ontology synonyms) and carry frequency or accuracy
                cues; a document scoped to this disease in the manifest is searched by cues alone
    extract     LLM quotes and stated numbers, cached under kb/.extract_cache by (file hash, disease), the SAME
                cache the offline compiler uses, so a later `kb compile` reuses the work and vice versa
    verify      verbatim span, numbers in the quote, subject named, disease named for multi-disease documents
    assemble    tier-table confidence, higher tier wins, conflicts recorded

The card is marked `on_demand`: same sources and checks as a compiled card, so no confidence cap, and it may be
used by the reverse index; it is cached under cache/on_demand/ and reported in the case's coverage log. The online
PubMed fallback (online.py) runs only after this one finds nothing, and only if DX_LIVE_FALLBACK=1.

Manifest entries with an empty `diseases` list are general references (a textbook, a collection of chapters): the
offline compiler skips them, and this path searches them whenever a disease is missing.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Optional

from dxagent.config import Settings
from dxagent.knowledge.cards import CardStore, DiseaseCard
from dxagent.knowledge.compiler.assemble import VocabularyMapper, build_card
from dxagent.knowledge.compiler.documents import Passage, full_text, load_passages, select_passages
from dxagent.knowledge.compiler.extract import ClaimExtractor, verify_extraction
from dxagent.knowledge.compiler.sources import Manifest, SourceDoc, sha256_file
from dxagent.knowledge.compiler.verify import norm_text
from dxagent.knowledge.ontology import iter_names, norm_name
from dxagent.llm import LLMLike

log = logging.getLogger(__name__)
DOC_SUFFIXES = {".pdf", ".html", ".htm", ".txt", ".md"}


class _Doc:
    def __init__(self, src: SourceDoc, path: Path, sha: str, passages: list[Passage]):
        self.src, self.path, self.sha, self.passages = src, path, sha, passages
        self.norm = norm_text(full_text(passages))
        self.low = " " + norm_name(full_text(passages)) + " "


class LocalDocumentIndex:
    """The downloaded documents, parsed once (parsed passages cached by file hash under kb/.passage_cache)."""

    def __init__(self, sources_dir: Path, cache_dir: Optional[Path] = None):
        self.sources_dir = Path(sources_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._docs: Optional[list[_Doc]] = None

    def _passages(self, path: Path, sha: str) -> list[Passage]:
        cp = self.cache_dir / f"{sha[:16]}.json" if self.cache_dir else None
        if cp and cp.exists():
            return [Passage(**p) for p in json.loads(cp.read_text(encoding="utf-8"))]
        ps = load_passages(path)
        if cp:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps([p.__dict__ for p in ps]), encoding="utf-8")
        return ps

    def docs(self) -> list[_Doc]:
        if self._docs is None:
            out = []
            manifest = Manifest.load(self.sources_dir / "manifest.yaml")
            for src in manifest.sources:
                if src.kind == "curated_db" or not src.path:
                    continue
                path = self.sources_dir / src.path
                if not path.exists() or path.suffix.lower() not in DOC_SUFFIXES:
                    continue
                try:
                    sha = sha256_file(path)
                    out.append(_Doc(src, path, sha, self._passages(path, sha)))
                except Exception as exc:  # one unreadable file must not disable the others
                    log.warning("Skipping source %s: %s", src.id, exc)
            self._docs = out
        return self._docs

    def candidates(self, names: list[str], curie: str, max_docs: int) -> list[tuple[_Doc, bool]]:
        """(document, scoped to this disease) for documents that name the disease, best first: scoped documents,
        then lower tier, then more mentions."""
        keys = [k for k in {norm_name(n) for n in names} if len(k) >= 4]
        ranked = []
        for d in self.docs():
            scoped = any(norm_name(x) in keys or x == curie for x in d.src.diseases) and len(d.src.diseases) == 1
            hits = sum(d.low.count(f" {k} ") for k in keys)
            if scoped or hits:
                ranked.append((not scoped, d.src.tier or 3, -hits, d.src.id, d, scoped))
        ranked.sort(key=lambda x: x[:4])
        return [(r[4], r[5]) for r in ranked[:max_docs]]


class LocalFallback:
    """compile_one(disease_id, ontology_id) -> on_demand card from the downloaded sources, or None."""

    def __init__(self, settings: Settings, llm: Optional[LLMLike], store: CardStore,
                 index: Optional[LocalDocumentIndex] = None):
        self.s = settings
        self.store = store
        self.index = index or LocalDocumentIndex(settings.sources_dir, settings.kb_dir / ".passage_cache")
        self.extractor = ClaimExtractor(llm, settings.kb_dir / ".extract_cache")
        self.tried: set[str] = set()  # diseases already searched in this run (found or not)

    def compile_one(self, disease_id: str, curie: str) -> Optional[DiseaseCard]:
        index = self.store.ontology.disease
        if index is None or not curie or curie not in index or disease_id in self.tried:
            return None
        self.tried.add(disease_id)
        names = list(iter_names(index, curie, 12))
        docs = self.index.candidates(names, curie, self.s.local_fallback_max_docs)
        if not docs:
            return None
        test_lines = [f"{t.id}: {', '.join(t.outcomes)}" + (f" (also: {', '.join(t.aliases)})" if t.aliases else "")
                      for t in self.store.tests()]
        claims = []
        for doc, scoped in docs:
            sel = select_passages(doc.passages, names, require_name=not scoped)
            ex, _ = self.extractor.extract(doc.src, doc.sha, curie, names[0], names[1:], sel, test_lines)
            if ex is None:
                continue
            got, _ = verify_extraction(ex, doc.src, curie, names, doc.norm, self.store, self.s.target_population,
                                       require_disease_mention=not scoped)
            claims.extend(got)
        if not claims:
            return None
        mapper = VocabularyMapper(self.store, self.store.ontology.phenotype, self.s.dedup_auto, self.s.hpo_min_term_depth)
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        card, _ = build_card(curie, index, claims, mapper, self.store, now, status="on_demand")
        if card is None:
            return None
        self.store.add_registry(mapper.new_defs.values())
        return self.store.add(card)
