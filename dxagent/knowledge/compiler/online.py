"""Live fallback (v0.4, OFF by default: DX_LIVE_FALLBACK=1 to enable).

Used only for a disease that resolves in the ontology but has no card of its own and no carded ancestor. Online
sources are restricted to PubMed records whose publication type is a guideline, systematic review or meta-analysis,
and only their abstracts are read, so every source is tier 3 ("abstract"). The abstracts go through the same
extraction, verification and assembly as an offline compile. The card is marked provisional: confidence is capped
(`live_fallback_conf_cap`, default 0.5), it is never admitted through the reverse index, and it is cached under
cache/provisional/ until an offline compile replaces it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Callable, Optional

from dxagent.config import Settings
from dxagent.knowledge.cards import CardStore, DiseaseCard
from dxagent.knowledge.compiler.assemble import VocabularyMapper, build_card
from dxagent.knowledge.compiler.documents import Passage
from dxagent.knowledge.compiler.extract import ClaimExtractor, verify_extraction
from dxagent.knowledge.compiler.sources import SourceDoc
from dxagent.knowledge.compiler.verify import norm_text
from dxagent.knowledge.ontology import iter_names
from dxagent.llm import LLMLike

log = logging.getLogger(__name__)
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PUB_TYPES = '(systematic review[pt] OR meta-analysis[pt] OR practice guideline[pt] OR guideline[pt])'


def _http_get(url: str, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "dxagent/0.4"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def pubmed_records(names: list[str], retmax: int = 5, http: Callable[[str], bytes] = _http_get) -> list[dict[str, str]]:
    """[{pmid, title, abstract, year, pubtypes}] for guideline / systematic-review / meta-analysis records."""
    names = [n for n in names if len(n) >= 4][:4]
    if not names:
        return []
    term = "(" + " OR ".join(f'"{n}"[tiab]' for n in names) + ") AND " + PUB_TYPES
    try:
        q = urllib.parse.urlencode({"db": "pubmed", "retmode": "json", "retmax": retmax, "sort": "relevance", "term": term})
        ids = json.loads(http(f"{EUTILS}esearch.fcgi?{q}"))["esearchresult"]["idlist"]
        if not ids:
            return []
        q = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
        root = ET.fromstring(http(f"{EUTILS}efetch.fcgi?{q}"))
    except Exception as exc:
        log.warning("PubMed fallback failed for %s: %s", names[0], exc)
        return []
    out = []
    for art in root.iter("PubmedArticle"):
        abstract = " ".join("".join(t.itertext()) for t in art.iter("AbstractText"))
        if not abstract:
            continue
        out.append({"pmid": art.findtext(".//PMID") or "", "title": art.findtext(".//ArticleTitle") or "", "abstract": abstract,
                    "year": art.findtext(".//PubDate/Year") or "",
                    "pubtypes": ", ".join(pt.text or "" for pt in art.iter("PublicationType"))})
    return out


class LiveFallback:
    def __init__(self, settings: Settings, llm: LLMLike, store: CardStore, http: Callable[[str], bytes] = _http_get):
        self.s = settings
        self.store = store
        self.http = http
        self.extractor = ClaimExtractor(llm, settings.cache_dir / "extract")

    def compile_one(self, disease_id: str, curie: str) -> Optional[DiseaseCard]:
        index = self.store.ontology.disease
        if index is None or not curie or curie not in index:
            return None
        names = list(iter_names(index, curie, 8))
        records = pubmed_records(names, http=self.http)
        if not records:
            return None
        test_lines = [f"{t.id}: {', '.join(t.outcomes)}" for t in self.store.tests()]
        claims = []
        for r in records:
            text = f"{r['title']} {r['abstract']}"
            src = SourceDoc(id=f"pmid_{r['pmid']}", kind="abstract", title=r["title"], edition=r["year"],
                            licence="PubMed abstract", access="open", url=f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/")
            sha = hashlib.sha256(text.encode()).hexdigest()
            ex, _ = self.extractor.extract(src, sha, curie, names[0], names[1:], [Passage("abstract", text)], test_lines)
            if ex is None:
                continue
            got, _ = verify_extraction(ex, src, curie, names, norm_text(text), self.store, self.s.target_population,
                                       require_disease_mention=False)
            claims.extend(got)
        if not claims:
            return None
        mapper = VocabularyMapper(self.store, self.store.ontology.phenotype, self.s.dedup_auto, self.s.hpo_min_term_depth)
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        card, _ = build_card(curie, index, claims, mapper, self.store, now, status="provisional",
                             conf_cap=self.s.live_fallback_conf_cap)
        if card is None:
            return None
        self.store.add_registry(mapper.new_defs.values())
        return self.store.add(card)
