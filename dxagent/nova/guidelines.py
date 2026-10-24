"""The contest's guideline library, read at run time.

The contest environment may mount a library of Markdown guideline documents read-only; NOVA_GUIDELINES_DIR points
there. Each document starts with its title, publisher, year, language, source URL and licence, and a `catalog.csv`
may list them.

Licence: competition use only, non-commercial. Keep each document's source and licence. Many documents
(CC BY-ND, BY-NC-ND, KOGL type 4) forbid distributing changed versions (translations, summaries, restructurings).
This module therefore ships nothing derived from the documents. It reads them in the contest environment and quotes
short excerpts, with title, publisher and year, into the assessment prompt of the case at hand.

    lib = GuidelineLibrary.from_config(cfg)   # prepare(): read the catalog, or scan document headers
    docs = lib.search(["pulmonary embolism", "폐색전증"], k=2)
    excerpts = lib.excerpts(docs, terms=["pulmonary embolism", "treatment", "anticoagulation"], max_chars=1200)

Retrieval is lexical and cheap: titles first (catalog or header), then paragraphs scored by query terms. Column names of
the catalog are matched loosely; without a catalog, document headers are read.
"""

from __future__ import annotations

import csv
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

_WORD = re.compile(r"[a-z][a-z0-9\-]{2,}|[가-힣]{2,}")
_STOP = {"the", "and", "for", "with", "management", "guideline", "guidelines", "clinical", "practice", "update",
         "recommendations", "recommendation", "diagnosis", "treatment", "adult", "adults", "patients", "patient",
         "acute", "chronic", "disease", "진료지침", "권고", "환자", "질환", "가이드라인"}
_PLAN_TERMS = ("diagnos", "treat", "therapy", "management", "test", "imaging", "admission", "follow", "refer",
               "first-line", "antibiotic", "dose", "진단", "치료", "검사", "입원", "추적", "의뢰", "투여", "약물")


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower()) if w not in _STOP]


@dataclass
class Doc:
    path: Path
    title: str
    org: str = ""
    year: str = ""
    lang: str = ""
    license: str = ""
    tokens: set = field(default_factory=set)


def _col(header: list[str], *names: str) -> Optional[str]:
    low = {h.lower().strip(): h for h in header}
    for n in names:
        for k, h in low.items():
            if n in k:
                return h
    return None


def _parse_header(path: Path, limit: int = 2500) -> dict[str, str]:
    """Title / publisher / year / language / licence from the top of a document (front matter, key: value lines, or
    the first heading)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            head = fh.read(limit)
    except OSError:
        return {}
    out: dict[str, str] = {}
    keys = {"title": ("title", "제목"), "org": ("publisher", "organization", "organisation", "issuer", "발행기관", "기관",
                                                "source org"),
            "year": ("year", "연도", "발행연도", "published"), "lang": ("language", "lang", "언어"),
            "license": ("license", "licence", "라이선스")}
    for line in head.splitlines()[:40]:
        if line.lstrip().startswith("#"):
            if "title" not in out and line.lstrip("# ").strip():
                out["title"] = line.lstrip("# ").strip()
            continue
        m = re.match(r"^\s*(?:[-*]\s*)?(?:\*\*)?([^:：*]{1,30}?)(?:\*\*)?\s*[:：]\s*(.+)$", line)
        if m:
            k, v = m.group(1).strip().lower(), m.group(2).strip()
            for name, aliases in keys.items():
                if name not in out and any(a in k for a in aliases):
                    out[name] = v
    if "year" in out:
        y = re.search(r"(19|20)\d{2}", out["year"])
        out["year"] = y.group(0) if y else ""
    return out


class GuidelineLibrary:
    def __init__(self, root: Optional[Path]):
        self.root = Path(root) if root else None
        self.docs: list[Doc] = []
        self.loaded = False
        self.source = ""

    @classmethod
    def from_config(cls, cfg) -> "GuidelineLibrary":
        root = cfg.guidelines_dir or os.environ.get("NOVA_GUIDELINES_DIR") or "./guidelines"
        return cls(Path(root)) if cfg.use_guidelines else cls(None)

    def available(self) -> bool:
        return bool(self.root and self.root.is_dir())

    def load(self, time_budget_s: float = 120.0) -> int:
        """Index titles (catalog.csv, else document headers). Safe to call more than once."""
        if self.loaded or not self.available():
            return len(self.docs)
        t0 = time.time()
        cat = self.root / "catalog.csv"
        if cat.exists():
            self._load_catalog(cat)
            self.source = "catalog.csv"
        if not self.docs:
            for p in sorted(self.root.rglob("*.md")):
                if time.time() - t0 > time_budget_s:
                    break
                if p.name.lower() == "readme.md":
                    continue
                h = _parse_header(p)
                title = h.get("title") or p.stem.replace("_", " ")
                self.docs.append(Doc(p, title, h.get("org", ""), h.get("year", ""), h.get("lang", ""), h.get("license", "")))
            self.source = "headers"
        for d in self.docs:
            d.tokens = set(_tokens(d.title + " " + d.path.stem.replace("_", " ")))
        self.loaded = True
        return len(self.docs)

    def _load_catalog(self, path: Path) -> None:
        try:
            with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            return
        if not rows:
            return
        header = list(rows[0].keys())
        c_path = _col(header, "path", "file", "파일", "경로", "md")
        c_title = _col(header, "title", "제목", "name")
        c_org = _col(header, "publisher", "organ", "issuer", "발행", "기관")
        c_year = _col(header, "year", "연도", "date")
        c_lang = _col(header, "lang", "언어")
        c_lic = _col(header, "licen", "라이선스")
        for r in rows:
            rel = (r.get(c_path) or "").strip() if c_path else ""
            p = (self.root / rel) if rel else None
            if p is None or not p.exists():
                continue
            y = re.search(r"(19|20)\d{2}", r.get(c_year) or "") if c_year else None
            self.docs.append(Doc(p, (r.get(c_title) or p.stem) if c_title else p.stem, (r.get(c_org) or "") if c_org else "",
                                 y.group(0) if y else "", (r.get(c_lang) or "") if c_lang else "",
                                 (r.get(c_lic) or "") if c_lic else ""))

    def get(self, paths: Iterable[str]) -> list[Doc]:
        """Documents by catalog path (a playbook's references), in the given order; missing ones are skipped."""
        self.load()
        want = [str(p).strip().lstrip("/") for p in paths if str(p).strip()]
        by_rel: dict[str, Doc] = {}
        for d in self.docs:
            try:
                by_rel[d.path.relative_to(self.root).as_posix()] = d
            except ValueError:
                by_rel[d.path.as_posix()] = d
        return [by_rel[w] for w in want if w in by_rel]

    def search(self, queries: Iterable[str], k: int = 2) -> list[Doc]:
        """Documents whose title best covers any one query (each query scored on its own, best doc per query first)."""
        self.load()
        picked: list[Doc] = []
        for q in queries:
            qt = set(_tokens(q))
            if not qt:
                continue
            best, score = None, 0.0
            for d in self.docs:
                hit = qt & d.tokens
                if not hit:
                    continue
                s = len(hit) / len(qt) - 0.01 * max(0, len(d.tokens) - len(qt)) + (0.02 * (int(d.year) - 2000) / 25 if d.year.isdigit() else 0)
                if s > score and d not in picked:
                    best, score = d, s
            if best is not None and score >= 0.5:
                picked.append(best)
            if len(picked) >= k:
                break
        return picked

    def excerpts(self, docs: list[Doc], terms: Iterable[str], max_chars: int = 2400) -> list[dict[str, str]]:
        """Best paragraphs of each document for the query terms and plan words, within a total character budget."""
        if not docs:
            return []
        qt = set(_tokens(" ".join(terms)))
        per_doc = max(300, max_chars // len(docs))
        out = []
        for d in docs:
            try:
                text = d.path.read_text(encoding="utf-8", errors="replace")[:400_000]
            except OSError:
                continue
            paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=#)", text) if len(p.strip()) > 80]
            scored = []
            for i, p in enumerate(paras[1:], 1):  # skip the header block
                low = p.lower()
                s = sum(1 for t in qt if t in low) * 2 + sum(1 for t in _PLAN_TERMS if t in low)
                if s:
                    scored.append((s, i, p))
            chosen, used = [], 0
            for s, i, p in sorted(scored, key=lambda x: (-x[0], x[1])):
                p = re.sub(r"\s+", " ", p)
                if used + len(p) > per_doc:
                    p = p[: max(0, per_doc - used)]
                if len(p) < 60:
                    break
                chosen.append((i, p))
                used += len(p)
            if chosen:
                body = " … ".join(p for _, p in sorted(chosen))
                out.append({"title": d.title, "org": d.org, "year": d.year, "text": body, "path": str(d.path)})
        return out
