"""Local documents -> located passages (v0.4). Deterministic, offline.

Supported: PDF (pdfplumber, page text plus tables as tab-separated rows; install the `pdf` extra), HTML (standard
library parser, scripts and styles dropped), plain text and Markdown. Each passage carries a locator ("p.12",
"p.12 table 1", "sec.7") that the extractor must cite and the card keeps.

`select_passages` chooses which passages to send to the extractor for one disease: those that name the disease
(label or ontology synonyms) and carry frequency or accuracy language, best first, within a character budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from dxagent.knowledge.ontology import norm_name


@dataclass
class Passage:
    locator: str
    text: str


class _HTMLText(HTMLParser):
    BLOCK = {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "table", "br", "td", "th"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer"):
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n\n" if tag not in ("td", "th") else "\t")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _paragraphs(text: str, prefix: str, max_chars: int = 1800) -> list[Passage]:
    out, buf, n = [], [], 0
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"[ \t]+", " ", para).strip()
        if not para:
            continue
        if buf and sum(len(b) for b in buf) + len(para) > max_chars:
            n += 1
            out.append(Passage(f"{prefix}{n}", "\n".join(buf)))
            buf = []
        buf.append(para)
    if buf:
        n += 1
        out.append(Passage(f"{prefix}{n}", "\n".join(buf)))
    return out


def load_passages(path: Path) -> list[Passage]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pdfplumber
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise RuntimeError("PDF sources need pdfplumber: pip install -e '.[pdf]'") from exc
        out: list[Passage] = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                for j, chunk in enumerate(_paragraphs(text, "", 2400), start=1):
                    out.append(Passage(f"p.{i}" + (f"#{j}" if j > 1 else ""), chunk.text))
                for t, table in enumerate(page.extract_tables() or [], start=1):
                    rows = ["\t".join((c or "").replace("\n", " ") for c in row) for row in table if row]
                    if rows:
                        out.append(Passage(f"p.{i} table {t}", "\n".join(rows)))
        return out
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in (".html", ".htm"):
        p = _HTMLText()
        p.feed(raw)
        raw = "".join(p.parts)
    return _paragraphs(raw, "sec.")


CUES = re.compile(r"(\d+(\.\d+)?\s?%|\bsensitivit|\bspecificit|likelihood ratio|\bLR\b|\bn\s?=|\bof \d+ patients|\bpresent in\b|"
                  r"\busually\b|\bcommon(ly)?\b|\brare(ly)?\b|\bfrequen|\bdiagnos|\bcriteria\b|\bdifferential\b|\buntreated\b|"
                  r"\bmortality\b|\btreat(ment|ed)\b)", re.I)


def select_passages(passages: list[Passage], names: list[str], budget_chars: int = 24000, max_passages: int = 16,
                    require_name: bool = True) -> list[Passage]:
    """Passages scored by disease-name mentions and frequency/accuracy cues; kept in document order.

    `require_name=False` for a source devoted to this one disease (a disease-specific guideline rarely repeats the
    disease name in every paragraph); passages are then chosen by cues alone."""
    keys = [k for k in {norm_name(n) for n in names} if len(k) >= 3]
    scored = []
    for i, p in enumerate(passages):
        low = " " + norm_name(p.text) + " "
        hits = sum(low.count(f" {k} ") for k in keys)
        if hits == 0 and require_name:
            continue
        cues = len(CUES.findall(p.text))
        scored.append((hits * 2 + cues, i, p))
    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen, total = [], 0
    for _, i, p in scored:
        if len(chosen) >= max_passages or total + len(p.text) > budget_chars:
            continue
        chosen.append((i, p))
        total += len(p.text)
    return [p for _, p in sorted(chosen, key=lambda x: x[0])]


def full_text(passages: list[Passage]) -> str:
    return "\n".join(p.text for p in passages)
