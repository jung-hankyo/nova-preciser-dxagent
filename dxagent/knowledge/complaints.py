"""Complaint registry (v0.3.1, P1): LLM-normalized free-text complaints that consolidate themselves.

The canonical complaint is the key under which priors and card prevalence are stored, so near-synonyms
("chest pain", "chest discomfort", "chest tightness") must resolve to ONE entry or they never pool. The scheme
mirrors A3 finding codes:

    alias table (exact, normalized)          -> that entry
    concept Dice >= dedup_auto (0.85)         -> merge automatically, record the alias
    dedup_confirm (0.60) <= Dice < dedup_auto -> one LLM confirmation against the candidates
    otherwise                                 -> a new entry

Each canonical complaint carries a typical adult triage acuity (1 most urgent .. 5 least, KTAS/CTAS scale),
assigned once by the LLM the first time the complaint is seen and cached. It is used only as a tiebreak when
several complaints are equally the stated reason for the visit, and it is stable across cases because it is
cached. Real triage acuity depends on modifiers (vital signs, pain severity); a per-complaint value is a
deliberate simplification.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from dxagent.knowledge.cards import norm_key, slug
from dxagent.knowledge.lexicon import concepts

log = logging.getLogger(__name__)

COMPLAINT_MATCH_SYSTEM = """You maintain the list of presenting complaints used by a diagnostic agent.

Given a new complaint label and a short list of existing complaints:
- same_as: the key of the existing complaint that means the same presenting problem, or empty if none does.
  Treat lay and clinical wording as the same ("shortness of breath" = "dyspnea"); treat different body sites or
  different symptoms as different ("chest pain" != "abdominal pain", "weakness" != "fatigue").
- acuity: the typical triage acuity of an unselected adult presenting with the NEW complaint, on the KTAS/CTAS
  scale: 1 resuscitation, 2 emergent, 3 urgent, 4 less urgent, 5 non-urgent."""


class Complaint(BaseModel):
    key: str
    label: str
    aliases: list[str] = Field(default_factory=list)
    acuity: int = 3
    source: str = "seed"


def similarity(a: str, b: str) -> float:
    """Concept Dice between two complaint phrases (lexicon synonyms and stemming, no embeddings)."""
    ca, cb = set(concepts(a)), set(concepts(b))
    if not ca or not cb:
        return 0.0
    return 2 * len(ca & cb) / (len(ca) + len(cb))


class ComplaintRegistry:
    FILE = "_complaints.yaml"

    def __init__(self, seed_dir: Path, drafted_dir: Optional[Path] = None, llm=None,
                 auto: float = 0.85, confirm: float = 0.60):
        self.seed_dir = Path(seed_dir)
        self.drafted_dir = Path(drafted_dir) if drafted_dir else None
        self.llm = llm
        self.auto, self.confirm = auto, confirm
        self._lock = threading.RLock()
        self._items: dict[str, Complaint] = {}
        self._alias: dict[str, str] = {}
        for d in (self.seed_dir, self.drafted_dir):
            if d and (d / self.FILE).exists():
                for row in yaml.safe_load((d / self.FILE).read_text(encoding="utf-8")) or []:
                    c = Complaint.model_validate(row)
                    have = self._items.get(c.key)
                    if have is not None:
                        have.aliases = list(dict.fromkeys(have.aliases + c.aliases))
                    else:
                        self._items[c.key] = c
        for c in self._items.values():
            self._index(c)

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: str) -> Optional[Complaint]:
        return self._items.get(key)

    def _index(self, c: Complaint) -> None:
        for name in [c.key, c.label, *c.aliases]:
            k = norm_key(name)
            if k:
                self._alias.setdefault(k, c.key)

    def _persist(self) -> None:
        if not self.drafted_dir:
            return
        self.drafted_dir.mkdir(parents=True, exist_ok=True)
        rows = [c.model_dump() for c in sorted(self._items.values(), key=lambda c: c.key)]
        (self.drafted_dir / self.FILE).write_text(yaml.safe_dump(rows, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def _best(self, label: str) -> list[tuple[float, Complaint]]:
        scored = []
        for c in self._items.values():
            s = max(similarity(label, n) for n in [c.label, *c.aliases])
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: (-x[0], x[1].key))
        return scored

    def resolve(self, label: str) -> tuple[Optional[Complaint], str]:
        """Return (canonical complaint, how) with how in {alias, auto, confirmed, new, empty}."""
        label = (label or "").strip().lower()
        if not slug(label):
            return None, "empty"
        with self._lock:
            hit = self._alias.get(norm_key(label))
            if hit:
                return self._items[hit], "alias"
            scored = self._best(label)
            if scored and scored[0][0] >= self.auto:
                c = scored[0][1]
                c.aliases = list(dict.fromkeys(c.aliases + [label]))
                self._index(c)
                self._persist()
                return c, "auto"
            cands = [c for s, c in scored if s >= self.confirm][:5]
            same, acuity = "", 3
            if self.llm is not None:
                try:
                    from dxagent.schemas import ComplaintMatchOutput

                    lines = "\n".join(f"- {c.key}: {c.label}" + (f" ({', '.join(c.aliases[:4])})" if c.aliases else "")
                                      for c in cands) or "- (none)"
                    out = self.llm.structured(ComplaintMatchOutput, COMPLAINT_MATCH_SYSTEM,
                                              f"New complaint: {label}\n\nExisting complaints:\n{lines}")
                    same, acuity = out.same_as.strip(), int(out.acuity)
                except Exception as exc:  # a failed call must never block perception
                    log.warning("Complaint match failed: %s", exc)
            match = next((c for c in cands if same and same in (c.key, c.label)), None)
            if match is not None:
                match.aliases = list(dict.fromkeys(match.aliases + [label]))
                self._index(match)
                self._persist()
                return match, "confirmed"
            c = Complaint(key=slug(label), label=label, acuity=min(max(acuity, 1), 5), source="llm")
            if c.key in self._items:  # slug collision with a differently worded entry: same key, same complaint
                return self._items[c.key], "alias"
            self._items[c.key] = c
            self._index(c)
            self._persist()
            return c, "new"
