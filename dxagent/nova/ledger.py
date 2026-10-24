"""Encounter ledger: every action, every reply, and the turn it was received in.

The S and O sections of the SOAP note are written ONLY from these entries ("what was not asked is not credited"),
and every O item carries its turn number, as the contest requires.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Entry:
    id: str
    turn: Optional[int]  # contest turn at which the reply was received (0: opening and vitals); None if refused
    kind: str  # opening | vitals | SAY | EXAM | TEST | closing
    section: str  # cc | vitals | hpi | pmh | meds | allergy | fhx | shx | ob | ros | exam | test | education
    label: str  # Korean label (slot label or what the question was meant to learn)
    text: str  # what the doctor said or requested
    reply: str  # what came back
    accepted: bool = True
    codes: list[str] = field(default_factory=list)
    source: str = ""  # slot | core | bank | critic | closing | opening
    item: str = ""  # chart item name the entry fills in the note ("Allergy", "Fever/Chill", "Chest")
    note: str = ""  # chart line written for the note


class Ledger:
    def __init__(self) -> None:
        self.entries: list[Entry] = []

    def add(self, **kw: Any) -> Entry:
        e = Entry(id=f"e{len(self.entries)}", **kw)
        self.entries.append(e)
        return e

    def accepted(self) -> list[Entry]:
        return [e for e in self.entries if e.accepted]

    def section(self, name: str) -> list[Entry]:
        return [e for e in self.entries if e.accepted and e.section == name]

    def asked_texts(self) -> set[str]:
        return {e.text for e in self.entries if e.kind in ("SAY", "EXAM", "TEST")}

    def transcript(self, limit_chars: int = 9000) -> str:
        """T<turn> [kind] doctor -> reply, oldest first (trimmed from the start if too long)."""
        lines = []
        for e in self.entries:
            if not e.accepted:
                continue
            if e.kind == "opening":
                lines.append(f"T0 [opening] patient: {e.reply}")
            elif e.kind == "vitals":
                lines.append(f"T0 [vitals] {e.reply}")
            else:
                lines.append(f"T{e.turn} [{e.kind}] doctor: {e.text} -> patient/record: {e.reply}")
        out = "\n".join(lines)
        return out if len(out) <= limit_chars else "...\n" + out[-limit_chars:]

    def to_list(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self.entries]
