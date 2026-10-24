"""Cases for simulated visits, in one shape:

    {"case_id", "playbook", "opening", "vital_signs", "script", "exams": {row: finding}, "key": {...}}

from our YAML files (experiments/data/nova_cpx/*.yaml, the kit's case form with an answer key) or from the kit's
sample-case folders (<case>/case.json; no key). The sample cases are contest material: they are read from where the
user keeps them, never copied into the repository.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "experiments" / "data" / "nova_cpx"


def from_yaml(path: Path) -> list[dict[str, Any]]:
    out = []
    for c in yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []:
        out.append({"case_id": str(c["case_id"]), "playbook": c.get("playbook"), "opening": c["opening"],
                    "vital_signs": c.get("vital_signs", ""), "script": c.get("script", ""),
                    "exams": dict(c.get("exams") or {}), "key": dict(c.get("key") or {}), "source": str(path)})
    return out


def from_sample(folder: Path) -> dict[str, Any]:
    c = json.loads((Path(folder) / "case.json").read_text(encoding="utf-8"))
    start = c.get("start_information") or {}
    rows = {r.get("exam", ""): r.get("result", "") for r in c.get("exam_results") or [] if r.get("exam")}
    return {"case_id": str(c.get("case_id") or Path(folder).name), "playbook": None,
            "opening": start.get("opening", ""), "vital_signs": start.get("vital_signs", ""),
            "script": c.get("patient_script", ""), "exams": rows, "key": {}, "source": str(folder)}


def load(paths: Iterable[str | Path] = (), ids: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Cases from YAML files, folders of YAML files, sample folders (<case>/case.json) or a folder of them."""
    cases: list[dict[str, Any]] = []
    for p in [Path(x) for x in paths] or [DEFAULT_DIR]:
        if p.is_file() and p.suffix in (".yaml", ".yml"):
            cases += from_yaml(p)
        elif (p / "case.json").exists():
            cases.append(from_sample(p))
        elif p.is_dir():
            for f in sorted(p.glob("*.yaml")):
                cases += from_yaml(f)
            for f in sorted(p.glob("*/case.json")):
                cases.append(from_sample(f.parent))
    want = set(ids)
    return [c for c in cases if not want or c["case_id"] in want]
