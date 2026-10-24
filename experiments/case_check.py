"""Check simulated CPX cases written in the samples' YAML format (experiments/data/nova_cpx/*.yaml).

    python experiments/case_check.py experiments/data/nova_cpx/cases_c.yaml [...]     check files
    python experiments/case_check.py --vocab                                          print the exam vocabulary
    python experiments/case_check.py --history                                        print the generic history items

A case:

    - case_id: M01
      playbook: fever            # the presentation playbook the opening must route to (null: none should match)
      opening: 이름은 ...이고, 나는 ... 살 남자이다. ...해서 병원에 왔다.
      vital_signs: "혈압 : 120/80 mmHg\\n맥박 : 80 회/분\\n호흡 : 16 회/분\\n체온 : 36.5°C"
      script: |                  # what the patient knows, one fact per sentence, plain lay Korean
        ...
      exams:                     # the examiner's table: "<한국어> (<English>)": finding
        전반적 외관 시진 (General appearance): ...
      key:
        diagnosis: <English diagnosis>
        history: [onset, pmh, ..., {case_item: "(regex|alternatives)"}]
        exams: [<rows of the table that matter>]

Checks: required fields and types; the opening states name, age and sex in a form the agent parses; the vital signs
parse; the opening routes to `playbook`; table rows are named "<ko> (<en>)" (vocabulary names are expected, others
are warned about); key exams are rows of the table; history items are generic names or compilable regexes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dxagent.nova import playbooks as P  # noqa: E402
from dxagent.nova.korean import is_pediatric, parse_demographics, parse_vitals  # noqa: E402

GENERIC_NAMES = ("onset", "character", "location", "radiation", "severity", "timing", "aggravating", "relieving",
                 "prior", "pmh", "meds", "surgery", "fh", "smoking", "alcohol", "lmp", "pregnancy", "job")
ROW = re.compile(r"^(?P<ko>[^()]+?)\s*\((?P<en>.+)\)$")  # the English may hold parentheses


def vocabulary() -> dict[str, str]:
    return {**P.SAMPLE_EXAMS, **P.EXTRA_EXAMS}


def check_case(c: dict, vocab: dict[str, str], ids: set[str]) -> tuple[list[str], list[str]]:
    errs, warns = [], []
    cid = c.get("case_id", "?")
    for k in ("case_id", "opening", "vital_signs", "script", "exams", "key"):
        if k not in c:
            errs.append(f"missing {k}")
    if errs:
        return errs, warns
    if "playbook" not in c:
        errs.append("missing playbook (the id the opening must route to, or null)")
    op = str(c["opening"])
    age, sex = parse_demographics(op)
    if age is None:
        errs.append(f"age not parsed from the opening: {op[:60]!r}")
    if sex is None:
        errs.append(f"sex not parsed from the opening: {op[:60]!r}")
    labs, line = parse_vitals(c["vital_signs"])
    if len(labs) < 3:
        errs.append(f"vital signs: only {len(labs)} values parsed from {c['vital_signs']!r}")
    want = c.get("playbook")
    if want is not None and want not in ids:
        errs.append(f"unknown playbook id {want!r}")
    pb = P.select(op, is_pediatric(op, age), sex)
    got = pb.id if pb is not None else None
    if "playbook" in c and got != want:
        errs.append(f"routes to {got!r}, expected {want!r}")
    if not isinstance(c["exams"], dict) or len(c["exams"]) < 6:
        errs.append("exams: a table of at least 6 rows is expected")
    else:
        for name, result in c["exams"].items():
            m = ROW.match(str(name).strip())
            if not m:
                errs.append(f"exam row not '<ko> (<en>)': {name!r}")
                continue
            ko = m.group("ko").strip()
            if ko not in vocab:
                warns.append(f"exam row outside the vocabulary: {name!r}")
            elif vocab[ko] != m.group("en").strip():
                warns.append(f"exam row English differs from the vocabulary ({vocab[ko]!r}): {name!r}")
            if not str(result).strip():
                errs.append(f"empty finding for {name!r}")
    key = c["key"] if isinstance(c["key"], dict) else {}
    if not str(key.get("diagnosis", "")).strip():
        errs.append("key.diagnosis missing")
    for e in key.get("exams", []) or []:
        if e not in (c["exams"] or {}):
            errs.append(f"key exam not in the table: {e!r}")
    hist = key.get("history", []) or []
    if len(hist) < 5:
        warns.append("key.history has fewer than 5 items")
    for h in hist:
        if isinstance(h, str):
            if h not in GENERIC_NAMES:
                errs.append(f"unknown generic history item {h!r}")
        elif isinstance(h, dict):
            for k, rx in h.items():
                try:
                    re.compile(str(rx))
                except re.error as exc:
                    errs.append(f"history item {k}: bad regex {rx!r} ({exc})")
        else:
            errs.append(f"history item of type {type(h).__name__}")
    script = str(c["script"])
    if len(script) < 400:
        warns.append(f"script is short ({len(script)} characters)")
    dx = str(key.get("diagnosis", ""))
    if dx and dx.lower() in script.lower():
        warns.append("the script names the diagnosis")
    return [f"{cid}: {e}" for e in errs], [f"{cid}: {w}" for w in warns]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--vocab", action="store_true")
    ap.add_argument("--history", action="store_true")
    a = ap.parse_args()
    vocab = vocabulary()
    if a.vocab:
        for ko, en in vocab.items():
            print(f"{ko} ({en})")
        return 0
    if a.history:
        print(", ".join(GENERIC_NAMES))
        return 0
    ids = set(P.load_all())
    n_err, n = 0, 0
    seen: set[str] = set()
    for f in a.files:
        cases = yaml.safe_load(Path(f).read_text(encoding="utf-8")) or []
        for c in cases:
            n += 1
            errs, warns = check_case(c, vocab, ids)
            if c.get("case_id") in seen:
                errs.append(f"{c.get('case_id')}: duplicate case_id")
            seen.add(c.get("case_id"))
            for e in errs:
                print("ERROR", e)
            for w in warns:
                print("warn ", w)
            n_err += len(errs)
    print(f"{n} cases, {n_err} errors")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
