"""Summarize live runs (sample.py with the development key) from the per-case JSON logs, for review.

    NOVA_LOG_DIR=runs/live/v081 python sample.py --case <ID>          # writes runs/live/v081/<case>.json
    python -B experiments/nova_live_report.py runs/live/v081           # writes <case>.md next to each log

Each <case>.md holds what the review needs and nothing else: the outcome (playbook, diagnosis, turns, refused
actions), the model calls (role, tokens, seconds, errors), the dialogue as asked and answered, the items dropped for
turns, the visit's events (focus, time-out, follow-ups, backfill) and the submitted note. The printed table compares
cases at a glance. The logs hold no key or token; they do hold the sample's dialogue, so keep them out of git (runs/
is ignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VISIT_ROLES = ("plan", "focus", "review", "assess", "notes")
EVENTS = ("select", "focus", "focus_failed", "review", "review_failed", "follow_up", "backfill", "assess",
          "syndrome_diagnosis", "refused", "blocked", "bundled", "exam_skipped", "budget", "chart_lines_dropped")


def _calls(rec: dict) -> tuple[list[str], dict]:
    rows, tot = [], {"calls": 0, "failed": 0, "in": 0, "out": 0, "s": 0.0}
    for c in rec.get("llm_trace") or []:
        tot["calls"] += 1
        tot["failed"] += int(not c.get("ok"))
        tot["in"] += int(c.get("in") or 0)
        tot["out"] += int(c.get("out") or 0)
        tot["s"] += float(c.get("s") or 0)
        rows.append(f"| {c.get('role')} | {'ok' if c.get('ok') else 'FAILED'} | {c.get('in', 0)} | {c.get('out', 0)} | "
                    f"{c.get('s', 0)} | {str(c.get('error', '')).replace('|', '/')[:120]} |")
    return rows, tot


def _dialogue(rec: dict) -> list[str]:
    out = []
    for e in rec.get("ledger") or []:
        if e.get("kind") in ("opening", "vitals"):
            out.append(f"- [{e.get('kind')}] {e.get('reply') or e.get('text')}")
            continue
        mark = "" if e.get("accepted", True) else f" REFUSED {','.join(e.get('codes') or [])}"
        out.append(f"- t{e.get('turn')} {e.get('kind')} ({e.get('section')}, {e.get('source')}): {e.get('text')}"
                   f" -> {str(e.get('reply') or '').strip()}{mark}")
    return out


def _note(rec: dict) -> list[str]:
    out, note = [], rec.get("note") or {}
    for sec in ("S", "O", "A", "P"):
        for field, text in (note.get(sec) or {}).items():
            out += [f"### {sec}.{field}", "", "```", str(text).strip() or "(empty)", "```", ""]
    return out


def report(rec: dict, name: str) -> tuple[str, dict]:
    rows, tot = _calls(rec)
    refused = [e for e in rec.get("events") or [] if e.get("kind") == "refused"]
    events = [e for e in rec.get("events") or [] if e.get("kind") in EVENTS]
    row = {"case": name, "playbook": rec.get("playbook") or "-", "turns": rec.get("turns"),
           "refused": len(refused), "calls": tot["calls"], "failed": tot["failed"], "out": tot["out"],
           "s": round(tot["s"]), "errors": len(rec.get("errors") or []), "diagnosis": rec.get("diagnosis") or ""}
    md = [f"# Live run {name}", "",
          f"- Playbook: {row['playbook']}; turns: {row['turns']}; refused actions: {row['refused']}",
          f"- Diagnosis: {row['diagnosis']}",
          f"- Model calls: {tot['calls']} ({tot['failed']} failed); tokens in {tot['in']}, out {tot['out']}; "
          f"{tot['s']:.0f} s in calls; case {rec.get('budget', {}).get('seconds')} s",
          f"- Leading (focus/time-out): {rec.get('leading')}", "",
          "## Model calls", "", "| role | ok | in | out | s | error |", "|---|---|---|---|---|---|", *rows, "",
          "## Dialogue", "", *_dialogue(rec), "",
          "## Dropped for turns", "", *(f"- {d}" for d in rec.get("dropped") or ["(none)"]), "",
          "## Events", "", *(f"- t{e.get('turn')} {json.dumps(e, ensure_ascii=False)[:600]}" for e in events), "",
          "## Errors", "", *(f"- {e.get('where')}: {e.get('error')}" for e in rec.get("errors") or []),
          *([] if rec.get("errors") else ["(none)"]), "",
          "## Note", "", *_note(rec)]
    return "\n".join(str(x) for x in md), row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="case logs (.json) or directories of them")
    args = ap.parse_args()
    files = [f for p in map(Path, args.paths) for f in (sorted(p.glob("*.json")) if p.is_dir() else [p])]
    if not files:
        print("no case logs found: was NOVA_LOG_DIR set when sample.py ran?", file=sys.stderr)
        return 1
    print("| case | playbook | turns | refused | calls | failed | out tokens | call s | errors | diagnosis |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        md, r = report(rec, f.stem[:16])
        f.with_suffix(".md").write_text(md + "\n", encoding="utf-8")
        print(f"| {r['case']} | {r['playbook']} | {r['turns']} | {r['refused']} | {r['calls']} | {r['failed']} | "
              f"{r['out']} | {r['s']} | {r['errors']} | {r['diagnosis'][:60]} |")
    print(f"\nwrote {len(files)} report(s) next to the logs (.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
