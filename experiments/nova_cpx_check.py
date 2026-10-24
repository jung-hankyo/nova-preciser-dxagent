"""Coverage of our own CPX-style answer keys, played through the competition's development-kit simulator without a model.

    NOVA_OFFLINE=1 python -B experiments/nova_cpx_check.py --devkit <dev kit> [--cases experiments/data/nova_cpx]
        [--agent run.py] [--out runs/cpx_check] [--label fixtures_on]

The cases (experiments/data/nova_cpx/*.yaml) are written for this project in the kit's case format. Each has a
`key`: history items (generic items below, or case-specific regexes) and the exam rows that matter. The contest's
checklist is not published, so this measures our proxy, not the score.

The patient and examiner are the scripted stand-ins of nova_devkit_check.py (bigram matching), and the agent runs
offline: no topic plan and no model hypotheses. The numbers are therefore the deterministic floor of the visit — what
the agent asks when the model contributes nothing — and they compare configurations (for example NOVA_USE_FIXTURES=0
against 1) on identical replies. They say nothing about diagnostic accuracy.

Per case: history coverage (key items asked / key items), key exam rows obtained, maneuvers the table lacks
("unlisted": answered with a finding the examiner writes, usually normal),
refused actions (a turn each), turns, the detected complaint groups and the pediatric flag.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nova_devkit_check import ScriptedModels  # noqa: E402

# Generic history items: regexes matched against the doctor's SAY lines.
GENERIC = {
    "onset": r"(언제부터|언제 시작|얼마나 됐|얼마나 되었|언제였)",
    "character": r"(어떻게 아프|어떤 느낌|양상|어떤 식으로|어떻게 불편|어떻게 아파)",
    "location": r"(어디가|어느 부위|어디를|어디에|위치|어느 쪽)",
    "radiation": r"(퍼지|번지|뻗|다른 곳)",
    "severity": r"(얼마나 심|몇 점|점수|정도가|일상생활)",
    "timing": r"(계속|자주|얼마나 오래|지속|몇 번|하루에|간헐|얼마 동안)",
    "aggravating": r"(심해지|악화|더 아프|더 심|더 불편)",
    "relieving": r"(좋아지|나아지|완화|덜 아프|편해지|덜하)",
    "prior": r"(전에도|이전에|예전에)",
    "pmh": r"(앓는 병|앓고 있|지병|진단받|고혈압|당뇨)",
    "meds": r"(드시는 약|먹는 약|복용|약이 있|약을)",
    "surgery": r"(수술|입원)",
    "fh": r"가족",
    "smoking": r"(담배|흡연)",
    "alcohol": r"(술|음주)",
    "lmp": r"(마지막 생리|마지막 월경|생리는 언제)",
    "pregnancy": r"(임신|성관계)",
    "job": r"(직업|무슨 일|일을 하|하시는 일)",
}


def materialize(src: Path, dest: Path) -> list[tuple[Path, dict]]:
    """Write each YAML case as <dest>/<case_id>/case.json in the kit's case format; return (folder, key)."""
    out = []
    for f in sorted(src.glob("*.yaml")):
        for c in yaml.safe_load(f.read_text(encoding="utf-8")) or []:
            folder = dest / c["case_id"]
            folder.mkdir(parents=True, exist_ok=True)
            case = {"case_id": c["case_id"],
                    "start_information": {"opening": c["opening"], "vital_signs": c["vital_signs"]},
                    "patient_script": c["opening"] + "\n\n" + c["script"].strip().replace("\n", "\n\n"),
                    "exam_results": [{"exam": k, "result": v} for k, v in c["exams"].items()]}
            (folder / "case.json").write_text(json.dumps(case, ensure_ascii=False, indent=1), encoding="utf-8")
            out.append((folder, c["key"]))
    return out


def key_items(key: dict) -> dict[str, str]:
    items: dict[str, str] = {}
    for h in key.get("history", []):
        if isinstance(h, str):
            items[h] = GENERIC[h]
        else:
            items.update(h)
    return items


def score(sim, key: dict) -> dict:
    says = [e["content"] for e in sim.events if e["action"] == "SAY" and e.get("content")]
    exams = [e for e in sim.events if e["action"] == "EXAM" and not e.get("refused")]
    items = key_items(key)
    asked = sorted(i for i, rx in items.items() if any(re.search(rx, s) for s in says))
    got = {sim.rows[e["effective_content"]]["exam"] for e in exams if e.get("effective_content")}
    want = set(key.get("exams", []))
    return {"history": round(len(asked) / max(1, len(items)), 3), "missed": sorted(set(items) - set(asked)),
            "key_exams": f"{len(want & got)}/{len(want)}", "key_exam_frac": len(want & got) / max(1, len(want)),
            "missed_exams": sorted(want - got), "unlisted": sum(1 for e in exams if not e.get("effective_content")),
            "refused": sum(1 for e in sim.events if e.get("refused")),
            "says": len(says), "exams": len(exams)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devkit", required=True, help="unpacked nova-devkit folder (nova_local.py, nova_sdk.py)")
    ap.add_argument("--cases", default=str(Path(__file__).resolve().parent / "data" / "nova_cpx"))
    ap.add_argument("--agent", default="run.py")
    ap.add_argument("--out", default="runs/cpx_check")
    ap.add_argument("--label", default="run")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    cases = materialize(Path(a.cases), out / "cases")

    sys.path.insert(0, str(Path(a.devkit).resolve()))
    import nova_local  # type: ignore  # the development kit's simulator

    agent_path = Path(a.agent).resolve()
    sys.path.insert(0, str(agent_path.parent))
    spec = importlib.util.spec_from_file_location("nova_agent", agent_path)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    from dxagent.nova.korean import detect_groups, is_pediatric, parse_demographics

    rows = []
    for folder, key in cases:
        case = nova_local.load_case(folder)
        sim = nova_local.Simulator(case, doctor=ScriptedModels(), sim=ScriptedModels(), minutes=30)
        client = nova_local.LocalClient(sim)
        t0 = time.time()
        state = agent.prepare()
        client.ready()
        error = None
        try:
            agent.consult(client, state)
        except nova_local.NovaError as exc:
            error = exc.code
        s = nova_local.save_result(sim, out / a.label / case["case_id"], t0)
        opening = case["start_information"]["opening"]
        age, _ = parse_demographics(opening)
        rows.append({"case": case["case_id"], "status": s["status"], "turns": s["turns"], "error": error,
                     "groups": detect_groups(opening), "pediatric": is_pediatric(opening, age),
                     "diagnosis": sim.diagnosis, **score(sim, key)})
    n = len(rows)
    summary = {"label": a.label, "cases": n,
               "completed": sum(r["status"] == "completed" for r in rows),
               "history_mean": round(sum(r["history"] for r in rows) / max(1, n), 3),
               "key_exam_mean": round(sum(r["key_exam_frac"] for r in rows) / max(1, n), 3),
               "unlisted_total": sum(r["unlisted"] for r in rows),
               "refused_total": sum(r["refused"] for r in rows),
               "turns_mean": round(sum(r["turns"] for r in rows) / max(1, n), 1),
               "no_group": [r["case"] for r in rows if not r["groups"]]}
    (out / a.label).mkdir(parents=True, exist_ok=True)
    (out / a.label / "summary.json").write_text(json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    for r in rows:
        print(f"{r['case']} {r['status'][:4]} t={r['turns']:>2} hx={r['history']:.2f} ex={r['key_exams']} ul={r['unlisted']} rf={r['refused']} "
              f"grp={','.join(r['groups']) or '-'}{' ped' if r['pediatric'] else ''} dx={r['diagnosis']}")
        print(f"    missed: {', '.join(r['missed'])} | exams: {', '.join(m.split(' (')[0] for m in r['missed_exams'])}")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["completed"] == n else 1


if __name__ == "__main__":
    sys.exit(main())
