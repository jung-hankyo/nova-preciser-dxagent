"""Play the competition's sample cases through its development-kit simulator, without any model.

    python experiments/nova_devkit_check.py --devkit <dev kit> --samples <sample cases> \
        [--agent run.py] [--out runs/devkit_check] [--audit]

The development kit (not included here) holds nova_local.py, a simulator with the server's rules:
action body fields, 30 NFC characters, turn limit, EXAM_NOT_SPECIFIC and EXAM_NOT_RECOGNIZED (both cost a turn),
record fields and size, Doctor call limits. Normally it needs OpenAI-compatible models for the patient, the examiner
and the Doctor. Here they are replaced by scripted stand-ins:
- the patient answers with the script sentence that best matches the question (character bigrams);
- the examiner returns the exam-table row whose name best matches the request; for a maneuver the table lacks, the
  kit asks the examiner to write a finding, and the stand-in writes "<maneuver>: 이상 없음 (표 밖 진찰)";
- the Doctor model answers "OK".
Run the agent with NOVA_OFFLINE=1, so the only Doctor call is the one each preliminary case requires.

What it checks: the agent completes every sample through the real SDK client class, every DIAGNOSE is accepted,
exam requests map to table rows (bigram proxy for the examiner model: "unlisted" means the wording is far from the
table's names; the server would write a finding for it), and, with --audit, that nothing is written outside the results directory and no network connection
is opened (the server's file system is read-only and offline). It says nothing about medical quality. For that, use
the kit's local-model runner, or a server sample run with the development key.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path


def grams(text: str) -> set[str]:
    t = re.sub(r"[\s()\[\]·,./?!~:]+", "", (text or "").lower())
    return {t[i:i + 2] for i in range(len(t) - 1)}


QUESTION_ENDINGS = grams("하세요있나요있으세요어떠세요하나요되나요입니까")
EXAM_ENDINGS = grams("하겠습니다확인보겠습니다")


class ScriptedModels:
    """Patient, examiner and Doctor stand-ins with nova_local.Endpoint's interface."""

    base_url, model = "local://scripted", "scripted"

    def complete(self, messages, max_tokens=2048, json_mode=False, **options):
        system, user = messages[0]["content"], messages[-1]["content"]
        if system.startswith("당신은 OSCE") and "진찰표에 없습니다" in system:  # a finding for an unlisted maneuver
            req = json.loads(user)
            result = f"{(req.get('maneuver') or req.get('request') or '').strip()}: 이상 없음 (표 밖 진찰)"
            return json.dumps({"result": result}, ensure_ascii=False), "stop", 50, 10
        if system.startswith("당신은 진료실에 온 환자"):
            facts = json.loads(system[system.index("\n\n{") + 2:])
            utterance = json.loads(user)["의사"]
            sentences = [s.strip() for s in re.split(r"(?<=[.!?다])\s+", facts["대본"]) if len(s.strip()) > 4]
            want = grams(utterance) - QUESTION_ENDINGS
            best, score = None, 0.0
            for s in sentences:
                sc = len(want & grams(s)) / (len(want) or 1)
                if sc > score:
                    best, score = s, sc
            if "?" not in utterance:
                answer = "네, 알겠습니다."
            elif best is None or score < 0.25:
                answer = "아니요, 그런 건 없어요."
            else:
                answer = best[:60]
            return json.dumps({"content": answer, "effective_content": utterance}, ensure_ascii=False), "stop", 50, 20
        if system.startswith("당신은 OSCE"):
            table = system.split("진찰표 (id | 진찰):\n", 1)[1]
            rows = [line.split(" | ", 1) for line in table.splitlines() if " | " in line]
            request = json.loads(user)["request"]
            if re.search(r"(다 해|전부|모두|이상 소견만)", request):
                return json.dumps({"id": None, "reason": "too_broad"}), "stop", 50, 10
            want = grams(request) - EXAM_ENDINGS
            best, score = None, 0.0
            for rid, name in rows:
                g = grams(name.split("(")[0])
                sc = len(want & g) / (len(want | g) or 1)
                if sc > score:
                    best, score = rid, sc
            out = {"id": best} if best and score >= 0.18 else {"id": None, "reason": "unlisted", "maneuver": request}
            return json.dumps(out, ensure_ascii=False), "stop", 50, 10
        return "OK", "stop", 10, 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devkit", required=True, help="unpacked nova-devkit folder (nova_local.py, nova_sdk.py)")
    ap.add_argument("--samples", required=True, help="folder with <case>/case.json (samples ZIP or devkit/samples)")
    ap.add_argument("--agent", default="run.py")
    ap.add_argument("--out", default="runs/devkit_check")
    ap.add_argument("--audit", action="store_true", help="record file writes and network connections")
    a = ap.parse_args()
    out_dir = Path(a.out).resolve()
    events: list[tuple[str, str]] = []
    if a.audit:
        def hook(event, args):
            if event == "open" and args and isinstance(args[0], str):
                mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ""
                flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
                if ((any(c in mode for c in "wax+") or flags & 0o3) and not args[0].startswith(str(out_dir))
                        and "__pycache__" not in args[0]):  # bytecode caches: the interpreter's, skipped on read-only
                    events.append(("write", args[0]))
            elif event == "socket.connect":
                events.append(("connect", str(args[1])[:80]))
        sys.addaudithook(hook)

    sys.path.insert(0, str(Path(a.devkit).resolve()))
    import nova_local  # type: ignore  # the development kit's simulator

    agent_path = Path(a.agent).resolve()
    sys.path.insert(0, str(agent_path.parent))
    spec = importlib.util.spec_from_file_location("nova_agent", agent_path)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)

    summary = []
    for folder in sorted(p.parent for p in Path(a.samples).glob("*/case.json")):
        case = nova_local.load_case(folder)
        sim = nova_local.Simulator(case, doctor=ScriptedModels(), sim=ScriptedModels(), minutes=30)
        client = nova_local.LocalClient(sim)
        t0 = time.time()
        state = agent.prepare()
        client.ready()
        try:
            result = agent.consult(client, state)
        except nova_local.NovaError as exc:
            result = {"error": exc.code}
        s = nova_local.save_result(sim, out_dir / case["case_id"], t0)
        exams = [e for e in sim.events if e["action"] == "EXAM"]
        summary.append({"case": case["case_id"], "status": s["status"], "turns": s["turns"],
                        "doctor_calls": s["doctor_calls"], "result": result,
                        "exam_rows": [sim.rows[e["effective_content"]]["exam"] for e in exams if e.get("effective_content")],
                        "unlisted": [e["content"] for e in exams if not e.get("effective_content")]})
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if a.audit:
        print(json.dumps({"writes_or_connections": events[:50]}, ensure_ascii=False))
    ok = all(r["status"] == "completed" and r["doctor_calls"] >= 1 for r in summary) and not events
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
