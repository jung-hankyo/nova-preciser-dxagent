"""Run simulated visits: one case under one scenario, a matrix of them, or a record/replay visit.

    python -m sim                                   every case, the quick scenarios
    python -m sim --scenarios all --workers 4       every case, every scenario
    python -m sim --ids L01,M05 --scenarios oracle,model_slow -v
    python -m sim --agent-root /tmp/unpacked_zip    the agent as unpacked from the submission ZIP
    python -m sim replay --case L05 --dir runs/replay/L05     a visit whose model calls are answered by hand

Each visit runs the agent's real entry point (run.py: prepare(), then consult(client, state), as nova_sdk.run_agent
does) against sim/server.py through an SDK-faithful client, under a virtual clock. Results: <out>/results.jsonl (one
row per visit), <out>/summary.md; the exit code is 1 when any invariant failed.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import logging
import os
import re
import sys
import time
import traceback
import zlib
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]

from sim import cases as C  # noqa: E402
from sim.check import evaluate  # noqa: E402
from sim.check import key_items as C_key_items  # noqa: E402
from sim.client import MirrorClient, real_sdk_available, real_sdk_client  # noqa: E402
from sim.doctor import PendingCall, ReplayDoctor  # noqa: E402
from sim.scenarios import QUICK, SCENARIOS, Scenario  # noqa: E402
from sim.server import FakeServer  # noqa: E402
from sim.vclock import VirtualClock, virtual_time  # noqa: E402

_AGENTS: dict[str, Any] = {}
BASE_ENV = {"NOVA_GUIDELINES_DIR": str(ROOT / ".no-guidelines"), "NOVA_LOG_DIR": "", "NOVA_OFFLINE": "0",
            "NOVA_TRANSPORT": "auto", "NOVA_ROUND": "prelim"}


def load_agent(root: Path = ROOT) -> Any:
    """run.py of the given tree, imported once per process (its dxagent package first on sys.path)."""
    root = Path(root).resolve()
    key = str(root)
    if key not in _AGENTS:
        if key not in sys.path:
            sys.path.insert(0, key)
        spec = importlib.util.spec_from_file_location(f"nova_run_{zlib.crc32(key.encode())}", root / "run.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _AGENTS[key] = mod
    return _AGENTS[key]


@contextlib.contextmanager
def _env(values: dict[str, str]):
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v == "":
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def seed_of(case_id: str, scenario: str) -> int:
    return zlib.crc32(f"{case_id}|{scenario}".encode("utf-8"))


def run_case(case: dict, scenario: Scenario, agent: Any = None, replay_dir: Optional[Path] = None,
             keep_log: bool = False) -> dict[str, Any]:
    agent = agent or load_agent()
    clock = VirtualClock()
    doctor = ReplayDoctor(replay_dir) if replay_dir is not None else scenario.doctor(case)
    server = FakeServer(case, clock, doctor, faults=scenario.make_faults(seed_of(case["case_id"], scenario.name)),
                        patient_behaviour=scenario.patient)
    if scenario.real_sdk and not real_sdk_available():
        return {"case": case["case_id"], "scenario": scenario.name, "ok": True, "skipped": "nova_sdk.py not importable"}
    client = real_sdk_client(server) if scenario.real_sdk else MirrorClient(server)
    outcome: dict[str, Any] = {"exception": None, "result": None}
    pending: Optional[PendingCall] = None
    t0 = time.perf_counter()
    with _env({**BASE_ENV, **scenario.env}), virtual_time(clock):
        try:
            state = agent.prepare()
            client.ready()
            outcome["result"] = agent.consult(client, state)
        except PendingCall as exc:
            pending = exc
        except Exception as exc:  # consult() must never raise
            outcome["exception"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"
    real_s = round(time.perf_counter() - t0, 2)
    rec = getattr(agent, "_CASE", {}).get("last_record") or {}
    if pending is not None:
        return {"case": case["case_id"], "scenario": scenario.name, "pending": str(pending.path), "index": pending.index,
                "role": pending.role, "turn": server.turns, "real_s": real_s}
    res = evaluate(case, server, outcome, scenario.expect_submission)
    _obtained(case, res, rec)
    res.update(scenario=scenario.name, real_s=real_s, playbook=rec.get("playbook", ""),
               errors=[e.get("where") for e in rec.get("errors", [])],
               events=sorted({e["kind"] for e in rec.get("events", [])}),
               lost=len(getattr(getattr(client, "net", None), "lost", [])))
    if keep_log:
        res["server_log"] = server.log
        res["submission"] = server.submission
        res["record"] = {k: rec.get(k) for k in ("events", "errors", "llm_trace", "final", "dropped", "closing_said")}
    return res


def _obtained(case: dict, res: dict, rec: dict) -> None:
    """metrics.obtained: the key history items asked, or told by the patient before being asked (the agent recorded
    the answer and did not ask again: ledger entries with source "volunteered")."""
    m = res.get("metrics")
    items = C_key_items(case.get("key") or {})
    if not m or not items:
        return
    told = [e.get("ask") or e.get("text") or "" for e in rec.get("ledger") or [] if e.get("source") == "volunteered"]
    got = {i for i in m.get("missed", []) if any(re.search(items[i], t) for t in told)}
    m["told"] = sorted(got)
    m["obtained"] = round((len(items) - len(m.get("missed", [])) + len(got)) / len(items), 3)


# -- the matrix ----------------------------------------------------------------------------------------------------

_W: dict[str, Any] = {}


def _init(root: str, verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.CRITICAL, stream=sys.stderr)
    _W["agent"] = load_agent(Path(root))


def _one(job: tuple[dict, str]) -> dict[str, Any]:
    case, name = job
    try:
        return run_case(case, SCENARIOS[name], _W["agent"])
    except BaseException as exc:  # a harness failure is a result too
        return {"case": case["case_id"], "scenario": name, "ok": False, "failed": ["harness"],
                "exception": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"}


def matrix(cases: list[dict], names: list[str], workers: int = 1, root: Path = ROOT,
           verbose: bool = False) -> list[dict[str, Any]]:
    jobs = [(c, n) for n in names for c in cases]
    if workers <= 1:
        _init(str(root), verbose)
        return [_one(j) for j in jobs]
    import multiprocessing as mp

    ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context()
    with ctx.Pool(workers, initializer=_init, initargs=(str(root), verbose)) as pool:
        return list(pool.imap(_one, jobs, chunksize=2))


def summarize(rows: list[dict[str, Any]]) -> str:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["scenario"], []).append(r)
    out = ["| scenario | visits | ok | failures | turns | refused | model calls | out tokens | minutes | history | "
           "obtained | key exams | dx match |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
        return round(sum(xs) / len(xs), 2) if xs else "-"

    for name, rs in by.items():
        live = [r for r in rs if not r.get("skipped")]
        m = [r.get("metrics", {}) for r in live]
        fails: dict[str, int] = {}
        for r in live:
            for f in r.get("failed", []):
                fails[f] = fails.get(f, 0) + 1
        dxm = [x.get("dx_match") for x in m if x.get("dx_match") is not None]
        out.append(f"| {name} | {len(live)}{' (+%d skipped)' % (len(rs) - len(live)) if len(rs) > len(live) else ''} | "
                   f"{sum(1 for r in live if r.get('ok'))} | "
                   f"{', '.join(f'{k} {v}' for k, v in sorted(fails.items())) or '-'} | {mean([x.get('turns') for x in m])} | "
                   f"{mean([x.get('turn_refusals') for x in m])} | {mean([x.get('chat_calls') for x in m])} | "
                   f"{mean([x.get('out_tokens') for x in m])} | "
                   f"{mean([(x.get('seconds') or 0) / 60 for x in m if x.get('seconds')])} | "
                   f"{mean([x.get('history') for x in m])} | {mean([x.get('obtained') for x in m])} | "
                   f"{mean([x.get('key_exams') for x in m])} | "
                   f"{round(sum(dxm) / len(dxm), 2) if dxm else '-'} |")
    bad = [r for r in rows if not r.get("ok") and not r.get("skipped")]
    if bad:
        out += ["", "Failures:"]
        for r in bad[:60]:
            exc = (r.get("exception") or "").splitlines()[:1]
            out.append(f"- {r['scenario']} / {r['case']}: {', '.join(r.get('failed', []))}"
                       + (f" ({exc[0][:160]})" if exc else ""))
    return "\n".join(out)


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "replay":
        return replay_main(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m sim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=[], help="YAML files, case folders or the samples folder")
    ap.add_argument("--ids", default="", help="comma-separated case ids")
    ap.add_argument("--scenarios", default="quick", help="quick | all | comma-separated names")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--agent-root", default=str(ROOT), help="the tree holding run.py (e.g. an unpacked ZIP)")
    ap.add_argument("--out", default="", help="write results.jsonl and summary.md here")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    cases = C.load(a.cases, [x for x in a.ids.split(",") if x])
    names = (list(SCENARIOS) if a.scenarios == "all" else list(QUICK) if a.scenarios == "quick"
             else [x for x in a.scenarios.split(",") if x])
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown or not cases:
        print(f"unknown scenarios {unknown}" if unknown else "no cases", file=sys.stderr)
        return 2
    t0 = time.time()
    rows = matrix(cases, names, a.workers, Path(a.agent_root), a.verbose)
    text = summarize(rows)
    print(text)
    print(f"\n{len(rows)} visits ({len(cases)} cases x {len(names)} scenarios) in {time.time() - t0:.0f} s")
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "results.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        (out / "summary.md").write_text(text + "\n", encoding="utf-8")
    return 0 if all(r.get("ok") or r.get("skipped") for r in rows) else 1


# -- record / replay ---------------------------------------------------------------------------------------------


def replay_main(argv: list[str]) -> int:
    """One visit whose model calls are answered from files. Exit 3 with the request's path when a call has no answer
    yet (write the answer to the .response.txt it names, then run again); exit 0 when the visit is complete, with the
    record in <dir>/record.json and a readable <dir>/visit.md."""
    ap = argparse.ArgumentParser(prog="python -m sim replay")
    ap.add_argument("--case", required=True)
    ap.add_argument("--cases", nargs="*", default=[])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--scenario", default="oracle", help="the server and patient conditions (the model is the files)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.CRITICAL)
    found = C.load(a.cases, [a.case])
    if not found:
        print(f"no case {a.case}", file=sys.stderr)
        return 2
    folder = Path(a.dir)
    res = run_case(found[0], SCENARIOS[a.scenario], replay_dir=folder, keep_log=True)
    if res.get("pending"):
        print(f"PENDING call {res['index']} ({res['role']}) at turn {res['turn']}: answer "
              f"{folder / f'call_{res[chr(105) + chr(110) + chr(100) + chr(101) + chr(120)]}.request.md'} "
              f"into {res['pending']}")
        return 3
    # the record (with the answer key) goes beside the folder, not into it: whoever answers the calls never sees it
    out = folder.parent / f"{folder.name}.result"
    out.mkdir(parents=True, exist_ok=True)
    (out / "record.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out / "visit.md").write_text(visit_markdown(found[0], res), encoding="utf-8")
    print(f"DONE {a.case}: ok={res['ok']} failed={res['failed']} turns={res['metrics']['turns']} "
          f"diagnosis={res['metrics']['diagnosis']!r}; record in {out}")
    return 0


def visit_markdown(case: dict, res: dict) -> str:
    lines = [f"# {case['case_id']}: {case.get('opening', '')}", "", f"Vital signs: {case.get('vital_signs', '')}", "",
             f"Key diagnosis: {(case.get('key') or {}).get('diagnosis', '')}", "", "## Dialogue", ""]
    for e in res.get("server_log", []):
        if e["kind"] in ("SAY", "EXAM", "TEST") and not e.get("duplicate"):
            reply = e.get("reply") or (f"REFUSED {e['code']}" if e.get("code") else "")
            lines.append(f"- T{e.get('turn')} {e['kind']}: {e.get('content')} -> {reply}")
    sub = res.get("submission") or {}
    lines += ["", f"## Diagnosis: {sub.get('diagnosis', '')}", ""]
    for s, fields in (sub.get("note") or {}).items():
        for k, v in fields.items():
            lines += [f"### {s}.{k}", "", "```", v or "(empty)", "```", ""]
    lines += ["## Invariants", "", json.dumps(res.get("invariants"), ensure_ascii=False), "",
              "## Metrics", "", json.dumps(res.get("metrics"), ensure_ascii=False, default=str)]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
