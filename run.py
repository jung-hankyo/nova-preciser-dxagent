#!/usr/bin/env python3
"""N.O.V.A submission entry point (dxagent, simple approach).

Contract (the competition's SDK and starter package, which are not included here):

    nova_sdk.run_agent(prepare, consult)
        state = prepare()           once per case, up to 10 minutes, no case access: knowledge cards, the guideline
                                    index (NOVA_GUIDELINES_DIR), the SDK's note template
        client.ready()
        consult(client, state)      up to 30 minutes: client.session() (start information, deadline), client.say /
                                    client.exam (/ client.test in the final round), the Doctor model through
                                    client.chat() (preliminary round; at least one call per case), and finally
                                    client.diagnose(note=..., diagnosis=...), whose answer consult returns

ZIP root: run.py, requirements.txt (name==version, wheels only), dxagent/. The server runs `python run.py` with a
read-only file system; this file writes nothing and keeps nothing
between cases. The start-up block at the bottom is the starter package's: `run_agent(prepare, consult)`.

Development (not used by the harness):
    python run.py --case dxagent/nova/cases/sample_pe.json [--offline]   one local case through consult()
    python run.py --bench [CASES...] [--offline] [--judge]               mini-bench over local case files
    python run.py --stdio                                                JSON lines on stdin/stdout
Model access outside the harness: NOVA_API_BASE and NOVA_DEV_KEY (a participant's personal development key). Every other setting: dxagent/nova/config.py (NOVA_*).
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("run")

# Assets of the current case only (each case gets a fresh run state; nothing is carried across cases).
_CASE: dict[str, Any] = {}


def prepare(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Load what the visit needs. Safe to call more than once; never raises."""
    from dxagent.nova.agent import NovaAgent
    from dxagent.nova.config import NovaConfig

    _CASE.clear()
    try:
        cfg = NovaConfig.from_env()
        agent = NovaAgent(cfg, offline=cfg.offline)
        info = agent.prepare(time_budget_s=float(os.getenv("NOVA_PREPARE_BUDGET", "300")))
    except Exception as exc:  # consult() retries the setup itself
        log.error("prepare failed: %s", exc)
        return {"_dxagent_prepared": False, "error": str(exc)[:300]}
    _CASE.update(agent=agent, info=info)
    log.info("prepared: %s", info)
    return {"_dxagent_prepared": True, **info}


def consult(client: Any, state: Any = None, *args: Any, **kwargs: Any) -> None:
    """Play one case with the SDK client and submit the note with client.diagnose()."""
    from dxagent.nova.llm import HttpTransport, NovaLLM
    from dxagent.nova.sdk import SdkEnv, SdkTransport, ensure_chat_call

    if "agent" not in _CASE:
        prepare()
    agent = _CASE.get("agent")
    if agent is None:  # setup failed twice: minimal agent without guidelines
        from dxagent.nova.agent import NovaAgent
        from dxagent.nova.config import NovaConfig

        cfg = NovaConfig.from_env(use_guidelines=False)
        agent = NovaAgent(cfg, offline=cfg.offline)
        _CASE["agent"] = agent
    cfg = agent.cfg

    use_sdk = callable(getattr(client, "chat", None)) and (cfg.transport == "sdk" or (
        cfg.transport == "auto" and cfg.round == "prelim"))
    sdk_transport = SdkTransport(client) if callable(getattr(client, "chat", None)) else None
    if not cfg.offline:
        transport = sdk_transport if use_sdk else HttpTransport(cfg.api_base, cfg.api_key)
        agent.llm_factory = lambda budget: NovaLLM(cfg, transport=transport, budget=budget)

    def before_submit() -> None:
        if cfg.round == "prelim" and sdk_transport is not None:
            ensure_chat_call(sdk_transport)

    env = SdkEnv(client, state, max_turns=cfg.max_turns, before_submit=before_submit,
                 action_timeout=cfg.action_timeout, time_limit_s=cfg.time_limit_s)
    try:
        rec = agent.run(env)
    except Exception as exc:  # agent.run always reaches DIAGNOSE; this guards only start-up failures
        log.error("consult failed before the visit: %s", exc)
        return _emergency_diagnose(client, state, before_submit)
    _CASE["last_record"] = rec
    log.info("case %s: %d turns, diagnosis %r, %d errors, sdk log %d rows", rec.get("case_id"), rec.get("turns", 0),
             rec.get("diagnosis"), len(rec.get("errors", [])), len(env.log))
    return rec.get("env_result")  # the server's answer to DIAGNOSE (sample.py prints it)


def _emergency_diagnose(client: Any, state: Any, before_submit) -> Any:
    from dxagent.nova.sdk import opening_from_state
    from dxagent.nova.soap import empty_note, sdk_empty_record

    try:
        before_submit()
        note = empty_note(sdk_empty_record())
        text = opening_from_state(state, client).text
        note["S"]["cc"] = f"C.C : {text}"[:2000] if text else "C.C : 확인 필요"
        return client.diagnose(note=note, diagnosis="진단 미정")
    except Exception as exc:
        log.error("emergency DIAGNOSE failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Start-up
# ---------------------------------------------------------------------------


def _sdk_entry() -> int:
    """`python run.py` with no arguments, as the starter package does: nova_sdk.run_agent(prepare, consult)."""
    try:
        import nova_sdk  # type: ignore
    except ImportError:
        sys.stderr.write("nova_sdk.py not found next to run.py (copy it from the starter package). For local runs "
                         "see `python run.py --help`.\n")
        return 2
    if callable(getattr(nova_sdk, "run_agent", None)):
        nova_sdk.run_agent(prepare, consult)
        return 0
    for name in ("run", "main", "serve", "start_agent"):
        fn = getattr(nova_sdk, name, None)
        if not callable(fn):
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            params = {}
        if "prepare" in params and "consult" in params:
            return int(fn(prepare=prepare, consult=consult) or 0)
        if len([p for p in params.values() if p.default is p.empty]) == 2:
            return int(fn(prepare, consult) or 0)
    sys.stderr.write("nova_sdk has no runner this file recognizes: copy the start-up block of the starter package's "
                     "run.py here.\n")
    return 2


def _print_case(rec: dict, env) -> None:
    out = sys.stdout
    for a in env.actions:
        flag = "" if a["accepted"] else f"  [refused: {a.get('why', '')}]"
        out.write(f"T{a.get('turn', '-')} {a['kind']}: {a['text']}{flag}\n")
        if a["accepted"]:
            out.write(f"      -> {a.get('reply', '')}\n")
    out.write(f"\nDIAGNOSE: {rec['diagnosis']}\n\n{(rec.get('soap') or {}).get('text', '')}\n\n")
    b = rec["budget"]
    out.write(f"turns {rec['turns']} | LLM calls {b['calls']} (ok {b['calls_ok']}) | tokens in {b['input_tokens']} "
              f"out {b['output_tokens']} | {b['seconds']}s | final from {rec['final'].get('source')} | "
              f"violations {len(env.violations)} | errors {len(rec['errors'])}\n")


def _dev_main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", help="one local case file (JSON), played through consult() with a local client")
    ap.add_argument("--bench", nargs="*", help="case files or directories for the mini-bench")
    ap.add_argument("--stdio", action="store_true", help="JSON lines on stdin/stdout (development driver)")
    ap.add_argument("--offline", action="store_true", help="no model: offline stand-ins for every role")
    ap.add_argument("--round", choices=["prelim", "final"])
    ap.add_argument("--profile", choices=["standard", "lean"])
    ap.add_argument("--sim-llm", action="store_true", help="simulated patient answers with the model")
    ap.add_argument("--judge", action="store_true", help="bench: score each case with the LLM rubric judge")
    ap.add_argument("--out", default="", help="bench output directory / case record file")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.round:
        os.environ["NOVA_ROUND"] = args.round
    if args.profile:
        os.environ["NOVA_PROFILE"] = args.profile
    if args.offline:
        os.environ["NOVA_OFFLINE"] = "1"

    from dxagent.nova.config import NovaConfig

    cfg = NovaConfig.from_env()
    if args.bench is not None:
        from dxagent.nova.bench import run_bench

        summary = run_bench(args.bench or [str(ROOT / "dxagent" / "nova" / "cases")], cfg, offline=args.offline,
                            out=args.out or "runs/nova_bench", sim_llm=args.sim_llm, judge=args.judge)
        print(json.dumps(summary["mean"], ensure_ascii=False, indent=1))
        return 0

    if args.case:
        from dxagent.nova.devclient import LocalClient
        from dxagent.nova.llm import HttpTransport

        case = json.loads(Path(args.case).read_text(encoding="utf-8"))
        patient_llm, chat = None, None
        if not args.offline:
            from dxagent.nova.budget import Budget
            from dxagent.nova.llm import NovaLLM

            http = HttpTransport(cfg.api_base, cfg.api_key)
            chat = lambda messages, **kw: http.chat({"model": cfg.model, "messages": messages, **kw},  # noqa: E731
                                                    timeout=cfg.call_timeout)
            if args.sim_llm:
                sim_cfg = NovaConfig.from_env(max_calls=10_000, max_input_tokens=10**9, max_output_tokens=10**9)
                patient_llm = NovaLLM(sim_cfg, budget=Budget(sim_cfg), name="simulated-patient")
        client = LocalClient(case, cfg, chat=chat, patient_llm=patient_llm)
        consult(client, prepare())
        rec = _CASE.get("last_record") or {}
        _print_case(rec, client.env)
        if args.out:
            Path(args.out).write_text(json.dumps({**rec, "violations": client.env.violations}, ensure_ascii=False,
                                                 indent=1, default=str), encoding="utf-8")
        return 0

    if args.stdio:
        from dxagent.nova.agent import NovaAgent
        from dxagent.nova.env import StdioEnv

        agent = NovaAgent(cfg, offline=args.offline)
        env = StdioEnv()
        while not env.eof:
            try:
                agent.run(env)
            except EOFError:
                break
            except Exception as exc:  # never die between cases
                log.error("case failed outside the encounter: %s", exc)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        sys.exit(_sdk_entry())
    sys.exit(_dev_main())
