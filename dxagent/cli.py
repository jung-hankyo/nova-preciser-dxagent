"""Command-line interface.

    dxagent chat                 talk to the agent as the patient (you also report exam and test results)
    dxagent chat --offline       same loop with keyword stand-ins, no API key needed (plumbing check)
    dxagent bench -n 20          run on AgentClinic MedQA cases with a simulated patient, then score
    dxagent bench --agent baseline -n 20
    dxagent download             fetch AgentClinic datasets into data/agentclinic/

    dxagent bench-report RUN_DIR [RUN_DIR ...] --out FILE   one summary file for several runs (v0.4.1)

    Knowledge base (v0.4: compiled offline from cited sources; nothing is drafted during a case):
    dxagent kb fetch             download MONDO, HPO and HPO annotations into sources/ontology; build indexes in kb/
    dxagent kb sources [--check] list the source manifest; --check verifies files and records SHA-256
    dxagent kb compile [list]    compile kb/ from the manifest (and an optional disease list file)
    dxagent kb coverage <list>   which diseases in a list resolve, have a card, inherit one, or are missing
    dxagent kb stats | list | show <id or name>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
import time
from pathlib import Path

from dxagent import __version__
from dxagent.config import PROJECT_DIR, get_settings

TYPE_LABEL = {
    "history": "",
    "exam": "Examination",
    "poc": "Bedside test",
    "lab": "Lab order",
    "imaging": "Imaging order",
    "advanced_imaging": "Imaging order",
    "procedure": "Procedure",
}

HELP = """Commands: /state (differential)  /findings  /help  /quit
You are the patient. When the doctor examines you or orders tests, type the findings or results
as the clinical team would report them (e.g. "Temp 38.4 C, HR 112, crackles at the right base")."""


def _wrap(prefix: str, text: str) -> str:
    return textwrap.fill(text, width=100, initial_indent=prefix, subsequent_indent=" " * len(prefix))


def _print_state(st: dict) -> None:
    diff = st.get("differential", [])
    if not diff:
        print("  (no differential yet)")
        return
    print("  Differential:")
    for d in diff[:8]:
        flag = " [can't-miss]" if d.get("cant_miss") else ""
        card = "" if d.get("card_status") == "compiled" else f" ({d.get('card_status')})"
        print(f"    {d['prob']:6.1%}  {d['name']}{flag}{card}")
    print(f"    {st.get('other_prob', 0):6.1%}  other")
    print(f"  Entropy {st.get('entropy', 0):.2f} | turn {st.get('turn', 0)} | test cost {st.get('cost_spent', 0):.0f}")


def _print_findings(st: dict) -> None:
    for f in sorted(st.get("findings", []), key=lambda x: x["turn"]):
        sign = "+" if f["present"] else "-"
        print(f"    T{f['turn']} {sign}{f['label'] or f['code']} ({f['source']}, conf {f['confidence']:.2f})")


KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google_genai": "GOOGLE_API_KEY",
           "deepseek": "DEEPSEEK_API_KEY"}


def _preflight(specs: list[str]) -> bool:
    """Fail early with a clear message if a provider key is missing."""
    import os

    missing = sorted({KEY_ENV[s.split(":", 1)[0]] for s in specs if s.split(":", 1)[0] in KEY_ENV and not os.getenv(KEY_ENV[s.split(":", 1)[0]])})
    if missing:
        print(f"Missing API key(s): {', '.join(missing)}. Put them in .env (see .env.example), or add --offline for a plumbing check.")
        return False
    return True


def _build_runtime(args, settings=None):
    from dxagent.runtime import Runtime

    settings = settings or get_settings()
    if getattr(args, "max_turns", None):
        settings.max_turns = args.max_turns
    if getattr(args, "offline", False):
        from dxagent.offline import make_offline_llm
        from dxagent.runtime import make_store

        settings.contrast_cards = settings.live_fallback = settings.local_fallback = False
        store = make_store(settings, fixtures=True)  # demo fixtures: plumbing only, not knowledge
        return Runtime.create(settings, doctor=make_offline_llm(store), store=store)
    return Runtime.create(settings)


def cmd_chat(args) -> int:
    from dxagent.session import DxSession

    s = get_settings()
    if not args.offline and not _preflight([s.doctor_model, s.card_model, s.prior_model or s.doctor_model]):
        return 2
    rt = _build_runtime(args)
    sess = DxSession(rt)
    print(f"dxagent {__version__} | doctor model: {rt.doctor.name} | cards loaded: {len(rt.store)}")
    print(HELP)
    print()
    try:
        opening = input("Patient> ").strip()
    except (EOFError, KeyboardInterrupt):
        return 0
    if not opening:
        opening = "I haven't been feeling well."
    res = sess.start(opening, source="patient")
    while True:
        label = TYPE_LABEL.get(res.action_type, "")
        if res.done:
            break
        print()
        print(_wrap(f"Doctor{' [' + label + ']' if label else ''}: ", res.message))
        if args.show_state:
            _print_state(sess.state)
        prompt = "Patient> " if res.action_type == "history" else "Results> "
        while True:
            try:
                text = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if text in ("/quit", "/exit"):
                return 0
            if text == "/state":
                _print_state(sess.state)
                continue
            if text == "/findings":
                _print_findings(sess.state)
                continue
            if text == "/help":
                print(HELP)
                continue
            if text:
                break
        res = sess.reply(text)

    rep = res.report or {}
    print()
    print(_wrap("Doctor: ", res.message))
    print()
    print("Final report")
    print(f"  Diagnosis: {rep.get('final_diagnosis')}")
    print(_wrap("  Rationale: ", rep.get("rationale", "")))
    for d in rep.get("differential", [])[:5]:
        print(f"    {d['prob']:6.1%}  {d['name']}")
    print(f"  Committed because: {rep.get('commit_reason')}")
    print(f"  Turns: {rep.get('turns')} | test cost: {rep.get('cost_spent', 0):.0f}")
    if rep.get("recommended_next_steps"):
        print("  Next steps: " + "; ".join(rep["recommended_next_steps"]))
    print("  This is a research prototype, not medical advice.")

    out = PROJECT_DIR / "runs" / f"chat_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    st = sess.state
    out.write_text(json.dumps({k: st.get(k) for k in ("transcript", "findings", "differential", "action_log", "final_report", "critic_notes")}, indent=2, ensure_ascii=False))
    print(f"  Saved transcript: {out}")
    return 0


def cmd_bench(args) -> int:
    from dxagent.bench.runner import BenchModels, run_benchmark

    settings = get_settings()
    if args.max_turns:
        settings.max_turns = args.max_turns
    if args.objective:
        settings.objective = args.objective
    if args.perturb:
        settings.kb_perturb, settings.kb_perturb_seed = args.perturb, args.perturb_seed
    if not args.offline and not _preflight([settings.doctor_model, settings.card_model, settings.prior_model or settings.doctor_model,
                                             settings.patient_model, settings.measurement_model, settings.judge_model]):
        return 2
    rt, models = None, None
    if args.offline:
        from dxagent.offline import make_offline_simulator

        rt = _build_runtime(args, settings)
        sim = make_offline_simulator()
        models = BenchModels(patient=sim, measurement=sim, judge=rt.doctor, baseline_doctor=None)
    summary = run_benchmark(
        settings,
        dataset=args.dataset,
        n=args.n,
        start=args.start,
        agent=args.agent,
        workers=args.workers,
        out_dir=Path(args.out) if args.out else None,
        verbose=args.verbose,
        data_path=Path(args.data) if args.data else None,
        rt=rt,
        models=models,
    )
    print()
    print(json.dumps(summary, indent=2))
    return 0


def cmd_bench_report(args) -> int:
    from dxagent.bench.report import bundle
    from dxagent.runtime import make_store

    store = None if args.no_kb else make_store(get_settings())
    data = bundle([Path(d) for d in args.runs], Path(args.out), store)
    for r in data["runs"]:
        m = r["metrics"]
        print(f"{r['run']:40s} n={m.get('n_ok')} top1 {m.get('top1')} top3 {m.get('top3')} brier {m.get('brier_top1')} "
              f"ece {m.get('ece')} turns {m.get('turns_mean')} cost {m.get('cost_mean')}")
    for c in data["comparisons"]:
        print(f"{c['a']} -> {c['b']}: n={c['n']} diff {c.get('diff')} ci {c.get('ci95')} p={c.get('mcnemar_p')}")
    print(f"-> {args.out}")
    return 0


def cmd_download(args) -> int:
    from dxagent.env.agentclinic import DATASETS, download

    dest = PROJECT_DIR / "data" / "agentclinic"
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for p in download(dest, names):
        print(f"ok  {p}")
    return 0


def cmd_kb(args) -> int:
    s = get_settings()
    if args.action == "fetch":
        from dxagent.knowledge.compiler.fetch import fetch

        fetch(s.sources_dir, s.kb_dir)
        return 0
    if args.action == "sources":
        from dxagent.knowledge.compiler.sources import Manifest, sha256_file

        mpath = s.sources_dir / "manifest.yaml"
        m = Manifest.load(mpath)
        for o in m.ontologies:
            print(f"ontology {o.id:16s} {o.version:28s} {o.path}")
        bad = 0
        for d in m.sources:
            path = s.sources_dir / d.path if d.path else None
            ok = bool(path and path.exists())
            bad += not ok
            if args.check and ok:
                d.sha256 = sha256_file(path)
            print(f"{'ok ' if ok else 'MISSING'} {d.id:28s} {d.kind:18s} tier {d.tier} {d.edition:10s} {len(d.diseases):3d} diseases  {d.licence}")
        if args.check:
            m.save(mpath)
        print(f"{len(m.sources)} sources, {bad} missing files ({mpath})")
        return 1 if bad else 0
    if args.action == "compile":
        from dxagent.knowledge.compiler.compiler import KnowledgeCompiler

        llm = None
        names = None
        if args.target:
            names = [l.strip() for l in Path(args.target).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
        if not args.no_llm:
            if not _preflight([s.card_model]):
                print("(use --no-llm to compile structured sources only, or extractions already cached)")
                return 2
            from dxagent.llm import LLM

            llm = LLM(s.card_model, 0.0 if s.temperature is None else s.temperature, s.max_tokens)
        res = KnowledgeCompiler(s, llm).compile(extra_diseases=names, hpo_scope=args.hpo_scope)
        print(json.dumps(res["build"]["counts"], indent=2))
        return 0

    from dxagent.runtime import make_store

    store = make_store(s)
    if args.action == "coverage":
        if not args.target:
            print("usage: dxagent kb coverage <disease list file>")
            return 1
        names = [l.strip() for l in Path(args.target).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
        from collections import Counter

        tally: Counter = Counter()
        for n in names:
            cid, oid, how = store.resolve_disease(n)
            if not cid:
                status = "unresolved"
            elif store.get(cid) is not None:
                status = f"card ({store.get(cid).provenance.status})"
            elif store.inherited_from(cid):
                status = f"inherits {store.inherited_from(cid)}"
            else:
                status = "no knowledge"
            tally[status.split(" ")[0]] += 1
            print(f"{n[:44]:44s} {oid or '-':16s} {how:10s} {status}")
        print(", ".join(f"{k} {v}" for k, v in tally.most_common()) + f" (of {len(names)})")
        return 0
    if args.action in ("list", "stats"):
        from collections import Counter

        if args.action == "list":
            for cid, name in sorted(store.known_diseases()):
                print(f"{cid:20s} {name}")
        cards_n = len(store)
        print(f"{cards_n} disease cards, {len(store._contrasts)} contrast cards, {len(store.tests())} tests, {len(store.vocabulary())} findings")
        if store.kb_info:
            print(json.dumps({k: store.kb_info.get(k) for k in ("compiler", "compiled_at", "counts", "ontology_index_versions")}, indent=2))
        if cards_n <= 2000:
            print("by status: " + ", ".join(f"{k} {v}" for k, v in Counter(c.provenance.status for c in store.all_cards()).items()))
        return 0
    if args.action == "show":
        card = store.get(args.target or "")
        if not card:
            cid, oid, how = store.resolve_disease(args.target or "")
            card = store.get(cid) if cid else None
            if not card:
                via = store.inherited_from(cid) if cid else None
                print(f"No card for {args.target!r}" + (f" ({oid}; inherits {via})" if via else (f" ({oid})" if oid else "")))
                return 1
        print(json.dumps(card.model_dump(mode="json", exclude_none=True, exclude_defaults=True), indent=1))
        for cc, side in store.contrasts_for(card.id):
            print(f"contrast: {cc.a} vs {cc.b} ({len(cc.findings)} findings, {len(cc.tests)} tests)")
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dxagent", description="Knowledge-first diagnostic agent (LangGraph).")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--log", default="WARNING", help="log level (DEBUG, INFO, WARNING)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chat", help="talk to the agent in the terminal (you are the patient)")
    c.add_argument("--show-state", action="store_true", help="print the differential after every turn")
    c.add_argument("--max-turns", type=int, default=None)
    c.add_argument("--offline", action="store_true", help="keyword stand-ins instead of an LLM (no API key)")
    c.set_defaults(fn=cmd_chat)

    b = sub.add_parser("bench", help="run on AgentClinic cases and score")
    b.add_argument("--dataset", default="medqa", choices=["medqa", "medqa_extended"])
    b.add_argument("-n", type=int, default=None, help="number of cases (default: all)")
    b.add_argument("--start", type=int, default=0)
    b.add_argument("--agent", default="dx", choices=["dx", "baseline"])
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--max-turns", type=int, default=None)
    b.add_argument("--out", default=None, help="output directory")
    b.add_argument("--data", default=None, help="path to a custom AgentClinic-format JSONL")
    b.add_argument("--verbose", action="store_true", help="print each conversation (workers=1 only)")
    b.add_argument("--offline", action="store_true", help="plumbing check with offline stand-ins")
    b.add_argument("--objective", choices=["accuracy", "clinical"], default=None, help="loss: exact diagnosis or harm-weighted")
    b.add_argument("--perturb", type=float, default=None, help="robustness check: scale every frequency's odds by up to x/÷ this factor")
    b.add_argument("--perturb-seed", type=int, default=0)
    b.set_defaults(fn=cmd_bench)

    d = sub.add_parser("download", help="download AgentClinic datasets")
    d.add_argument("--dataset", default="all", choices=["all", "medqa", "medqa_extended"])
    d.set_defaults(fn=cmd_download)

    r = sub.add_parser("bench-report", help="combine benchmark runs into one summary file for analysis")
    r.add_argument("runs", nargs="+", help="run directories (each with results.jsonl)")
    r.add_argument("--out", default="dxagent_live_summary.json")
    r.add_argument("--no-kb", action="store_true", help="skip the truth-coverage check against the current knowledge base")
    r.set_defaults(fn=cmd_bench_report)

    k = sub.add_parser("kb", help="knowledge base: fetch ontologies, compile from sources, inspect")
    k.add_argument("action", choices=["fetch", "sources", "compile", "coverage", "stats", "list", "show"])
    k.add_argument("target", nargs="?", help="disease list file (compile/coverage) or disease id or name (show)")
    k.add_argument("--check", action="store_true", help="sources: verify files and record SHA-256 in the manifest")
    k.add_argument("--no-llm", action="store_true", help="compile: structured sources and cached extractions only")
    k.add_argument("--hpo-scope", choices=["all", "listed", "none"], default=None, help="compile: which HPO-annotated diseases")
    k.set_defaults(fn=cmd_kb)

    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.WARNING), format="%(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
