"""Run the agent (or the baseline) on AgentClinic cases and score it (plan section 7).

Outputs, under runs/<timestamp>_<agent>_<dataset>/:
    results.jsonl   one record per case: transcript, differential, verdict, turns, cost, errors, LLM usage (v0.4.1)
    summary.json    top-1 / top-3 accuracy, mean turns, mean test cost, Brier score, calibration bins
    run_meta.json   version, git commit, models, settings snapshot, knowledge-base build record (v0.4.1)
    report.json     the single-run report of dxagent/bench/report.py (v0.4.1); combine runs with `dxagent bench-report`
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from dxagent import __version__, usage
from dxagent.bench.baseline import run_baseline_case
from dxagent.config import PROJECT_DIR, Settings
from dxagent.env.agentclinic import DATASETS, Case, SimulatedEncounter, download, load_cases
from dxagent.llm import LLM, LLMLike
from dxagent.prompts import JUDGE_SYSTEM, judge_user
from dxagent.runtime import Runtime
from dxagent.schemas import JudgeOutput
from dxagent.session import DxSession

log = logging.getLogger(__name__)


@dataclass
class BenchModels:
    patient: LLMLike
    measurement: LLMLike
    judge: LLMLike
    baseline_doctor: Optional[LLMLike] = None

    @classmethod
    def from_settings(cls, s: Settings) -> "BenchModels":
        return cls(
            patient=LLM(s.patient_model, s.temperature, 1024, role="patient"),
            measurement=LLM(s.measurement_model, s.temperature, 2048, role="measurement"),
            judge=LLM(s.judge_model, s.temperature, 1024, role="judge"),
        )


def family(spec: str) -> str:
    provider, _, model = spec.partition(":")
    return provider


def judge_case(judge: LLMLike, reference: str, final: str, differential: list[str]) -> JudgeOutput:
    try:
        return judge.structured(JudgeOutput, JUDGE_SYSTEM, judge_user(reference, final, differential))
    except Exception as exc:
        log.warning("Judge failed: %s", exc)
        return JudgeOutput(top1_correct=False, match_rank=None, rationale=f"judge error: {exc}")


def run_dx_case(rt: Runtime, enc: SimulatedEncounter, echo: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
    sess = DxSession(rt)
    res = sess.start(enc.presentation(), source="referral")
    if echo:
        echo(f"  [referral] {enc.presentation()}")
    guard = rt.settings.max_turns + 5
    while not res.done and guard > 0:
        guard -= 1
        reply = enc.respond(res.message, res.action_type)
        if echo:
            echo(f"  [doctor:{res.action_type}] {res.message}")
            echo(f"  [{'patient' if res.action_type == 'history' else 'records'}] {reply}")
        res = sess.reply(reply)
    st = sess.state
    report = res.report or {}
    return {
        "final_diagnosis": report.get("final_diagnosis", "Undetermined"),
        "differential": [d["name"] for d in report.get("differential", [])],
        "differential_probs": report.get("differential", []),
        "top1_prob": report.get("differential", [{}])[0].get("prob") if report.get("differential") else None,
        "turns": st.get("turn", 0),
        "cost_spent": st.get("cost_spent", 0.0),
        "commit_reason": report.get("commit_reason", ""),
        "cards_used": report.get("cards_used", {}),
        "critic_notes": st.get("critic_notes", []),
        "action_log": st.get("action_log", []),
        "transcript": st.get("transcript", []),
        "hypothesis_trace": st.get("hypothesis_trace", []),
        "n_hypotheses": len(st.get("hypotheses", [])),
        "list_effect_log": st.get("list_effect_log", []),  # v0.3 B1 diagnostic
        "other_prob": report.get("other_prob"),
        "report_dissent": report.get("report_dissent", []),  # E1
        "llm_final_diagnosis": report.get("llm_final_diagnosis", ""),
        "vetoes": report.get("vetoes", []),  # D2
        "unresolved_objection": report.get("unresolved_objection"),
        # v0.3.1 (P1) diagnostics
        "prior_context": report.get("prior_context"),
        "prior_log": report.get("prior_log", []),
        "prior_sensitivity": report.get("prior_sensitivity", {}),
        "context_coverage": report.get("context_coverage", {}),
        # v0.4.1 knowledge-layer diagnostics
        "coverage_misses": report.get("coverage_misses", []),
        "knowledge_base": report.get("knowledge_base"),
        "evidence": report.get("evidence", {}),
        "n_findings": report.get("n_findings"),
        "n_tests": report.get("n_tests"),
        # v0.4.1 module logs
        "perceive_log": st.get("perceive_log", []),
        "contrast_log": st.get("contrast_log", []),
        "actions_taken": st.get("actions_taken", []),
        "test_results": [{"test_id": t.get("test_id"), "outcome": t.get("outcome"), "turn": t.get("turn")} for t in st.get("test_results", [])],
        "final_hypotheses": [{"disease_id": h["disease_id"], "name": h.get("name"), "source": h.get("prior_source", ""),
                              "added_turn": h.get("added_turn")} for h in st.get("hypotheses", [])],
    }


def judge_alternatives(judge: LLMLike, reference: str, out: dict[str, Any], limit: int = 6) -> dict[str, bool]:
    """v0.4.1: is each alternative the agent's modules named (report dissent, critic veto alternatives, the report
    LLM's own diagnosis) the reference diagnosis? One judge call per distinct name, at most `limit`."""
    names: list[str] = []
    for d in out.get("report_dissent", []) or []:
        names.append(d.get("dissent", ""))
    for v in out.get("vetoes", []) or []:
        sig = v.get("sig") or []
        if len(sig) > 1:
            names.append(str(sig[1]).replace("_", " "))
    if out.get("llm_final_diagnosis"):
        names.append(out["llm_final_diagnosis"])
    res: dict[str, bool] = {}
    for n in dict.fromkeys(n for n in names if n and n.strip()):
        if len(res) >= limit:
            break
        if n.strip().lower() == (out.get("final_diagnosis") or "").strip().lower():
            continue
        res[n] = bool(judge_case(judge, reference, n, [n]).top1_correct)
    return res


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if not r.get("error")]
    n = len(ok)
    if n == 0:
        return {"n": 0, "errors": len(records)}
    top1 = sum(r["top1_correct"] for r in ok) / n
    top3 = sum(1 for r in ok if r.get("match_rank") and r["match_rank"] <= 3) / n
    summary: dict[str, Any] = {
        "n": n,
        "errors": len(records) - n,
        "top1_accuracy": round(top1, 4),
        "top3_accuracy": round(top3, 4),
        "mean_turns": round(sum(r["turns"] for r in ok) / n, 2),
        "mean_test_cost": round(sum(r["cost_spent"] for r in ok) / n, 1),
        "leak_flags": sum(r.get("leak_flags", 0) for r in ok),
    }
    probs = [(r["top1_prob"], r["top1_correct"]) for r in ok if r.get("top1_prob") is not None]
    if probs:
        summary["brier_top1"] = round(sum((p - float(c)) ** 2 for p, c in probs) / len(probs), 4)
        bins = []
        for lo in [0.0, 0.2, 0.4, 0.6, 0.8]:
            sel = [(p, c) for p, c in probs if lo <= p < lo + 0.2 or (lo == 0.8 and p == 1.0)]
            if sel:
                bins.append({"bin": f"{lo:.1f}-{lo + 0.2:.1f}", "n": len(sel), "mean_conf": round(sum(p for p, _ in sel) / len(sel), 3), "accuracy": round(sum(c for _, c in sel) / len(sel), 3)})
        summary["calibration"] = bins
        summary.update(list_diagnostics(ok, bins_conf=probs))
    full = [brier_full(r) for r in ok]
    full = [b for b in full if b is not None]
    if full:
        summary["brier_full"] = round(sum(full) / len(full), 4)  # G4: over the whole differential plus "other"
    summary["records_mentions"] = sum(len(r.get("leaks", {}).get("records", [])) for r in ok)  # G2
    dis = [r for r in ok if r.get("report_dissent")]
    summary["report_dissent_cases"] = len(dis)  # E1
    summary.update(p1_summary(ok))
    return summary


def p1_summary(ok: list[dict[str, Any]]) -> dict[str, Any]:
    """v0.3.1 diagnostics: does the prior decide answers, does the generator leak, is context lost."""
    from collections import Counter

    from dxagent.priors import BUCKETS

    out: dict[str, Any] = {}
    sens = [r["prior_sensitivity"] for r in ok if r.get("prior_sensitivity")]
    if sens:
        out["prior_decides_flat"] = round(sum(bool(s["changed_flat"]) for s in sens) / len(sens), 4)
        out["prior_decides_tempered"] = round(sum(bool(s["changed_tempered"]) for s in sens) / len(sens), 4)
    logs = [e for r in ok for e in r.get("prior_log", [])]
    gaps = [abs(BUCKETS.index(e["assessor"]) - BUCKETS.index(e["proposed"])) for e in logs if e.get("assessor") and e.get("proposed")]
    if gaps:  # leak detector: the proposer saw the findings, the assessor did not
        out["prior_leak_gap"] = {"mean_buckets": round(sum(gaps) / len(gaps), 3), "share_ge2": round(sum(g >= 2 for g in gaps) / len(gaps), 4),
                                 "n": len(gaps)}
    votes = [e["votes"] for e in logs if len(e.get("votes", [])) > 1]
    if votes:  # bucket stability across the K assessor samples
        out["prior_vote_disagreement"] = round(sum(len(set(v)) > 1 for v in votes) / len(votes), 4)
    if logs:
        out["prior_routes"] = dict(Counter(e["route"] for e in logs))
    ctx = [r["prior_context"] for r in ok if r.get("prior_context")]
    if ctx:
        out["complaint_rule"] = dict(Counter(c.get("rule") for c in ctx))
        out["complaint_finding_match"] = dict(Counter(c.get("finding_match") for c in ctx))
    cov = [r["context_coverage"] for r in ok if r.get("context_coverage")]
    if cov:
        tot, miss = sum(c["opening_intake"][0] for c in cov), sum(c["opening_intake"][1] for c in cov)
        out["context_uncovered_opening_intake"] = round(miss / tot, 4) if tot else None
        cats: dict[str, list[int]] = {}
        for c in cov:
            for k, (n, m) in c.get("by_category", {}).items():
                cats.setdefault(k, [0, 0])
                cats[k][0] += n
                cats[k][1] += m
        out["context_uncovered_by_category"] = {k: {"n": n, "uncovered": round(m / n, 3)} for k, (n, m) in sorted(cats.items())}
    return out


def brier_full(r: dict[str, Any]) -> float | None:
    """G4: multiclass Brier over the reported differential plus "other", with the truth at the judge's match rank
    (or "other" when the truth is not listed). Consistent with top-1 now that the answer is the computed top."""
    probs = r.get("differential_probs") or []
    if not probs or r.get("other_prob") is None:
        return None
    rank = r.get("match_rank")
    tot = sum((d["prob"] - (1.0 if rank == i + 1 else 0.0)) ** 2 for i, d in enumerate(probs))
    return tot + (r["other_prob"] - (1.0 if not rank or rank > len(probs) else 0.0)) ** 2


def _ece(pairs: list[tuple[float, bool]]) -> float:
    if not pairs:
        return float("nan")
    tot = 0.0
    for lo in [0.0, 0.2, 0.4, 0.6, 0.8]:
        sel = [(p, c) for p, c in pairs if lo <= p < lo + 0.2 or (lo == 0.8 and p == 1.0)]
        if sel:
            tot += len(sel) * abs(sum(p for p, _ in sel) / len(sel) - sum(c for _, c in sel) / len(sel))
    return tot / len(pairs)


def list_diagnostics(records: list[dict[str, Any]], bins_conf=None) -> dict[str, Any]:
    """v0.3 B1 diagnostic: does list composition, not evidence, move the reported probabilities?

    - list_effect: per list change, leader probability with the old list vs the new list, same evidence.
    - calibration_by_list_length: ECE of the top-1 probability, split by differential size at commit.
    """
    deltas = [e["p_new_list"] - e["p_same_evidence_old_list"] for r in records for e in r.get("list_effect_log", [])]
    out: dict[str, Any] = {}
    if deltas:
        out["list_effect"] = {
            "events": len(deltas),
            "mean_abs_delta": round(sum(abs(d) for d in deltas) / len(deltas), 4),
            "max_abs_delta": round(max(abs(d) for d in deltas), 4),
        }
    groups: dict[str, list[tuple[float, bool]]] = {}
    for r in records:
        if r.get("top1_prob") is None or r.get("n_hypotheses") is None:
            continue
        n = r["n_hypotheses"]
        key = "<=5" if n <= 5 else "6-8" if n <= 8 else ">=9"
        groups.setdefault(key, []).append((r["top1_prob"], bool(r["top1_correct"])))
    if groups:
        out["calibration_by_list_length"] = {k: {"n": len(v), "ece": round(_ece(v), 4),
                                                 "mean_conf": round(sum(p for p, _ in v) / len(v), 3),
                                                 "accuracy": round(sum(c for _, c in v) / len(v), 3)}
                                             for k, v in sorted(groups.items())}
    return out


def run_benchmark(
    settings: Settings,
    dataset: str = "medqa",
    n: Optional[int] = None,
    start: int = 0,
    agent: str = "dx",
    workers: int = 1,
    out_dir: Optional[Path] = None,
    verbose: bool = False,
    data_dir: Optional[Path] = None,
    data_path: Optional[Path] = None,
    rt: Optional[Runtime] = None,
    models: Optional[BenchModels] = None,
    printer: Callable[[str], None] = print,
) -> dict[str, Any]:
    if data_path is None:
        data_dir = data_dir or PROJECT_DIR / "data" / "agentclinic"
        data_path = data_dir / DATASETS[dataset]
        if not data_path.exists():
            printer(f"Downloading {DATASETS[dataset]} ...")
            download(data_dir, [dataset])
    cases: list[Case] = load_cases(data_path)
    cases = cases[start : start + n] if n else cases[start:]

    if models is None and family(settings.doctor_model) == family(settings.patient_model):
        printer(
            "Note: doctor and patient share a model family; AgentClinic found this can inflate scores. "
            "Set DX_PATIENT_MODEL to another provider for headline results."
        )

    models = models or BenchModels.from_settings(settings)
    if agent == "dx":
        rt = rt or Runtime.create(settings)
    elif agent == "baseline":
        models.baseline_doctor = models.baseline_doctor or LLM(settings.doctor_model, settings.temperature, 1024, role="doctor")
    else:
        raise ValueError(f"unknown agent {agent!r}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or PROJECT_DIR / "runs" / f"{stamp}_{agent}_{dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    lock = threading.Lock()
    records: list[dict[str, Any]] = []

    from dxagent.bench.report import _git_commit, run_report, settings_snapshot

    meta = {"dxagent_version": __version__, "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "agent": agent, "dataset": dataset, "n_cases": len(cases),
            "models": {"doctor": settings.doctor_model, "card": settings.card_model, "prior": settings.prior_model,
                       "patient": models.patient.name, "measurement": models.measurement.name, "judge": models.judge.name},
            "settings": settings_snapshot(settings),
            "knowledge_base": (rt.store.kb_info if rt is not None and getattr(rt.store, "kb_info", None) else None),
            "cards_loaded": len(rt.store) if rt is not None else None}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")

    def one(case: Case) -> dict[str, Any]:
        usage.set_case(case.index)
        enc = SimulatedEncounter(case, models.patient, models.measurement, terms_llm=models.judge)
        t0 = time.time()
        rec: dict[str, Any] = {"index": case.index, "reference": case.diagnosis, "objective": case.objective}
        try:
            echo = printer if (verbose and workers == 1) else None
            if agent == "dx":
                try:  # v0.4.1: the reference's identity and knowledge status as of this case (measurement only)
                    from dxagent.bench.report import truth_status

                    rec["truth"] = truth_status(case.diagnosis, rt.store)
                except Exception as exc:
                    rec["truth"] = {"status": "error", "error": str(exc)[:80]}
                out = run_dx_case(rt, enc, echo)
            else:
                out = run_baseline_case(models.baseline_doctor, enc, settings.max_turns)
            verdict = judge_case(models.judge, case.diagnosis, out["final_diagnosis"], out["differential"] or [out["final_diagnosis"]])
            rec.update(out)
            rec.update({"top1_correct": verdict.top1_correct, "match_rank": verdict.match_rank, "judge_rationale": verdict.rationale})
            if agent == "dx":
                rec["alt_judgments"] = judge_alternatives(models.judge, case.diagnosis, out)
        except Exception as exc:
            rec.update({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        rec["leak_flags"] = enc.leak_flags
        rec["leaks"] = enc.leaks
        rec["seconds"] = round(time.time() - t0, 1)
        rec["llm_usage"] = usage.case_usage(case.index)
        usage.set_case(None)
        with lock:
            with open(results_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    printer(f"Running {agent} on {len(cases)} {dataset} cases -> {out_dir}")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, c): c for c in cases}
        for fut in as_completed(futures):
            rec = fut.result()
            records.append(rec)
            if rec.get("error"):
                printer(f"[{len(records)}/{len(cases)}] case {rec['index']}: ERROR {rec['error']}")
            else:
                mark = "correct" if rec["top1_correct"] else "wrong"
                printer(
                    f"[{len(records)}/{len(cases)}] case {rec['index']}: {mark} | ref: {rec['reference']} | "
                    f"dx: {rec['final_diagnosis']} | turns {rec['turns']} | cost {rec['cost_spent']:.0f}"
                )

    summary = summarize(records)
    doctor_name = rt.doctor.name if rt is not None else models.baseline_doctor.name
    summary.update({"agent": agent, "dataset": dataset, "doctor_model": doctor_name, "patient_model": models.patient.name})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        (out_dir / "report.json").write_text(json.dumps(run_report(out_dir, rt.store if rt is not None else None), indent=1,
                                                        ensure_ascii=False, default=str), encoding="utf-8")
    except Exception as exc:  # the report must never lose a finished run
        log.warning("report.json failed: %s", exc)
    return summary
