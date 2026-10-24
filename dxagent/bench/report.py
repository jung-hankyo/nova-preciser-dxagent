"""One file to analyze a live benchmark (v0.4.1).

    dxagent bench-report runs/A runs/B ... --out dxagent_live_summary.json

For every run directory (from `dxagent bench`, v0.3.1 or later; `results.jsonl` is required) the bundle holds:

    meta        dxagent version and git commit (of the reporting code and, if recorded, of the run), dataset, agent,
                models, settings snapshot, knowledge-base build record and source manifest summary
    metrics     top-1 / top-3, Brier, ECE, calibration bins, mean confidence, confident errors (p >= 0.8, wrong),
                turns, test cost, seconds, commit reasons, dissent and veto counts, errors
    usage       LLM calls and tokens per case, by role (doctor, knowledge, prior, patient, measurement, judge)
    knowledge   (dx agent) whether the reference diagnosis resolves in the ontology and has knowledge in the
                CURRENT knowledge base (own card status, inherited, none, unresolved); accuracy by that status;
                coverage misses, fallback outcomes, card status of the reported differential
    cases       one compact row per case; the wrong ones also carry a truncated transcript

and `comparisons`: every pair of runs on the same dataset, paired by case index (McNemar exact p, bootstrap CI of the
top-1 difference, discordant cases). The reference diagnoses are used only to MEASURE coverage here; never compile
knowledge from them.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

from dxagent import __version__

TRANSCRIPT_LINES = 40
TRANSCRIPT_CHARS = 240


def _git_commit(path: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=path, capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception:  # not a git checkout
        return ""


def _load(run_dir: Path) -> tuple[list[dict], dict, dict]:
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) if (run_dir / "summary.json").exists() else {}
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8")) if (run_dir / "run_meta.json").exists() else {}
    return rows, summary, meta


def _ece(pairs: list[tuple[float, bool]]) -> float:
    tot = 0.0
    for lo in [0.0, 0.2, 0.4, 0.6, 0.8]:
        sel = [(p, c) for p, c in pairs if lo <= p < lo + 0.2 or (lo == 0.8 and p >= 1.0)]
        if sel:
            tot += len(sel) * abs(sum(p for p, _ in sel) / len(sel) - sum(c for _, c in sel) / len(sel))
    return tot / len(pairs) if pairs else 0.0


def _mean(xs: list[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 4) if xs else None


def metrics(rows: list[dict]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    n = len(ok)
    out: dict[str, Any] = {"n": len(rows), "n_ok": n, "errors": len(rows) - n}
    if not n:
        return out
    pairs = [(float(r["top1_prob"]), bool(r["top1_correct"])) for r in ok if r.get("top1_prob") is not None]
    out.update({
        "top1": round(sum(bool(r.get("top1_correct")) for r in ok) / n, 4),
        "top3": round(sum(1 for r in ok if r.get("match_rank") and r["match_rank"] <= 3) / n, 4),
        "in_differential": round(sum(1 for r in ok if r.get("match_rank")) / n, 4),
        "turns_mean": _mean([r.get("turns", 0) for r in ok]), "turns_median": statistics.median(r.get("turns", 0) for r in ok),
        "cost_mean": _mean([r.get("cost_spent", 0.0) for r in ok]), "cost_median": statistics.median(r.get("cost_spent", 0.0) for r in ok),
        "seconds_mean": _mean([r.get("seconds", 0.0) for r in ok]),
        "commit_reasons": dict(Counter((r.get("commit_reason") or "").split(" ")[0] or "none" for r in ok).most_common()),
        "dissent_cases": sum(1 for r in ok if r.get("report_dissent")),
        "veto_cases": sum(1 for r in ok if r.get("vetoes")),
        "leak_flags": sum(r.get("leak_flags", 0) for r in ok),
        "error_messages": dict(Counter((r.get("error") or "")[:80] for r in rows if r.get("error")).most_common(5)),
    })
    if pairs:
        out.update({
            "brier_top1": round(sum((p - c) ** 2 for p, c in pairs) / len(pairs), 4),
            "ece": round(_ece(pairs), 4),
            "mean_confidence": round(sum(p for p, _ in pairs) / len(pairs), 4),
            "confident_errors": round(sum(1 for p, c in pairs if p >= 0.8 and not c) / n, 4),
            "accuracy_at_conf_0.8": _mean([float(c) for p, c in pairs if p >= 0.8]),
            "calibration": [{"bin": f"{lo:.1f}-{lo + 0.2:.1f}", "n": len(sel), "mean_conf": round(sum(p for p, _ in sel) / len(sel), 3),
                             "accuracy": round(sum(c for _, c in sel) / len(sel), 3)}
                            for lo in [0.0, 0.2, 0.4, 0.6, 0.8]
                            for sel in [[(p, c) for p, c in pairs if lo <= p < lo + 0.2 or (lo == 0.8 and p >= 1.0)]] if sel],
        })
    return out


def usage_summary(rows: list[dict]) -> dict[str, Any]:
    """Calls, tokens, error rate and estimated USD cost (bench/prices.py, list prices) per case and role."""
    from dxagent.bench.prices import cost_usd

    ok = [r for r in rows if r.get("llm_usage")]
    if not ok:
        return {"recorded": False}
    fields = ("calls", "errors", "input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens")
    roles: dict[str, Counter] = {}
    models: dict[str, Counter] = {}
    case_costs: list[float] = []
    unpriced: set[str] = set()
    for r in ok:
        for role, v in r["llm_usage"].get("by_role", {}).items():
            roles.setdefault(role, Counter()).update({k: v.get(k, 0) for k in fields})
        cost = 0.0
        for spec, v in (r["llm_usage"].get("by_model") or {}).items():
            models.setdefault(spec, Counter()).update({k: v.get(k, 0) for k in fields})
            c = cost_usd(spec, v.get("input_tokens", 0), v.get("output_tokens", 0), v.get("cache_read_tokens", 0))
            if c is None:
                unpriced.add(spec)
            else:
                cost += c
        case_costs.append(cost)
    n = len(ok)
    model_cost = {spec: (round(cost_usd(spec, c["input_tokens"], c["output_tokens"], c["cache_read_tokens"]) / n, 5)
                         if cost_usd(spec, 1, 1) is not None else None) for spec, c in models.items()}
    return {"recorded": True, "cases": n,
            "per_case_by_role": {role: {k: round(v / n, 1) for k, v in c.items()} for role, c in sorted(roles.items())},
            "error_rate_by_role": {role: round(c["errors"] / max(1, c["calls"]), 4) for role, c in sorted(roles.items())},
            "totals_by_role": {role: dict(c) for role, c in sorted(roles.items())},
            "per_case_by_model": {spec: {k: round(v / n, 1) for k, v in c.items()} for spec, c in sorted(models.items())},
            "est_cost_usd_per_case": round(sum(case_costs) / n, 5) if case_costs else None,
            "est_cost_usd_per_case_by_model": model_cost,
            "est_cost_usd_total": round(sum(case_costs), 4),
            "unpriced_models": sorted(unpriced),
            "price_note": "list prices from dxagent/bench/prices.py (DeepSeek at peak rates; off-peak is half); override with DX_PRICES"}


def knowledge_summary(rows: list[dict], store) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    if not ok or not any("differential_probs" in r for r in ok):
        return {"available": False}
    status_rows = []
    for r in ok:
        st = r.get("truth") or (truth_status(r.get("reference", ""), store) if store is not None else {"status": "unknown"})
        r["_truth"] = st
        status_rows.append((st["status"].split(":")[0], bool(r.get("top1_correct"))))
    by: dict[str, list[bool]] = {}
    for s, c in status_rows:
        by.setdefault(s, []).append(c)
    diff_status = Counter()
    for r in ok:
        for d in r.get("differential_probs", [])[:8]:
            diff_status[(d.get("card_status") or "unknown").split(":")[0]] += 1
    misses = [len(r.get("coverage_misses", []) or []) for r in ok]
    fb = Counter(m.get("fallback") or "none" for r in ok for m in (r.get("coverage_misses") or []))
    fb_secs = [m["fallback_seconds"] for r in ok for m in (r.get("coverage_misses") or []) if m.get("fallback_seconds") is not None]
    return {
        "available": True,
        "truth_knowledge_status": {k: {"n": len(v), "top1": round(sum(v) / len(v), 4)} for k, v in sorted(by.items())},
        "differential_card_status": dict(diff_status.most_common()),
        "coverage_misses_per_case": _mean(misses),
        "cases_with_coverage_miss": sum(1 for m in misses if m),
        "fallback_outcomes": dict(fb.most_common()),
        "fallback_seconds_mean": _mean(fb_secs),
        "note": "truth status is measured against the knowledge base present when this report was built",
    }


def truth_status(reference: str, store) -> dict[str, str]:
    cid, oid, how = store.resolve_disease(reference)
    if not cid:
        return {"status": "unresolved", "ontology_id": "", "how": how}
    card = store.get(cid)
    if card is not None:
        return {"status": card.provenance.status, "ontology_id": oid, "how": how, "id": cid}
    via = store.inherited_from(cid)
    return {"status": f"inherited:{via}" if via else "none", "ontology_id": oid, "how": how, "id": cid}


def _transcript(r: dict) -> list[str]:
    lines = []
    for t in r.get("transcript", [])[:TRANSCRIPT_LINES]:
        role = t.get("role", "?")
        text = str(t.get("text", t.get("content", "")))
        lines.append(f"{role}: {text[:TRANSCRIPT_CHARS]}")
    return lines


def case_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in sorted(rows, key=lambda x: x.get("index", 0)):
        row = {
            "index": r.get("index"), "reference": r.get("reference"), "final": r.get("final_diagnosis"),
            "correct": r.get("top1_correct"), "match_rank": r.get("match_rank"), "top1_prob": r.get("top1_prob"),
            "turns": r.get("turns"), "cost": r.get("cost_spent"), "seconds": r.get("seconds"),
            "commit_reason": r.get("commit_reason"), "n_hypotheses": r.get("n_hypotheses"),
            "differential": [[d.get("name"), round(d.get("prob", 0.0), 3), d.get("card_status", "")]
                             for d in (r.get("differential_probs") or [])[:5]],
            "other_prob": r.get("other_prob"),
            "coverage_misses": [[m.get("disease_id"), m.get("fallback", "")] for m in (r.get("coverage_misses") or [])],
            "truth": r.get("truth") or r.get("_truth"), "truth_route": (r.get("_trace") or {}).get("route"),
            "truth_first_turn": (r.get("_trace") or {}).get("first_turn"), "failure": r.get("_fail"),
            "dissent": [d.get("dissent") for d in (r.get("report_dissent") or [])], "vetoes": len(r.get("vetoes") or []),
            "llm_final": r.get("llm_final_diagnosis"), "alt_judgments": r.get("alt_judgments"),
            "leader_evidence": ((r.get("evidence") or {}).get((_final_ids(r) or [""])[0], []))[:6],
            "llm_calls": (r.get("llm_usage") or {}).get("total", {}).get("calls"),
            "tokens": [(r.get("llm_usage") or {}).get("total", {}).get("input_tokens"),
                       (r.get("llm_usage") or {}).get("total", {}).get("output_tokens")],
            "judge": (r.get("judge_rationale") or "")[:200], "error": r.get("error"),
        }
        if not r.get("top1_correct"):
            row["transcript"] = _transcript(r)
        out.append(row)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Module metrics (v0.4.1): one block per agent module, from the logs each case record carries
# ---------------------------------------------------------------------------------------------------------------


def _truth_id(r: dict, store) -> str:
    t = r.get("truth") or r.get("_truth") or {}
    if t.get("id"):
        return t["id"]
    if store is not None and r.get("reference"):
        st = truth_status(r["reference"], store)
        r["_truth"] = st
        return st.get("id", "")
    return ""


def _final_ids(r: dict) -> list[str]:
    return [d.get("disease_id", "") for d in (r.get("differential_probs") or [])]


def _route(src: str) -> str:
    return (src or "unknown").split(":")[0]


def hypothesis_trace_metrics(r: dict, tid: str) -> dict[str, Any]:
    """Truth's path through the differential: first turn listed, entry route, best rank and probability."""
    out: dict[str, Any] = {"traced": False}
    trace = r.get("hypothesis_trace") or []
    if not tid or not trace or "ids" not in trace[-1]:
        return out
    out["traced"] = True
    first = next((t for t in trace if tid in t.get("ids", {})), None)
    out["ever_listed"] = first is not None
    if first is not None:
        out["first_turn"] = first["turn"]
        out["route"] = _route(first["ids"][tid])
        probs = [t.get("probs", {}).get(tid) for t in trace if t.get("probs", {}).get(tid) is not None]
        out["max_prob"] = max(probs) if probs else None
        out["led_at_some_turn"] = any(t.get("probs") and max(t["probs"], key=t["probs"].get) == tid for t in trace)
        led = [t["turn"] for t in trace if t.get("probs") and max(t["probs"], key=t["probs"].get) == tid]
        out["first_lead_turn"] = led[0] if led else None
    out["in_final"] = tid in _final_ids(r)
    return out


def failure_class(r: dict, tid: str, tr: dict[str, Any]) -> str:
    """Why a wrong case failed, in the order the pipeline would have to fix it."""
    if r.get("error"):
        return "error"
    if r.get("top1_correct"):
        return "correct"
    status = ((r.get("truth") or r.get("_truth") or {}).get("status") or "unknown").split(":")[0]
    if not tid:
        return "truth_unresolved"
    finals = _final_ids(r)
    if finals and finals[0] == tid:
        return "naming_or_judge"  # the agent's top is the reference's ontology id, yet judged wrong
    if tr.get("traced") and not tr.get("ever_listed"):
        return "never_listed:" + ("no_knowledge" if status in ("none", "unresolved") else "has_knowledge")
    if tid in finals:
        return "outranked:" + ("no_knowledge" if status in ("none",) else "has_knowledge")
    if tr.get("ever_listed"):
        return "dropped"
    return "not_in_final"


def _bins(values: list[tuple[float, bool]], edges: list[float], label) -> dict[str, Any]:
    out = {}
    for lo, hi in zip(edges, edges[1:]):
        sel = [c for v, c in values if lo <= v < hi]
        if sel:
            out[label(lo, hi)] = {"n": len(sel), "top1": round(sum(sel) / len(sel), 4)}
    return out


def module_metrics(rows: list[dict], store) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error") and "differential_probs" in r]
    if not ok:
        return {"available": False}
    n = len(ok)
    M: dict[str, Any] = {"available": True}

    # -- hypothesis generation and entry routes (decision 11; the generator recall the reverse index trades on)
    tr_rows = []
    for r in ok:
        tid = _truth_id(r, store)
        tr = hypothesis_trace_metrics(r, tid)
        r["_trace"], r["_fail"] = tr, failure_class(r, tid, tr)
        tr_rows.append((r, tr))
    traced = [(r, t) for r, t in tr_rows if t.get("traced")]
    if traced:
        listed = [t for _, t in traced if t.get("ever_listed")]
        routes = Counter(t["route"] for t in listed)
        M["hypotheses"] = {
            "cases_traced": len(traced),
            "truth_ever_listed": round(len(listed) / len(traced), 4),
            "truth_in_final_differential": round(sum(1 for _, t in traced if t.get("in_final")) / len(traced), 4),
            "truth_led_at_some_turn": round(sum(1 for t in listed if t.get("led_at_some_turn")) / len(traced), 4),
            "entry_route": dict(routes.most_common()),
            "generator_recall": round(routes.get("generator", 0) / len(traced), 4),
            "reverse_index_rescues": routes.get("reverse_index", 0),
            "first_turn_listed_median": statistics.median(t["first_turn"] for t in listed) if listed else None,
            "accuracy_by_entry_route": {k: round(sum(bool(r.get("top1_correct")) for r, t in traced if t.get("route") == k)
                                                / max(1, sum(1 for _, t in traced if t.get("route") == k)), 4) for k in routes},
            "list_size_mean": _mean([r.get("n_hypotheses") or 0 for r in ok]),
        }
    M["failure_taxonomy"] = dict(Counter(r["_fail"] for r in ok).most_common())

    # -- belief dynamics: leader flips, turns spent after the truth first led (commit-threshold and stall tuning)
    flips, after_lead = [], []
    for r in ok:
        leaders = [max(t["probs"], key=t["probs"].get) for t in (r.get("hypothesis_trace") or []) if t.get("probs")]
        if leaders:
            flips.append(sum(1 for a, b in zip(leaders, leaders[1:]) if a != b))
        fl = (r.get("_trace") or {}).get("first_lead_turn")
        if fl is not None and r.get("top1_correct"):
            after_lead.append((r.get("turns") or 0) - fl)
    M["belief_dynamics"] = {
        "leader_flips_per_case": _mean(flips),
        "cases_with_3plus_flips": sum(1 for f in flips if f >= 3),
        "turns_after_truth_first_led_mean": _mean(after_lead),
        "turns_after_truth_first_led_median": statistics.median(after_lead) if after_lead else None,
    }

    # -- precision of hypothesis sources: of the diseases each route added, how many were the truth
    added: Counter = Counter()
    hits: Counter = Counter()
    for r in ok:
        tid = _truth_id(r, store)
        first_route: dict[str, str] = {}
        for t in r.get("hypothesis_trace") or []:
            for did, src in (t.get("ids") or {}).items():
                first_route.setdefault(did, _route(src))
        for did, route in first_route.items():
            added[route] += 1
            hits[route] += int(bool(tid) and did == tid)
    if added:
        M["hypothesis_sources"] = {route: {"added_per_case": round(c / n, 2), "precision": round(hits[route] / c, 4)}
                                   for route, c in added.most_common()}

    # -- prior (P1): the bucket the truth was given, and accuracy by it
    tb = []
    for r in ok:
        tid = _truth_id(r, store)
        entries = [e for e in (r.get("prior_log") or []) if e.get("disease_id") == tid]
        if tid and entries:
            tb.append((entries[-1].get("final") or "unknown", bool(r.get("top1_correct"))))
    if tb:
        M["truth_prior_bucket"] = {b: {"n": sum(1 for x, _ in tb if x == b), "top1": round(sum(c for x, c in tb if x == b) / sum(1 for x, _ in tb if x == b), 4)}
                                   for b in sorted({x for x, _ in tb})}

    # -- perceive (A2 lexicon hit rate on real replies, negation, coined codes, Korean)
    logs = [e for r in ok for e in (r.get("perceive_log") or [])]
    if logs:
        ext = sum(e.get("extracted", 0) for e in logs) or 1
        M["perceive"] = {
            "replies": len(logs), "codes_per_reply": round(ext / len(logs), 2),
            "lexicon_hit_rate": round(sum(e.get("from_lexicon", 0) for e in logs) / ext, 4),
            "from_cards_only_rate": round(sum(e.get("from_cards_only", 0) for e in logs) / ext, 4),
            "unoffered_code_rate": round(sum(e.get("unoffered", 0) for e in logs) / ext, 4),
            "negated_share": round(sum(e.get("negated", 0) for e in logs) / ext, 4),
            "vital_rule_findings": sum(e.get("vital_rule", 0) for e in logs),
            "test_results_coded": sum(e.get("tests", 0) for e in logs),
            "replies_with_hangul": sum(1 for e in logs if e.get("hangul")),
            "replies_with_no_codes": round(sum(1 for e in logs if not e.get("extracted") and not e.get("tests")) / len(logs), 4),
            "by_channel": {ch: {"replies": len(es), "codes_per_reply": round(sum(e.get("extracted", 0) for e in es) / len(es), 2)}
                           for ch, es in _group(logs, "channel").items()},
        }

    # -- belief and calibration by evidence volume (design report: calibration by number of findings, by list size)
    pairs_f = [(float(r.get("n_findings") or 0), bool(r.get("top1_correct"))) for r in ok if r.get("n_findings") is not None]
    if pairs_f:
        conf_by = {}
        for lo, hi in ((0, 11), (11, 21), (21, 41), (41, 10_000)):
            sel = [(float(r["top1_prob"]), bool(r.get("top1_correct"))) for r in ok
                   if r.get("n_findings") is not None and lo <= r["n_findings"] < hi and r.get("top1_prob") is not None]
            if sel:
                conf_by[f"{lo}-{hi - 1 if hi < 10_000 else '+'}"] = {"n": len(sel), "top1": round(sum(c for _, c in sel) / len(sel), 4),
                                                                     "mean_conf": round(sum(p for p, _ in sel) / len(sel), 4),
                                                                     "ece": round(_ece(sel), 4)}
        M["calibration_by_findings"] = conf_by
    tops = [(float(r["top1_prob"]), bool(r.get("top1_correct"))) for r in ok if r.get("top1_prob") is not None]
    M["commit_confidence_buckets"] = _bins(tops, [0.0, 0.4, 0.6, 0.8, 0.9, 0.95, 1.01],
                                           lambda lo, hi: f"{lo:.2f}-{min(hi, 1.0):.2f}")  # decision 12: is 0.90 worth it?
    M["other_mass_at_commit_mean"] = _mean([r.get("other_prob") or 0.0 for r in ok])

    # -- planner
    acts = [a for r in ok for a in (r.get("actions_taken") or [])]
    if acts:
        by_type = Counter(a.get("type") for a in acts)
        tests = [t for r in ok for t in (r.get("test_results") or [])]
        correct = sum(bool(r.get("top1_correct")) for r in ok)
        M["planner"] = {
            "turns_with_actions_per_case": round(len(acts) / n, 2),
            "action_turns_by_type": dict(by_type.most_common()),
            "actions_per_case_by_type": {k: round(v / n, 2) for k, v in by_type.items()},
            "free_questions_per_case": round(sum(1 for a in acts if a.get("free")) / n, 3),
            "test_results_per_case": round(len(tests) / n, 2),
            "test_yield_abnormal": round(sum(1 for t in tests if t.get("outcome") not in ("normal", "negative", "no_embolus")) / max(1, len(tests)), 4),
            "cost_per_correct": round(sum(r.get("cost_spent", 0.0) for r in ok) / max(1, correct), 1),
            "cost_per_case_by_type": {k: round(sum(a.get("cost", 0.0) for a in acts if a.get("type") == k) / n, 1) for k in by_type},
            "stall_commits": sum(1 for r in ok if (r.get("commit_reason") or "").startswith("stalled")),
            "unresolved_objections": sum(1 for r in ok if r.get("unresolved_objection")),
        }

    # -- critic veto (D2) and report dissent (E1), with judged alternatives
    vetoes = [v for r in ok for v in (r.get("vetoes") or [])]
    alt = lambda r, name: (r.get("alt_judgments") or {}).get(name)  # noqa: E731
    if vetoes:
        honored = [v for v in vetoes if v.get("honored")]
        judged = [alt(r, str((v.get("sig") or ["", ""])[1]).replace("_", " ")) for r in ok for v in (r.get("vetoes") or []) if v.get("honored")]
        judged = [j for j in judged if j is not None]
        M["critic_veto"] = {"vetoes": len(vetoes), "cases": sum(1 for r in ok if r.get("vetoes")), "honored": len(honored),
                            "why": dict(Counter(v.get("why") for v in vetoes).most_common()),
                            "honored_alternative_correct": round(sum(judged) / len(judged), 4) if judged else None,
                            "accuracy_in_veto_cases": round(sum(bool(r.get("top1_correct")) for r in ok if r.get("vetoes"))
                                                            / max(1, sum(1 for r in ok if r.get("vetoes"))), 4)}
    dis = [(r, d) for r in ok for d in (r.get("report_dissent") or [])]
    llm_right = []
    for r in ok:
        name = r.get("llm_final_diagnosis") or ""
        if not name:
            continue
        j = alt(r, name)
        llm_right.append(bool(r.get("top1_correct")) if j is None else j)
    M["report"] = {
        "dissent_cases": sum(1 for r in ok if r.get("report_dissent")),
        "dissent_outcomes": dict(Counter(d.get("outcome") for _, d in dis).most_common()),
        "dissent_correct": (round(sum(1 for r, d in dis if alt(r, d.get("dissent", ""))) / len(dis), 4) if dis else None),
        "accuracy_in_dissent_cases": (round(sum(bool(r.get("top1_correct")) for r in ok if r.get("report_dissent"))
                                            / max(1, sum(1 for r in ok if r.get("report_dissent"))), 4) if dis else None),
        # E1 counterfactual: accuracy had the report LLM's own diagnosis been the answer
        "top1_if_llm_name_won": round(sum(llm_right) / len(llm_right), 4) if llm_right else None,
        "top1_computed_same_cases": round(sum(bool(r.get("top1_correct")) for r in ok if r.get("llm_final_diagnosis")) / max(1, len(llm_right)), 4) if llm_right else None,
    }

    # -- contrast
    cl = [e for r in ok for e in (r.get("contrast_log") or [])]
    M["contrast"] = {"turns_with_pairs": len(cl), "cards_added_turns": sum(1 for e in cl if e.get("added")),
                     "cases_with_contrast": sum(1 for r in ok if any(e.get("added") for e in (r.get("contrast_log") or [])))}

    # -- safety: can't-miss recall and premature closure (design report section 7)
    cm_truth = []
    for r in ok:
        tid = _truth_id(r, store)
        card = store.get(tid) if (store is not None and tid) else None
        if card is not None and card.cant_miss:
            cm_truth.append(r)
    premature = [r for r in ok if any(d.get("cant_miss") and d.get("prob", 0) >= 0.03 for d in (r.get("differential_probs") or [])[1:])]
    M["safety"] = {
        "cant_miss_truth_cases": len(cm_truth),
        "cant_miss_recall_in_final_differential": (round(sum(1 for r in cm_truth if _truth_id(r, store) in _final_ids(r)) / len(cm_truth), 4)
                                                   if cm_truth else None),
        "cant_miss_top1": round(sum(bool(r.get("top1_correct")) for r in cm_truth) / len(cm_truth), 4) if cm_truth else None,
        "commits_with_cant_miss_alternative_ge_0.03": len(premature),
        "accuracy_in_those_commits": round(sum(bool(r.get("top1_correct")) for r in premature) / len(premature), 4) if premature else None,
    }

    # -- knowledge provenance behind the leader (rationale traceability)
    mix = {"correct": Counter(), "wrong": Counter()}
    for r in ok:
        finals = _final_ids(r)
        rows_ev = (r.get("evidence") or {}).get(finals[0], []) if finals else []
        for e in rows_ev:
            key = e.get("via", "none")
            if key == "own":
                key = f"own:{e.get('status', '?')}" + (f":tier{e['tier']}" if e.get("tier") else "")
            mix["correct" if r.get("top1_correct") else "wrong"][key] += abs(e.get("llr", 0.0))
    M["leader_evidence_sources"] = {k: {s: round(v / (sum(c.values()) or 1), 4) for s, v in c.most_common()} for k, c in mix.items() if c}

    # -- prior context (v0.3.1 P1) diagnostics carried over from summary.json are in legacy_summary
    return M


def _group(items: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for e in items:
        out.setdefault(str(e.get(key)), []).append(e)
    return out


def run_report(run_dir: Path, store=None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    rows, summary, meta = _load(run_dir)
    return {
        "run": run_dir.name,
        "meta": {**meta, "agent": summary.get("agent", meta.get("agent")), "dataset": summary.get("dataset", meta.get("dataset")),
                 "doctor_model": summary.get("doctor_model"), "patient_model": summary.get("patient_model")},
        "metrics": metrics(rows),
        "usage": usage_summary(rows),
        "knowledge": knowledge_summary(rows, store),
        "modules": module_metrics(rows, store),
        "prior_context_p1": {k: v for k, v in summary.items() if k.startswith(("prior_", "complaint_", "context_"))},
        "list_effect": {k: summary.get(k) for k in ("list_effect", "calibration_by_list_length", "brier_full") if k in summary},
        "leakage": {k: summary.get(k) for k in ("leak_flags", "records_mentions") if k in summary},
        "legacy_summary": summary,
        "cases": case_rows(rows),  # after module_metrics, which annotates each row's truth route and failure class
    }


def _paired(a: dict[int, bool], b: dict[int, bool], reps: int = 2000) -> dict[str, Any]:
    ks = sorted(set(a) & set(b))
    if not ks:
        return {"n": 0}
    diffs = [float(b[k]) - float(a[k]) for k in ks]
    rng = random.Random(7)
    boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(reps))
    b01 = sum(1 for k in ks if not a[k] and b[k])
    b10 = sum(1 for k in ks if a[k] and not b[k])
    m = b01 + b10
    p = 1.0 if m == 0 else min(1.0, 2 * sum(math.comb(m, i) for i in range(min(b01, b10) + 1)) / 2 ** m)
    return {"n": len(ks), "top1_a": round(sum(a[k] for k in ks) / len(ks), 4), "top1_b": round(sum(b[k] for k in ks) / len(ks), 4),
            "diff": round(sum(diffs) / len(diffs), 4), "ci95": [round(boots[int(0.025 * reps)], 4), round(boots[int(0.975 * reps)], 4)],
            "b_only_correct": b01, "a_only_correct": b10, "mcnemar_p": round(p, 5)}


def bundle(run_dirs: list[Path], out: Path, store=None) -> dict[str, Any]:
    runs = [run_report(d, store) for d in run_dirs]
    comps = []
    for i, ra in enumerate(runs):
        for rb in runs[i + 1:]:
            if ra["meta"].get("dataset") != rb["meta"].get("dataset"):
                continue
            ca = {c["index"]: bool(c["correct"]) for c in ra["cases"] if not c["error"]}
            cb = {c["index"]: bool(c["correct"]) for c in rb["cases"] if not c["error"]}
            res = _paired(ca, cb)
            refs = {c["index"]: (c["reference"], c["final"]) for c in ra["cases"]}
            fin_b = {c["index"]: c["final"] for c in rb["cases"]}
            res["discordant"] = [{"index": k, "reference": refs[k][0], "a": refs[k][1], "b": fin_b.get(k), "b_correct": cb[k]}
                                 for k in sorted(set(ca) & set(cb)) if ca[k] != cb[k]]
            comps.append({"a": ra["run"], "b": rb["run"], **res})
    data = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "dxagent_version": __version__,
            "dxagent_commit": _git_commit(Path(__file__).resolve().parents[2]), "runs": runs, "comparisons": comps}
    out = Path(out)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return data


def settings_snapshot(settings) -> dict[str, Any]:
    d = asdict(settings) if is_dataclass(settings) else dict(vars(settings))
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()
            if isinstance(v, (str, int, float, bool, type(None), Path))}
