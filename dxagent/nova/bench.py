"""Mini-bench: play case files against the rule-enforcing simulator, score them, summarize.

    python run.py --bench dxagent/nova/cases --offline              plumbing, no model
    python run.py --bench my_cases/ --out runs/A                     live (NOVA_API_BASE ...), keyword patient
    python run.py --bench my_cases/ --sim-llm --judge --out runs/B   model-played patient + LLM rubric judge

Writes <out>/<case_id>.json (record, transcript, violations, scores) and <out>/summary.json (means, per-case rows).
Compare two versions on the SAME cases (paired); with 5-20 cases only large differences are real.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Optional

from dxagent.nova.agent import NovaAgent
from dxagent.nova.budget import Budget
from dxagent.nova.config import NovaConfig
from dxagent.nova.env import LocalSimEnv
from dxagent.nova.judge import llm_judge, proxy_score


def load_cases(paths: list[str]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for p in paths:
        q = Path(p)
        files += sorted(q.glob("*.json")) if q.is_dir() else [q]
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def _transcript(env: LocalSimEnv, opening: str) -> str:
    lines = [f"T0 환자: {opening}"]
    for a in env.actions:
        if a["accepted"]:
            lines.append(f"T{a['turn']} 의사[{a['kind']}]: {a['text']}\n    -> {a['reply']}")
        else:
            lines.append(f"(거절됨) 의사[{a['kind']}]: {a['text']} ({a.get('why', '')})")
    return "\n".join(lines)


def run_bench(paths: list[str], cfg: Optional[NovaConfig] = None, offline: bool = False, out: str = "runs/nova_bench",
              sim_llm: bool = False, judge: bool = False) -> dict[str, Any]:
    cfg = cfg or NovaConfig.from_env()
    agent = NovaAgent(cfg, offline=offline)
    aux = None
    if (sim_llm or judge) and not offline:
        from dxagent.nova.llm import NovaLLM

        aux_cfg = NovaConfig.from_env(max_calls=10**6, max_input_tokens=10**12, max_output_tokens=10**12)
        aux = NovaLLM(aux_cfg, budget=Budget(aux_cfg), name="bench-aux")
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in load_cases(paths):
        env = LocalSimEnv(case, cfg, patient_llm=aux if sim_llm else None)
        rec = agent.run(env)
        proxy = proxy_score(case, rec, env.violations)
        transcript = _transcript(env, (case.get("opening") or {}).get("text", ""))
        lj = None
        if judge and aux is not None:
            usage = {"turns": rec["turns"], "llm_calls": rec["budget"]["calls"],
                     "input_tokens": rec["budget"]["input_tokens"], "output_tokens": rec["budget"]["output_tokens"]}
            lj = llm_judge(aux, case, transcript, (rec.get("soap") or {}).get("text", ""), rec["diagnosis"], usage)
        row = {"case_id": case.get("case_id"), "reference": case.get("diagnosis"), "diagnosis": rec["diagnosis"],
               "final_source": rec["final"].get("source"), "turns": rec["turns"], "llm_calls": rec["budget"]["calls"],
               "input_tokens": rec["budget"]["input_tokens"], "output_tokens": rec["budget"]["output_tokens"],
               "seconds": rec["budget"]["seconds"], "violations": len(env.violations), "errors": len(rec["errors"]),
               "proxy_total": proxy["total"], "judge_total": (lj or {}).get("total")}
        rows.append(row)
        (outdir / f"{case.get('case_id', 'case')}.json").write_text(json.dumps(
            {"row": row, "proxy": proxy, "judge": lj, "violations": env.violations, "transcript": transcript,
             "record": rec}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    def mean(k: str) -> Optional[float]:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        return round(statistics.mean(vals), 2) if vals else None

    summary = {"n": len(rows), "round": cfg.round, "profile": cfg.profile, "offline": offline,
               "mean": {k: mean(k) for k in ("proxy_total", "judge_total", "turns", "llm_calls", "input_tokens",
                                             "output_tokens", "seconds", "violations", "errors")},
               "rows": rows}
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary

