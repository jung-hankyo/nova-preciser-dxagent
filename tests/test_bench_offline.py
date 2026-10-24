"""The benchmark runner works end to end on real AgentClinic cases with offline stand-ins."""

import json
from pathlib import Path

from dxagent.bench.runner import BenchModels, run_benchmark
from dxagent.env.agentclinic import load_cases
from dxagent.knowledge.cards import CardStore
from dxagent.offline import make_offline_llm, make_offline_simulator
from dxagent.runtime import Runtime

FIXTURE = Path(__file__).parent / "fixtures" / "agentclinic_sample.jsonl"


def test_load_cases():
    cases = load_cases(FIXTURE)
    assert len(cases) == 3 and cases[0].diagnosis == "Myasthenia gravis"
    assert "Objective" not in cases[0].objective


def test_runner_writes_results(settings, tmp_path):
    settings.max_turns = 8
    store = CardStore(settings.registry_dir, None, card_dirs=[settings.fixture_dir])
    doctor = make_offline_llm(store)
    rt = Runtime.create(settings, doctor=doctor, store=store)
    sim = make_offline_simulator()
    models = BenchModels(patient=sim, measurement=sim, judge=doctor)
    summary = run_benchmark(settings, data_path=FIXTURE, rt=rt, models=models, out_dir=tmp_path, printer=lambda *_: None)
    assert summary["n"] == 3 and summary["errors"] == 0
    rows = [json.loads(l) for l in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert {"reference", "final_diagnosis", "turns", "transcript", "top1_correct"} <= set(rows[0])
    assert (tmp_path / "summary.json").exists()


def test_bench_writes_meta_and_report_and_bundles_runs(tmp_path, settings):
    from dxagent.bench.report import bundle
    from dxagent.bench.runner import BenchModels, run_benchmark
    from dxagent.cli import _build_runtime
    from dxagent.offline import make_offline_simulator

    class A:
        offline, max_turns = True, None

    dirs = []
    for name in ("a", "b"):
        rt = _build_runtime(A(), settings)
        sim = make_offline_simulator()
        out = tmp_path / name
        run_benchmark(settings, n=2, out_dir=out, rt=rt, models=BenchModels(patient=sim, measurement=sim, judge=rt.doctor),
                      printer=lambda *a: None)
        assert {"results.jsonl", "summary.json", "run_meta.json", "report.json"} <= {p.name for p in out.iterdir()}
        dirs.append(out)
    data = bundle(dirs, tmp_path / "bundle.json", rt.store)
    assert len(data["runs"]) == 2 and data["comparisons"][0]["n"] == 2
    run = data["runs"][0]
    assert {"meta", "metrics", "usage", "knowledge", "cases"} <= set(run)
    assert run["meta"]["settings"]["local_fallback"] is False and "models" in run["meta"]
    assert all("differential" in c and "truth" in c for c in run["cases"])


def test_module_metrics_trace_the_truth_and_classify_failures():
    from dxagent.bench.report import module_metrics

    def row(idx, correct, final_ids, trace_ids, truth="t", n_findings=12, prob=0.85, **kw):
        return {"index": idx, "reference": "Truth", "top1_correct": correct, "match_rank": 1 if correct else None,
                "top1_prob": prob, "turns": 6, "cost_spent": 100.0, "n_findings": n_findings, "n_hypotheses": 6,
                "truth": {"status": "compiled", "id": truth},
                "differential_probs": [{"disease_id": d, "prob": 0.5, "cant_miss": False, "card_status": "compiled"} for d in final_ids],
                "hypothesis_trace": [{"turn": i, "ids": ids, "probs": {k: 0.5 for k in ids}} for i, ids in enumerate(trace_ids)],
                "perceive_log": [{"channel": "history", "extracted": 4, "from_lexicon": 3, "from_cards_only": 1, "unoffered": 0, "negated": 1}],
                "actions_taken": [{"turn": 3, "type": "lab", "ids": ["lab_x"], "cost": 40.0}],
                "test_results": [{"test_id": "lab_x", "outcome": "elevated"}],
                "llm_final_diagnosis": kw.get("llm", ""), "alt_judgments": kw.get("alt", {}), **{k: v for k, v in kw.items() if k not in ("llm", "alt")}}

    rows = [
        row(0, True, ["t", "x"], [{"t": "generator:common"}]),
        row(1, False, ["x", "t"], [{"x": "generator:common"}, {"x": "generator", "t": "reverse_index:rare"}]),
        row(2, False, ["x"], [{"x": "generator:common"}]),
        row(3, False, ["x"], [{"x": "generator", "t": "critic:uncommon"}, {"x": "generator"}], llm="Truth", alt={"Truth": True}),
    ]
    M = module_metrics(rows, None)
    assert M["failure_taxonomy"] == {"correct": 1, "outranked:has_knowledge": 1, "never_listed:has_knowledge": 1, "dropped": 1}
    h = M["hypotheses"]
    assert h["truth_ever_listed"] == 0.75 and h["entry_route"] == {"generator": 1, "reverse_index": 1, "critic": 1}
    assert h["reverse_index_rescues"] == 1 and h["generator_recall"] == 0.25
    assert M["perceive"]["lexicon_hit_rate"] == 0.75 and M["planner"]["test_yield_abnormal"] == 1.0
    assert M["report"]["top1_if_llm_name_won"] == 1.0


def test_new_module_metrics_dynamics_sources_cost():
    from dxagent.bench.report import module_metrics, usage_summary

    trace = [{"turn": 0, "ids": {"x": "generator:common"}, "probs": {"x": 0.6}},
             {"turn": 1, "ids": {"x": "generator", "t": "reverse_index:rare"}, "probs": {"x": 0.4, "t": 0.5}},
             {"turn": 2, "ids": {"x": "generator", "t": "reverse_index"}, "probs": {"x": 0.1, "t": 0.85}}]
    r = {"index": 0, "reference": "T", "top1_correct": True, "match_rank": 1, "top1_prob": 0.85, "turns": 5,
         "cost_spent": 50.0, "n_findings": 8, "truth": {"status": "compiled", "id": "t"},
         "differential_probs": [{"disease_id": "t", "prob": 0.85, "cant_miss": False}],
         "hypothesis_trace": trace, "prior_log": [{"disease_id": "t", "final": "rare"}],
         "llm_usage": {"by_role": {"doctor": {"calls": 10, "errors": 1, "input_tokens": 40000, "output_tokens": 4000}},
                       "by_model": {"deepseek:deepseek-v4-pro?thinking=low": {"calls": 10, "input_tokens": 40000, "output_tokens": 4000}},
                       "total": {"calls": 10}}}
    M = module_metrics([r], None)
    assert M["belief_dynamics"]["leader_flips_per_case"] == 1 and M["belief_dynamics"]["turns_after_truth_first_led_mean"] == 4
    assert M["hypothesis_sources"]["reverse_index"]["precision"] == 1.0 and M["hypothesis_sources"]["generator"]["precision"] == 0.0
    assert M["truth_prior_bucket"] == {"rare": {"n": 1, "top1": 1.0}}
    U = usage_summary([r])
    assert U["error_rate_by_role"]["doctor"] == 0.1
    assert abs(U["est_cost_usd_per_case"] - (40000 * 1.32 + 4000 * 3.96) / 1e6) < 1e-6
