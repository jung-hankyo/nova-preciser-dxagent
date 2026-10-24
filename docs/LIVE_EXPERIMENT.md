# Live baseline experiment (v0.4.2)

Goal: first live numbers on AgentClinic MedQA and the contribution of the knowledge layer. Everything needed for
analysis ends up in ONE file, `dxagent_live_summary.json`, built by `dxagent bench-report`.

## 0. Setup

```bash
pip install -e ".[dev,pdf]"
cp .env.example .env                   # API keys; models below
pytest -q                              # 102 tests, offline
```

| Role | Setting | Default | Why |
|---|---|---|---|
| Doctor: extraction, hypotheses, interview, critic, report; also the LLM-only baseline | `DX_DOCTOR_MODEL` | `deepseek:deepseek-v4-pro?thinking=low` | Generator recall and extraction accuracy are the binding constraints |
| Knowledge: compile extraction, contrast cards | `DX_CARD_MODEL` | `deepseek:deepseek-v4-pro` | Verbatim quotes and numbers need precision, not reasoning; compile runs once and is cached |
| Prior assessor | `DX_PRIOR_MODEL` | `deepseek:deepseek-flash` | Bucket choice from S only |
| Patient, measurement | `DX_PATIENT_MODEL`, `DX_MEASUREMENT_MODEL` | `openai:gpt-5-mini?effort=minimal` | Different family from the doctor (AgentClinic: the patient model affects scores) |
| Judge | `DX_JUDGE_MODEL` | `openai:gpt-5-mini?effort=medium` | Not the doctor's family; hand-check the discordant cases |

Keys: `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`. Leave `DX_TEMPERATURE` unset. DeepSeek bills half price off-peak (peak:
01:00-04:00 and 06:00-10:00 UTC on weekdays); the report's cost estimate uses peak rates. Before sending MIMIC-IV cases to
any external API, check PhysioNet's credentialed-data rules; AgentClinic-MedQA cases are public.

## 1. Knowledge base

```bash
dxagent kb fetch                       # MONDO, HPO, HPO annotations; indexes in kb/ontology
# put documents in sources/docs/ and list them in sources/manifest.yaml (docs/SOURCES.md, docs/sources_manifest.template.yaml)
dxagent kb sources --check             # every file present; records SHA-256
dxagent kb compile                     # LLM extraction of verbatim quotes (cached in kb/.extract_cache), then code verification
dxagent kb stats                       # cards, findings, build record
```

Read `kb/compile_report.json` (rejected claims and reasons) before running cases. Do not compile from the benchmark's
diagnosis list; the report measures coverage of the reference diagnoses separately.

## 2. Smoke test, then the arms

```bash
DX_CACHE_DIR=cache/smoke dxagent bench -n 3 --verbose --out runs/smoke     # no errors; usage.per_case_by_role filled
```

Each dx arm gets its own `DX_CACHE_DIR`, so on-demand cards compiled in one arm never leak into another. Keep models,
`--dataset` and case range identical across arms; every comparison is paired by case.

```bash
DX_CACHE_DIR=cache/A dxagent bench --workers 4 --out runs/A_dx_v042                         # as shipped
dxagent bench --agent baseline --workers 4 --out runs/C_llm_only                            # same LLM, no knowledge layer (the bar)
DX_CACHE_DIR=cache/B DX_LOCAL_FALLBACK=0 dxagent bench --workers 4 --out runs/B_dx_nolocal   # compiled kb only
```

Ablations, as budget allows (prefix `DX_CACHE_DIR=cache/<arm>`): `DX_LIVE_FALLBACK=1` (does the PubMed fallback add
anything after the local documents?); `DX_COMMIT_THRESHOLD=0.90` (better accuracy and calibration for the extra cost?);
`--objective clinical` (can't-miss recall against cost); `DX_CRITIC=0` (what the critic and vetoes contribute);
`DX_REVERSE_INDEX=0` (the reverse index on real generator recall); `DX_CONTRAST=0` (the one LLM-only knowledge path);
`--perturb 2` (robustness to knowledge error). MedQA has 107 cases, so one run's top-1 has a 95% CI of roughly +/- 9 pp;
paired comparisons are tighter; repeat the core arms with `--dataset medqa_extended` for more power. If a run stops,
rerun with `--start <index>` into a new directory and pass both to bench-report only if they do not overlap.

## 3. One summary file

```bash
dxagent bench-report runs/A_dx_v042 runs/B_dx_nolocal runs/C_llm_only --out dxagent_live_summary.json
```

Per run it contains: **meta** (version, commit, models, settings, knowledge-base build); **metrics** (top-1, top-3,
in-differential, Brier, ECE, calibration bins, confident errors, turns, test cost, seconds, commit reasons, dissent and
veto cases, errors); **usage** (calls, tokens including cached and reasoning, error rate per role and model, estimated
USD per case); **knowledge** (reference diagnosis resolution and card status, accuracy by status, coverage misses,
fallback outcomes); **modules** (hypothesis recall and entry routes, failure taxonomy of wrong cases, extraction
diagnostics, calibration by findings and by commit confidence, belief dynamics, hypothesis-source precision, accuracy by
prior bucket, planner behavior, critic veto, report dissent and its counterfactual, contrast cards, safety, the sources
of the leader's evidence); **prior_context_p1, list_effect, leakage** (the v0.3.1 diagnostics); and **cases** (one row
per case with its failure class; wrong cases carry a truncated transcript). `comparisons` holds every pair of runs on the
same dataset (McNemar, bootstrap CI, discordant cases). The report costs a few extra judge calls per case: each distinct
alternative named by a dissent, a veto or the report LLM is judged against the reference (at most 6 per case).
