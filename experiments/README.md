# Experiments

Simulation studies of the research core (`docs/CORE.md`) and the local checks of the contest runtime. Raw paired analyses
are in `results/`; `patches/` holds two scenario patches for the v0.3.1 harness.

## Simulation studies of the core

All are simulation only: a hidden world model generates the patients, LLM roles are scripted stand-ins whose
assumptions are stated in each script's docstring, and every comparison is paired (identical pre-sampled answers) with
95% bootstrap CIs and exact McNemar tests. Variants were chosen on development worlds 11-13 and confirmed once on
held-out worlds 21-30 (5,000 cases). The acceptance rule, fixed beforehand: better accuracy or calibration, no metric
significantly worse. The worlds are easier and more evidence-rich than real cases, and the synthetic families share no
findings, so absolute numbers do not transfer.

| Study (script) | Question | Headline result |
|---|---|---|
| v0.1 vs v0.2 (`lr_vs_freq.py`) | Stored likelihood ratios vs per-disease frequencies | v0.2 +0.8 / +2.0 / +1.9 pp top-1 at knowledge noise x1 / x2 / x3, 26-28% less test cost, fewer turns. Representation drives accuracy and calibration, the pairwise objective drives cost. Against LRs measured in a mismatched comparison group, +2.7 to +8.5 pp. The can't-miss commit rule changed nothing; the report-noise patch (v0.2.1) doubled ECE and was not shipped |
| Mechanisms, harm term, drafting (`world_bench.py`) | v0.2 against MAI-DxO-, GraphDx- and AMIE-style mechanisms; harm weighting; live drafting | A MAI-DxO-style planner matches v0.2 accuracy (+0.6 pp) at +52% test cost; the others are 13-20 pp lower. The harm term cuts missed can't-miss diagnoses 5.8-7.0 pp for $56-142 more per case. Live drafting reaches 92-99% of oracle-card accuracy on unknown diseases (about +15 pp overall when 20% are unknown); generator recall sets the ceiling |
| Unified observation likelihood (`obs_model.py`) | One report likelihood for belief and planner | **Rejected:** top-1 +0.8 to +1.1 pp, but ECE +0.03 to +0.06 and more turns and cost; raising the commit threshold to 0.90 is the cheaper route (exploratory) |
| Card-based hypothesis sources (`hyp_recall.py`) | Does a reverse index fix the LLM's recall ceiling? | Yes: top-1 +24 / +15 / +4 pp at LLM recall 0.6 / 0.75 / 0.9 (at recall 0.6 the truth is in the list at commit 0.70 to 0.997), cheaper at low recall, +$18 and ECE +0.015 at 0.9. Adopted (v0.2.1); a mimic source adds nothing on top |
| v0.3 fixes (`v03_sim.py`, `v03_batch.py`, `v03_pairs.py`, `v03_analyze.py`, `v03_lexicon_eval.py`) | Primary-complaint prior (A1), lexical retrieval and canonical codes (A2, A3), card prevalence (C3), structured veto (D2), report dissent as objection (E1) | Held-out, all together: top-1 0.887 to 0.930, Brier -0.047, ECE -0.026, turns -0.46, cost -$27. A1 +1.4 pp; A2+A3 +0.7 pp (ECE +0.006, accepted); C3 +1.0 pp (missed can't-miss +2.2 pp); E1 +3.0 pp over letting the report LLM's name win; D2 contains a stubborn critic (-$96 per case at -0.3 pp). Retrieval offers the right code in 79% of held-out paraphrases. Screened out: background-rate pass, evidence-ratio admission, discovery prompts, open-world re-prompt, conditional batches, shadow list |
| v0.3.1 prior context (`p1_analyze.py`) | Does conditioning the prior on S only help? | Neutral on top-1, Brier, ECE and log loss when the generator obeys (+5.5 calls per case); slightly worse against a fully leaking one (Brier +0.004, +$11). The prior decides the leader in under 2% of cases and the agent is underconfident (0.900 vs 0.930), so double counting is nearly harmless here. Adopted for the structural guarantee (decision 20); `results/p1/`. The harness was made deterministic (serial thread pools) |
| v0.4 compiled knowledge (`v04_sim.py`, `v04_analyze.py`) | Compiled cards on the v0.3 held-out benchmark | The v0.4 code with v0.3's knowledge reproduces v0.3.1 in 5,000/5,000 cases, so every difference is knowledge: compiled 75% coverage 0.782 (0.830 with the fallback); full coverage 0.890 / 0.913 / 0.938 when 70% / 85% / 100% of entries are stated by a source; against live drafting with systematic errors, +4.7 pp with 2.8 pp fewer confident errors. Performance is set by coverage and the stated share. `results/v04_heldout/` |
| N.O.V.A protocol cost (`v05_protocol_sim.py`) | What the contest's action rules cost the unchanged core | One action per turn -0.9 pp top-1 (n.s.); no tests -18.9 pp (decisions 38-49 in `docs/decisions/v0.5-simple-approach.md`). `results/v05_protocol/` |

Commands (outputs go under `runs/`, git-ignored):

```bash
python experiments/lr_vs_freq.py --cases 1000 --noise 1 2 3 --stage simulate
python experiments/world_bench.py --stage arch --worlds 3 --cases 1500 --knowledge-sd 0.5 --out runs/wb   # also drafting, analyze
python experiments/v03_sim.py --arms v0.2 v0.3 no-A1 --seeds 21 22 23 24 25 26 27 28 29 30 --cases 500 --out DIR
python experiments/v03_pairs.py DIR OUT.json "v0.2:v0.3@default" "no-A1:v0.3@default"
python experiments/v03_sim.py --arms v0.3 v0.3+P1 --seeds 21 ... 30 --cases 500 --set LEAK=1.0 --out runs/p1_heldout
python experiments/p1_analyze.py runs/p1_heldout --base v0.3 --arms v0.3+P1 --tags LEAK=0.0 LEAK=1.0
python experiments/v04_sim.py --arms v0.4 v0.4+fallback v0.4+cov100 v0.4+v3know --out V04
python experiments/v04_analyze.py BASE V04 OUT.json
```

## Contest runtime: local CPX-style cases

`data/nova_cpx/*.yaml` holds 65 practice cases written for this project in the development kit's case format (opening,
vital signs, patient script, exam table), each with our own answer key: history items, matched by regex against the
doctor's SAY lines, and the exam rows that matter. L01-L18 (`cases_a`, `cases_b`) are the original set, M01-M37 (`cases_c`
to `cases_f`) cover the other presentations (56 of the 58 playbooks are exercised; `short_stature` and `weight_loss` by
the kit's sample cases), and S01-S10 (`cases_g`) are special situations: uncovered complaints, a counselling request, a
minor, a guardian answering, high suicide risk, an emergency or a red flag behind a common complaint. Each case names
the playbook its opening must route to; `case_check.py` validates the files. The same cases drive the simulated contest
in `sim/` (`docs/SUBMISSION.md`). `nova_cpx_check.py` writes them as `case.json` folders and plays them through the
development kit's simulator with the scripted patient and examiner of `nova_devkit_check.py`. The agent runs offline, so
the scores are the visit's deterministic floor (the focus call and the time-out add nothing). The cases and the playbooks
have the same author, so the gains partly measure agreement with that author's idea of each checklist.

```bash
NOVA_OFFLINE=1 python -B experiments/nova_cpx_check.py --devkit <dev kit> --label mylabel   # all 65 cases
python experiments/case_check.py experiments/data/nova_cpx/cases_c.yaml                          # validate case files
NOVA_OFFLINE=1 python -B experiments/nova_devkit_check.py --devkit <dev kit> --samples <sample cases> --audit
python -B experiments/nova_live_report.py runs/live/<version>      # per-case reports from live-run logs (docs/LIVE_RUNS.md)
```

| Version | History coverage | Key exams | Unlisted maneuvers | Refused | Turns (range) |
|---|---|---|---|---|---|
| 0.5.2, no playbooks (fixture cards on: 0.56 / 0.41 / 32.9) | 0.55 (in-group 0.61, out 0.46) | 0.36 | 37 | - | 28.7 |
| 0.6.0, playbooks | 0.85 (0.84 / 0.86) | 0.86 | 16 | - | 44.9 |
| 0.7.0, cut at 40 | 0.82 | 0.89 | 15 | - | 39.3 |
| 0.7.1, 40 an aim, cap 46 | 0.85 | 0.89 | 15 | 0 | 42.2 (34-49) |
| 0.7.2, one-action maneuvers | 0.85 | 0.86 | 18 | 0 | 42.2 |
| 0.8.0, evidence review (and 0.8.1) | 0.84 | 0.83 | 23 | 0 | 42.5 (35-48) |
| 0.9.0-pre, the same 18 cases | 0.84 | 0.83 | 23 | 0 | 42.5 (35-48) |
| 0.9.0-pre, all 65 cases (L 0.84 / 0.83; M 0.79 / 0.74; S 0.54 / 0.38) | 0.76 | 0.71 | 55 | 0 | 42.2 (max 48) |

"Unlisted" is a maneuver answered from outside the case's table. The fixture cards did not improve history coverage on
unseen cases, and exam coverage collapsed outside the old complaint groups (about 0.1 against 0.5), which motivated the
playbooks. The 0.7.2 drop in key exams is the stand-in's: the cases' tables use the old maneuver names, which its bigram
matcher cannot map to the new single-action names. The 0.8.0 misses are items the audits demoted to priority 3 for low
yield and two maneuvers replaced on the evidence, which a live model can add back. The 40-turn cut lost 10 single history
items across the 18 cases and gained 3.
