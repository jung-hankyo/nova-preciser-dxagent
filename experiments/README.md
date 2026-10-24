# Experiments

Simulation studies of the reasoning core. Raw paired analyses are in `results/`; `patches/` holds two scenario patches
for the v0.3.1 harness.

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
