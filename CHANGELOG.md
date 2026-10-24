# Changelog

Newest first. Decision numbers refer to `docs/decisions/`; test counts are the full suite at release (the runtime's share
in parentheses where given). Each branch's changelog stops at its own version.

## 0.4.2 (2026-10-02, `main`)

DeepSeek (OpenAI-compatible endpoint, thinking mode per spec) and OpenAI reasoning models; default profile DeepSeek agent
with OpenAI environment and judge; the usage ledger records cached and reasoning tokens and the report estimates cost per
case. 102 tests.

## 0.4.1 (2026-10-01)

Local-document retrieval as the primary fallback for a disease with no card; `docs/SOURCES.md` and the manifest template;
live-benchmark reporting (`dxagent bench-report`). 96 tests.

## 0.4.0 (2026-10-01)

The knowledge compiler. Decisions 29-35. Cards are compiled offline from a source manifest (the LLM copies verbatim
quotes, code verifies and computes, confidence comes from a fixed tier table); MONDO and HPO ontology spine; HPO
annotations give about 9,400 rare-disease cards without an LLM; card-less diseases inherit their nearest carded ancestor.
Removed: LLM-only card drafting. Held-out simulation: 0.938 top-1 with full coverage; the result is set by coverage and
the share of entries sources state. 91 tests.

## 0.3.1 (2026-10-01)

Prior context (decisions 20-22): the prior conditions only on age, sex, setting and the primary complaint through an
information-hidden assessor; a deterministic primary-complaint rule; context enters as evidence. Neutral in simulation
(+5.5 calls per case), adopted for the structural guarantee. 78 tests.

## 0.3.0

Six fixes, measured together on held-out worlds (top-1 0.887 to 0.930, Brier -0.047, ECE -0.026, 0.46 fewer turns, $27
less test cost per case): the primary complaint alone conditions the prior (+1.4 pp); a deterministic lexicon and
canonical finding codes (+0.7 pp); card prevalence for the reverse index (+1.0 pp); a critic's veto must name an
alternative; the report LLM's dissent is an objection (+3.0 pp over letting its name win).

## 0.2.1

The reverse index: known diseases the LLM never proposed join the differential when the evidence makes them at least 5%
likely. Top-1 +24 / +15 / +4 pp at LLM recall 0.6 / 0.75 / 0.9.

## 0.2.0

Frequencies replace stored likelihood ratios: cards hold `P(finding | disease)` and `P(test result | disease)`, ratios are
formed between the live diseases at run time, tests are categorical, and "A vs B" contrast cards are elicited when two
diseases are both live. Top-1 +0.8 to +2.0 pp and 26-28% less test cost than 0.1.

## 0.1.0

The first agent: a LangGraph interview loop (perceive, hypothesize, update belief, critic, plan, interview, report) with a
Bayesian belief update and a value-of-information planner, and likelihood ratios stored in its knowledge cards.
