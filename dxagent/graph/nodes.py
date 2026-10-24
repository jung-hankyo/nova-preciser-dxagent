"""Graph nodes. Each maps to a row of the node table in the design report (section 5).

    perceive        LLM            reply -> coded findings (numeric labs via card rules)
    hypothesize     LLM            findings -> candidate diseases with pretest-prior buckets (rescaled with the list; see belief.py)
    load_knowledge  deterministic  assign priors; check every hypothesis has knowledge (own card or carded ancestor);
                                   logs coverage misses; fallbacks: local documents (default on), then online
                                   PubMed (only if DX_LIVE_FALLBACK=1)
    update_belief   deterministic  per-disease frequencies vs background -> posterior differential;
                                   reverse index adds known diseases the evidence supports (>= 5% if added)
    contrast        LLM (cached)   for live confusable pairs without a contrast card: elicit "A vs B" knowledge
    critic          rules + LLM    premature-closure guard; may add hypotheses or veto commit
    plan            deterministic  intake script, then value-of-information action choice or commit
    interview       LLM            action(s) -> one conversational turn
    await_reply     interrupt      waits for the patient, the clinical team, or the simulator
    report          LLM            final diagnosis and rationale from the computed differential
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

from dxagent import usage
from typing import Any, Optional

from langgraph.types import interrupt

from dxagent.config import DEFAULT_ACTION_COST, PRIOR_BUCKET
from dxagent.knowledge.cards import norm_key, slug
from dxagent.prompts import (
    CRITIC_SYSTEM,
    DISCOVERY_SYSTEM,
    EXTRACT_SYSTEM,
    HYPOTHESIZE_PRIMARY_NOTE,
    OPEN_WORLD_NOTE,
    FREE_QUESTION_SYSTEM,
    HYPOTHESIZE_SYSTEM,
    INTAKE_ACTIONS,
    INTERVIEW_SYSTEM,
    REPORT_SYSTEM,
    critic_user,
    discovery_user,
    extract_user,
    free_question_user,
    hypothesize_user,
    interview_user,
    report_user,
)
from dxagent.knowledge.complaints import ComplaintRegistry
from dxagent.priors import PriorService, select_complaint, signature
from dxagent.reasoning.belief import (
    evidence_breakdown,
    apply_numeric_rules,
    apply_vital_rules,
    compute_posterior,
    context_coverage,
    prior_sensitivity,
    merge_findings,
    merge_tests,
    prune_hypotheses,
    reverse_index_candidates,
)
from dxagent.reasoning.planner import decide
from dxagent.reasoning.safety import should_call_critic
from dxagent.runtime import Runtime
from dxagent.schemas import (
    CriticOutput,
    ExtractionOutput,
    FreeQuestionOutput,
    HypothesisOutput,
    InterviewerOutput,
    ProposedHypothesis,
    ReportOutput,
)

log = logging.getLogger(__name__)

OPEN_HPI_FOLLOWUP = (
    "Thank you for telling me. Could you tell me more about it: when exactly it started, whether it came on "
    "suddenly or gradually, how it has changed since, and anything that makes it better or worse?"
)


class Nodes:
    def __init__(self, rt: Runtime):
        self.rt = rt
        self.s = rt.settings
        self.store = rt.store
        self.llm = rt.doctor
        # v0.3.1 (P1). The simulation harness passes a bare runtime, so both have defaults.
        self.priors = PriorService(getattr(rt, "prior_llm", None) or rt.doctor, self.store, self.s)
        self.complaints = getattr(rt, "complaints", None) or ComplaintRegistry(
            self.s.registry_dir, None, llm=rt.doctor, auto=self.s.dedup_auto, confirm=self.s.dedup_confirm)

    # ------------------------------------------------------------------ perceive

    def perceive(self, state: dict[str, Any]) -> dict[str, Any]:
        reply = (state.get("last_reply") or "").strip()
        if not reply:
            return {}
        turn = state.get("turn", 0)
        last_actions = state.get("last_actions", [])
        hyp_ids = [h["disease_id"] for h in state.get("hypotheses", [])]
        full_vocab = self.store.vocabulary()
        priority = [c for a in last_actions for c in a.get("reveals", [])]
        lex_codes: set[str] = set()
        if self.s.lexicon_vocab:
            # v0.3 (A2): codes are offered by what the reply SAYS (lexical retrieval over the whole registry),
            # then the current hypotheses' codes; never an alphabetical cut of the registry.
            vocab = {}
            for c, _ in self.store.lexicon().retrieve(reply, k=self.s.lexicon_k):
                if c in full_vocab:
                    vocab[c] = full_vocab[c]
                    lex_codes.add(c)
            for c, v in (self.store.vocabulary(hyp_ids) if hyp_ids else {}).items():
                vocab.setdefault(c, v)
        else:
            vocab = self.store.vocabulary(hyp_ids) if hyp_ids else dict(list(full_vocab.items())[:250])
        for c in priority:
            if c not in vocab and c in full_vocab:
                vocab[c] = full_vocab[c]
        test_ids = [a["test"] for a in last_actions if a.get("test")]
        for h in hyp_ids:
            test_ids += sorted(self.store.tests_for(h))
        tests = []
        for tid in dict.fromkeys(test_ids):
            td = self.store.test_def(tid)
            if td:
                tests.append((td.id, td.name, td.outcomes))

        out = self.llm.structured(
            ExtractionOutput,
            EXTRACT_SYSTEM,
            extract_user(state.get("presentation", ""), state.get("problem_rep", ""), last_actions, reply, vocab, priority, tests[:40]),
        )
        channel = last_actions[0]["type"] if last_actions else "history"
        new: list[dict[str, Any]] = []
        primary_seen = False
        for ef in out.findings:
            code = slug(ef.code)
            if not code:
                continue
            if self.s.canonical_codes:
                code = self._canonical_code(code, ef.label)
            source = channel if channel != "history" else ef.source
            conf = float(ef.confidence)
            if self.s.primary_complaint_only or self.s.prior_context:
                # v0.3 (A1): only the single primary complaint conditions the prior; the rest is evidence.
                # (P1 requires A1: the assessor sees only S, so nothing else may be withheld from the update.)
                cc = turn == 0 and ef.primary_complaint and not primary_seen
                primary_seen = primary_seen or cc
            else:
                # v0.2: the pretest prior conditions on the whole presentation, so none of it is re-counted.
                cc = turn == 0 or ef.chief_complaint
            new.append(
                {
                    "code": code,
                    "label": ef.label,
                    "present": ef.present,
                    "confidence": conf,
                    "source": source,
                    "evidence": ef.evidence,
                    "turn": turn,
                    "chief_complaint": cc,
                }
            )
        if (self.s.primary_complaint_only or self.s.prior_context) and turn == 0 and not primary_seen:
            for f in new:  # the LLM flagged none: the first present finding is the primary complaint
                if f["present"]:
                    f["chief_complaint"] = True
                    break
        llm_found = list(new)  # v0.4.1: what the extractor itself coded (before deterministic vital-sign rules)
        labs = [lv.model_dump() for lv in out.lab_values]
        new += apply_vital_rules(labs, self.store, turn)  # v0.3.1: deterministic vital signs override the LLM's coding
        new_tests = []
        for et in out.test_results:
            td = self.store.test_def(slug(et.test_id))
            outcome = slug(et.outcome)
            if td is not None and outcome in td.outcomes:
                new_tests.append({"test_id": td.id, "outcome": outcome, "confidence": float(et.confidence), "turn": turn, "evidence": et.evidence})
        new_tests += apply_numeric_rules(labs, self.store, turn)  # deterministic mapping overrides the LLM on ties

        # v0.4.1 diagnostics (A2 hit rate on real replies, negation share, codes the LLM coined)
        llm_codes = [f["code"] for f in llm_found]
        perceive_entry = {
            "turn": turn, "channel": channel, "offered": len(vocab), "offered_lexicon": len(lex_codes),
            "extracted": len(llm_found), "vital_rule": len(new) - len(llm_found), "from_lexicon": sum(1 for c in llm_codes if c in lex_codes),
            "from_cards_only": sum(1 for c in llm_codes if c in vocab and c not in lex_codes),
            "unoffered": sum(1 for c in llm_codes if c not in vocab), "negated": sum(1 for f in llm_found if not f["present"]),
            "tests": len(new_tests), "labs": len(labs), "hangul": bool(re.search(r"[\uac00-\ud7a3]", reply)),
        }
        context = dict(state.get("context", {}))
        if out.age is not None:
            context["age"] = out.age
        if out.sex is not None:
            context["sex"] = out.sex
        extra: dict[str, Any] = {}
        if self.s.prior_context:
            S = dict(state.get("prior_context") or {})
            if turn == 0 and not S.get("complaint"):
                S.update(self._choose_complaint(out, new, reply))
            for key, val in (("age", out.age), ("sex", out.sex), ("setting", out.setting)):
                if val not in (None, "", "unknown") and S.get(key) in (None, ""):  # first value wins: no flapping
                    S[key] = val
            extra["prior_context"] = S
        return {
            **extra,
            "findings": merge_findings(state.get("findings", []), new),
            "test_results": merge_tests(state.get("test_results", []), new_tests),
            "lab_values": state.get("lab_values", []) + labs,
            "context": context,
            "problem_rep": out.problem_representation or state.get("problem_rep", ""),
            "asked_codes": sorted(set(state.get("asked_codes", [])) | set(priority)),
            "perceive_log": state.get("perceive_log", []) + [perceive_entry],
        }

    def _choose_complaint(self, out: ExtractionOutput, new: list[dict[str, Any]], text: str) -> dict[str, Any]:
        """P1: pick the primary complaint by rule, flag exactly its finding, and return S's complaint fields."""
        low = text.lower()
        cands = []
        for i, ec in enumerate(out.complaints):
            c, how = self.complaints.resolve(ec.label)
            if c is None:
                continue
            q = ec.quote.strip().lower()
            pos = low.find(q) if q else -1
            cands.append({"key": c.key, "label": c.label, "raw": ec.label, "quote": ec.quote, "acuity": c.acuity,
                          "stated_reason": ec.stated_reason, "pos": pos if pos >= 0 else 10_000 + i,
                          "finding_code": ec.finding_code, "how": how})
        idx, rule = select_complaint(cands)
        flagged = next((f for f in new if f["chief_complaint"]), None)  # the extractor's own flag (A1 fallback)
        target, match = None, "none"
        if idx >= 0:
            ch = cands[idx]
            code = slug(ch["finding_code"])
            if code and self.s.canonical_codes:
                code = self._canonical_code(code, ch["raw"])
            target = next((f for f in new if code and f["code"] == code and f["present"]), None)
            match = "code" if target else match
            q = ch["quote"].strip().lower()
            if target is None and q:
                target = next((f for f in new if f["present"] and f["evidence"] and
                               (q in f["evidence"].lower() or f["evidence"].lower() in q)), None)
                match = "quote" if target else match
            if target is None:
                target, match = flagged, ("extractor_flag" if flagged else "none")
        else:
            target, match, rule = flagged, ("extractor_flag" if flagged else "none"), "fallback_finding"
            c, _ = self.complaints.resolve(flagged["label"] or flagged["code"]) if flagged else (None, "")
            ch = {"key": c.key, "label": c.label, "quote": flagged.get("evidence", ""), "acuity": c.acuity} if c else {}
        for f in new:
            f["chief_complaint"] = f is target
        return {"complaint": ch.get("key"), "complaint_label": ch.get("label"), "quote": ch.get("quote", ""),
                "acuity": ch.get("acuity"), "rule": rule, "finding": target["code"] if target else None,
                "finding_match": match,
                "candidates": [{k: c[k] for k in ("key", "raw", "stated_reason", "acuity", "how")} for c in cands]}

    def _canonical_code(self, code: str, label: str) -> str:
        """v0.3 (A3): resolve an extracted code to its canonical registry code (alias table, then a
        high-similarity lexical match). A genuinely new finding keeps its own code."""
        canon = self.store.canonical(code)
        if self.store.finding_def(canon) is not None:
            return canon
        sims = self.store.lexicon().similar(code, label, k=1)
        if sims and sims[0][1] >= self.s.dedup_auto:
            return sims[0][0]
        return canon

    # --------------------------------------------------------------- hypothesize

    def _merge_hypotheses(
        self, hyps: list[dict[str, Any]], proposed: list[ProposedHypothesis], turn: int, source: str
    ) -> tuple[list[dict[str, Any]], int]:
        by_id = {h["disease_id"]: dict(h) for h in hyps}
        names = {norm_key(h["name"]): h["disease_id"] for h in hyps}
        added = 0
        for p in proposed:
            # v0.4: a card match first, then the disease ontology (deterministic); an unresolved name keeps a slug id.
            cid, oid, how = self.store.resolve_disease(p.name, p.disease_id)
            if not cid:
                cid, oid, how = names.get(norm_key(p.name)) or slug(p.disease_id or p.name), "", "unresolved"
            if not cid:
                continue
            if cid in by_id:
                by_id[cid]["cant_miss"] = by_id[cid]["cant_miss"] or p.cant_miss
                continue
            label = self.store.ontology.disease.label(oid) if (oid and self.store.ontology.disease is not None) else ""
            by_id[cid] = {
                "disease_id": cid,
                "name": label or p.name,
                "prior": PRIOR_BUCKET[p.prior_bucket],
                "prior_source": f"{source}:{p.prior_bucket}",
                "cant_miss": p.cant_miss,
                "added_turn": turn,
                "ontology_id": oid,
                "resolution": how,
            }
            if self.s.prior_context:  # P1: the proposer's bucket is logged, never used; load_knowledge assigns
                by_id[cid].update({"proposed_bucket": p.prior_bucket, "prior_sig": None})
            names[norm_key(p.name)] = cid
            added += 1
        return list(by_id.values()), added

    def _discovery_target(self, state: dict[str, Any]) -> Optional[dict[str, Any]]:
        """v0.3 (C3): a clear, present finding that no card on the differential has explained for 2+ turns."""
        if not self.s.cardless_discovery or state.get("cardless_queries", 0) >= self.s.cardless_max_queries:
            return None
        done = set(state.get("discovery_done", []))
        streak = state.get("unexplained_streak", {})
        for f in state.get("findings", []):
            if (f["code"] not in done and f["present"] and not f.get("chief_complaint")
                    and f["confidence"] >= self.s.discovery_min_conf - 1e-9 and streak.get(f["code"], 0) >= 2):
                return f
        return None

    def _known(self) -> list[tuple[str, str]]:
        """Known-disease list for the generator prompt. A large compiled base is not listed (ids are resolved
        through the ontology instead), so the prompt never carries an arbitrary first-N slice of it."""
        return self.store.known_diseases() if len(self.store) <= self.s.max_known_in_prompt else []

    def hypothesize(self, state: dict[str, Any]) -> dict[str, Any]:
        turn = state.get("turn", 0)
        hyps = state.get("hypotheses", [])
        target = self._discovery_target(state)
        run = len(hyps) < 3 or turn <= 2 or bool(state.get("unexplained")) or turn % 2 == 0
        if not run and target is None:
            return {}
        update: dict[str, Any] = {}
        if run:
            system = HYPOTHESIZE_SYSTEM
            primary = ""
            if self.s.primary_complaint_only:
                system += HYPOTHESIZE_PRIMARY_NOTE
                pc = next((f for f in state.get("findings", []) if f.get("chief_complaint")), None)
                primary = (pc["label"] or pc["code"]) if pc else ""
                if self.s.prior_context and (state.get("prior_context") or {}).get("complaint_label"):
                    primary = state["prior_context"]["complaint_label"]
            out = self.llm.structured(
                HypothesisOutput,
                system,
                hypothesize_user(
                    state.get("presentation", ""),
                    state.get("problem_rep", ""),
                    state.get("findings", []),
                    state.get("differential", []),
                    state.get("unexplained", []),
                    self._known(),
                    primary,
                ),
            )
            hyps, added = self._merge_hypotheses(hyps, out.hypotheses, turn, "generator")
            # C1: the open-world guard, enforced. If the generator added nothing new to a non-empty list,
            # ask once more organized by organ system (capped per case).
            reprompts = state.get("open_world_reprompts", 0)
            if (self.s.open_world_enforce and added == 0 and state.get("hypotheses") and turn > 0
                    and reprompts < self.s.open_world_max):
                try:
                    out2 = self.llm.structured(
                        HypothesisOutput,
                        system + OPEN_WORLD_NOTE,
                        hypothesize_user(
                            state.get("presentation", ""), state.get("problem_rep", ""), state.get("findings", []),
                            state.get("differential", []), state.get("unexplained", []), self._known(),
                            primary,
                        ),
                    )
                    hyps, _ = self._merge_hypotheses(hyps, out2.hypotheses, turn, "open_world")
                except Exception as exc:
                    log.warning("Open-world re-prompt failed: %s", exc)
                update["open_world_reprompts"] = reprompts + 1
        if target is not None:
            try:
                out = self.llm.structured(
                    HypothesisOutput,
                    DISCOVERY_SYSTEM,
                    discovery_user(target["label"] or target["code"], state.get("problem_rep", ""),
                                   state.get("differential", []), self._known()),
                )
                hyps, _ = self._merge_hypotheses(hyps, out.hypotheses[:3], turn, "discovery")
            except Exception as exc:
                log.warning("Discovery query failed: %s", exc)
            update.update({"cardless_queries": state.get("cardless_queries", 0) + 1,
                           "discovery_done": state.get("discovery_done", []) + [target["code"]]})
        update["hypotheses"] = hyps
        return update

    # ------------------------------------------------------------ load_knowledge

    def load_knowledge(self, state: dict[str, Any]) -> dict[str, Any]:
        hyps = [dict(h) for h in state.get("hypotheses", [])]
        prior_logs: list[dict[str, Any]] = []
        S = state.get("prior_context") or {}
        if self.s.prior_context:  # P1: buckets from S alone, before anything is ranked by them
            hyps, prior_logs = self.priors.assign(S, hyps, state.get("turn", 0))
        post = {d["disease_id"]: d["prob"] for d in state.get("differential", [])}
        ranked = sorted(hyps, key=lambda h: post.get(h["disease_id"], h["prior"]), reverse=True)
        turn = state.get("turn", 0)
        forced = lambda h: h.get("cant_miss") or (  # noqa: E731
            h.get("added_turn") == turn and str(h.get("prior_source", "")).startswith(
                ("discovery",) + (("report_dissent",) if self.s.dissent_as_objection else ())))
        targets = ranked[: self.s.top_k_load] + [h for h in ranked[self.s.top_k_load :] if forced(h)]
        # v0.4: nothing is drafted. A hypothesis without knowledge (no own card, no carded ancestor) is logged as a
        # coverage miss. v0.4.1: fallbacks run in order for diseases the ontology resolves: the local documents in
        # sources/ first (on_demand card), then online PubMed abstracts only if enabled (provisional card).
        missing = [h for h in targets if not self.store.has_knowledge(h["disease_id"])]
        coverage = []
        fallbacks = list(getattr(self.rt, "fallbacks", None) or ([self.rt.fallback] if getattr(self.rt, "fallback", None) else []))
        via: dict[str, str] = {}
        if missing and fallbacks:
            for h in [h for h in missing if h.get("ontology_id")][: self.s.fallback_max]:
                for fb in fallbacks:
                    t0 = time.time()
                    try:
                        card = fb.compile_one(h["disease_id"], h["ontology_id"])
                    except Exception as exc:  # a fallback must never block the case
                        log.warning("%s failed for %s: %s", type(fb).__name__, h["disease_id"], exc)
                        card = None
                    via_s = round(time.time() - t0, 2)
                    if card is not None:
                        via[h["disease_id"]] = card.provenance.status
                        via[h["disease_id"] + "#s"] = via_s
                        break
        seen = {(c["disease_id"]) for c in state.get("coverage_log", [])}
        for h in missing:
            if h["disease_id"] not in seen:
                coverage.append({"turn": turn, "disease_id": h["disease_id"], "name": h["name"],
                                 "ontology_id": h.get("ontology_id", ""), "resolution": h.get("resolution", ""),
                                 "fallback": via.get(h["disease_id"], ""), "fallback_seconds": via.get(h["disease_id"] + "#s")})

        # Re-key hypotheses onto existing cards (a draft may have resolved to a known card by name).
        merged: dict[str, dict[str, Any]] = {}
        for h in hyps:
            card = self.store.get(h["disease_id"]) or self.store.get(h["name"])
            if card:
                h["disease_id"] = card.id
                h["cant_miss"] = h.get("cant_miss", False) or card.cant_miss
            if h["disease_id"] in merged:
                merged[h["disease_id"]]["prior"] = max(merged[h["disease_id"]]["prior"], h["prior"])
            else:
                merged[h["disease_id"]] = h
        out_h = list(merged.values())
        upd: dict[str, Any] = {"hypotheses": out_h, "critic_added": False}
        if coverage:
            upd["coverage_log"] = state.get("coverage_log", []) + coverage
        if self.s.prior_context:
            prior_logs += self.priors.refresh_cards(S, out_h, turn)
            if prior_logs:
                upd["prior_log"] = state.get("prior_log", []) + prior_logs
        return upd

    # ------------------------------------------------------------- update_belief

    def update_belief(self, state: dict[str, Any]) -> dict[str, Any]:
        hyps = state.get("hypotheses", [])
        findings = state.get("findings", [])
        tests = state.get("test_results", [])
        prev_ids = [d["disease_id"] for d in state.get("differential", [])]
        prior_added: list[dict[str, Any]] = []
        res = compute_posterior(hyps, findings, self.store, self.s, tests)
        kept = prune_hypotheses(hyps, res["differential"], self.s)
        turn = state.get("turn", 0)
        if len(kept) < len(hyps):
            res = compute_posterior(kept, findings, self.store, self.s, tests)
        if self.s.reverse_index and kept:
            S = state.get("prior_context") or {}
            added = reverse_index_candidates(kept, findings, self.store, self.s, tests, complaint=S.get("complaint"))
            if added and self.s.prior_context:  # card-keyed prior, LLM-free; reassessed only if S changes
                for c in added:
                    b = c["prior_source"].split(":", 1)[-1]
                    c.update({"prior_sig": signature(S), "prior_route": "card" if b in PRIOR_BUCKET else "default",
                              "prior_bucket": b if b in PRIOR_BUCKET else None})
            if added:
                kept = kept + [{**c, "added_turn": turn} for c in added]
                res = compute_posterior(kept, findings, self.store, self.s, tests)
                if self.s.prior_context:
                    prior_added = [{"turn": turn, "disease_id": c["disease_id"], "entry": c["prior_source"], "proposed": None,
                                    "card": c["prior_bucket"], "assessor": None, "final": c["prior_bucket"],
                                    "route": c["prior_route"], "reassigned": False} for c in added]
        trace = [t for t in state.get("hypothesis_trace", []) if t["turn"] != turn]
        trace.append({"turn": turn, "hypotheses": [h["name"] for h in kept],
                      "sources": {h["name"]: h.get("prior_source", "") for h in kept},
                      # v0.4.1: ids, routes and posteriors per turn (hypothesis recall, truth trajectory)
                      "ids": {h["disease_id"]: h.get("prior_source", "") for h in kept},
                      "probs": {d["disease_id"]: round(d["prob"], 4) for d in res["differential"][:12]},
                      "other": round(res["other_prob"], 4)}),
        out = {
            "hypotheses": kept,
            "hypothesis_trace": trace,
            "differential": res["differential"],
            "other_prob": res["other_prob"],
            "entropy": res["entropy"],
            "unexplained": res["unexplained"],
            "belief_trace": res["trace"],
        }
        if prior_added:
            out["prior_log"] = state.get("prior_log", []) + prior_added
        out.update(self._list_effect(state, prev_ids, kept, res, findings, tests, turn))
        out.update(self._streaks(state, res, turn))
        return out

    def _list_effect(self, state, prev_ids, kept, res, findings, tests, turn) -> dict[str, Any]:
        """B1 diagnostic: how much the leader's probability moved because the LIST changed, evidence held fixed."""
        new_ids = [h["disease_id"] for h in kept]
        if not prev_ids or set(prev_ids) == set(new_ids) or not res["differential"]:
            return {}
        prev_h = [h for h in kept if h["disease_id"] in set(prev_ids)]
        if not prev_h:
            return {}
        leader = res["differential"][0]["disease_id"]
        if leader not in {h["disease_id"] for h in prev_h}:
            return {}
        old = compute_posterior(prev_h, findings, self.store, self.s, tests)
        p_old = next(d["prob"] for d in old["differential"] if d["disease_id"] == leader)
        p_new = res["differential"][0]["prob"]
        entry = {"turn": turn, "leader": leader, "p_same_evidence_old_list": round(p_old, 4), "p_new_list": round(p_new, 4),
                 "added": sorted(set(new_ids) - set(prev_ids)), "removed": sorted(set(prev_ids) - set(new_ids))}
        return {"list_effect_log": state.get("list_effect_log", []) + [entry]}

    def _streaks(self, state: dict[str, Any], res: dict[str, Any], turn: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        # C3: consecutive turns each clear finding stayed unexplained (counted once per turn).
        if state.get("unexplained_turn") != turn:
            prev = state.get("unexplained_streak", {})
            out["unexplained_streak"] = {c: prev.get(c, 0) + 1 for c in res.get("unexplained_codes", [])}
            out["unexplained_turn"] = turn
        return out

    # ------------------------------------------------------------------ contrast

    def contrast(self, state: dict[str, Any]) -> dict[str, Any]:
        """Elicit "A vs B" knowledge for live confusable pairs that lack a contrast card (once per turn)."""
        turn = state.get("turn", 0)
        contrast = getattr(self.rt, "contrast", None)
        if contrast is None or state.get("contrast_turn") == turn or turn < len(self.s.intake_script):
            return {"contrast_added": False}
        live = [d for d in state.get("differential", []) if d["prob"] >= self.s.contrast_min_prob and d.get("has_card")][:4]
        # v0.4: a disease and its ontology ancestor are one subtree, not a confusable pair.
        pairs = [(a["disease_id"], b["disease_id"]) for i, a in enumerate(live) for b in live[i + 1 :]
                 if not self.store.has_contrast(a["disease_id"], b["disease_id"])
                 and not self.store.is_ancestor(a["disease_id"], b["disease_id"])
                 and not self.store.is_ancestor(b["disease_id"], a["disease_id"])]
        pairs = pairs[: self.s.max_contrasts_per_turn]
        added = False
        presentation = state.get("problem_rep") or state.get("presentation", "")
        if pairs:
            with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
                added = any(r is not None for r in pool.map(usage.bind(lambda p: contrast.draft_contrast(p[0], p[1], presentation)), pairs))
        return {"contrast_turn": turn, "contrast_added": added,
                "contrast_log": state.get("contrast_log", []) + [{"turn": turn, "pairs": [list(p) for p in pairs], "added": bool(added)}]}

    @staticmethod
    def route_after_contrast(state: dict[str, Any]) -> str:
        return "update_belief" if state.get("contrast_added") else "critic"

    # -------------------------------------------------------------------- critic

    def critic(self, state: dict[str, Any]) -> dict[str, Any]:
        if not self.s.critic:  # v0.4.1 ablation switch (DX_CRITIC=0)
            return {"critic_added": False}
        call, considering = should_call_critic(state, self.s)
        if not call:
            return {"critic_added": False}
        turn = state.get("turn", 0)
        try:
            out = self.llm.structured(
                CriticOutput,
                CRITIC_SYSTEM,
                critic_user(
                    state.get("problem_rep", ""),
                    state.get("differential", []),
                    state.get("findings", []),
                    state.get("unexplained", []),
                    considering,
                ),
            )
        except Exception as exc:
            log.warning("Critic failed: %s", exc)
            return {"critic_added": False, "critic_turn": turn}
        hyps, added = self._merge_hypotheses(state.get("hypotheses", []), out.missing_hypotheses[:3], turn, "critic")
        notes = state.get("critic_notes", []) + [f"T{turn}: {c}" for c in out.concerns[:3]]
        update: dict[str, Any] = {
            "critic_block": out.block_commit and not self.s.structured_veto,
            "critic_turn": turn,
            "critic_notes": notes,
        }
        if self.s.structured_veto and considering and out.veto_alternative.strip():
            hyps, vadded, vupdate = self._structured_veto(state, out, hyps, turn)
            added += vadded
            update.update(vupdate)
        update.update({"hypotheses": hyps, "critic_added": added > 0})
        return update

    def _structured_veto(self, state: dict[str, Any], out: CriticOutput, hyps: list[dict[str, Any]], turn: int):
        """D2: turn a named objection into something the arithmetic can settle.

        Not honored when it names the leader itself, when the same (leader, alternative, reason) objection was
        already raised and no finding or test has arrived since, or when the per-case failsafe ceiling is reached.
        If honored and the alternative is not on the list, it is added (and gets a card), so the objection enters
        the belief update; the planner then separates it from the leader only with a positive-value action.
        """
        diff = state.get("differential", [])
        lead = diff[0]["disease_id"] if diff else ""
        prop = ProposedHypothesis(name=out.veto_alternative, disease_id=slug(out.veto_alternative),
                                  prior_bucket="uncommon", cant_miss=out.veto_reason == "cant_miss_not_excluded",
                                  rationale=out.veto_evidence[:120])
        card = self.store.get(prop.disease_id) or self.store.get(prop.name)
        if card and (card.prevalence or card.base_prevalence):
            prop.prior_bucket = card.prevalence or card.base_prevalence
        alt = card.id if card else (self.store.resolve(prop.name, prop.disease_id) or prop.disease_id)
        n_obs = len(state.get("findings", [])) + len(state.get("test_results", []))
        sig = [lead, alt, out.veto_reason]
        past = state.get("vetoes", [])
        honored_n = sum(1 for v in past if v["honored"])
        why = "honored"
        if alt == lead:
            why = "names the leader"
        elif self.s.veto_no_repeat and any(v["sig"] == sig and v["n_obs"] == n_obs for v in past):
            why = "repeat without new evidence"
        elif honored_n >= self.s.veto_ceiling:
            why = "ceiling reached"
        honored = why == "honored"
        record = {"turn": turn, "sig": sig, "n_obs": n_obs, "honored": honored, "why": why,
                  "evidence": out.veto_evidence[:160]}
        added = 0
        if honored and alt not in {h["disease_id"] for h in hyps}:
            hyps, added = self._merge_hypotheses(hyps, [prop], turn, "critic_veto")
        upd = {"vetoes": past + [record], "veto": {"alternative": alt, "turn": turn} if honored else {}}
        return hyps, added, upd

    @staticmethod
    def route_after_critic(state: dict[str, Any]) -> str:
        return "load_knowledge" if state.get("critic_added") else "plan"

    # ---------------------------------------------------------------------- plan

    def plan(self, state: dict[str, Any]) -> dict[str, Any]:
        turn = state.get("turn", 0)
        if turn < len(self.s.intake_script):
            spec = INTAKE_ACTIONS[self.s.intake_script[turn]]
            action = {
                "id": spec["id"],
                "name": spec["name"],
                "type": spec["type"],
                "reveals": [],
                "cost": DEFAULT_ACTION_COST[spec["type"]],
                "risk": 0.0,
                "intake": True,
            }
            return {"last_actions": [action], "commit_reason": ""}

        findings = state.get("findings", [])
        tested = {t["test_id"] for t in state.get("test_results", [])}
        observed = {f["code"] for f in findings} | set(state.get("asked_codes", [])) | tested
        critic_block = bool(state.get("critic_block")) and state.get("critic_turn") == turn
        diff = state.get("differential", [])
        other = state.get("other_prob", 0.0)
        dec = decide(
            diff,
            other,
            self.store,
            self.s,
            turn,
            observed,
            set(state.get("done_action_ids", [])),
            critic_block=critic_block,
            stall_count=state.get("stall_count", 0),
            veto=(state.get("veto") if state.get("veto", {}).get("turn") == turn else None),
        )
        stall = state.get("stall_count", 0) + 1 if dec.best_voi <= self.s.voi_epsilon else 0
        log_entry = {
            "turn": turn,
            "reason": dec.reason,
            "best_voi": round(dec.best_voi, 5),
            "top_candidates": [{"id": s.action.id, "voi": round(s.voi, 5)} for s in dec.scored[:5]],
        }
        action_log = state.get("action_log", []) + [log_entry]
        if dec.commit:
            upd = {"phase": "committing", "commit_reason": dec.reason, "last_actions": [], "action_log": action_log, "stall_count": stall}
            if "unresolved objection" in dec.reason:
                upd["unresolved_objection"] = {"turn": turn, "alternative": state.get("veto", {}).get("alternative")}
            return upd

        vocab = self.store.vocabulary()
        if not dec.actions:
            action = {"id": f"free_q_{turn}", "name": "focused history question", "type": "history", "reveals": [], "cost": 0.0, "risk": 0.0, "free": True}
            taken = state.get("actions_taken", []) + [{"turn": turn, "ids": [action["id"]], "type": "history", "cost": 0.0,
                                                       "voi": 0.0, "free": True}]
            return {"last_actions": [action], "action_log": action_log, "stall_count": stall, "actions_taken": taken}

        actions = [
            {
                "id": s.action.id,
                "name": s.action.name,
                "type": s.action.type,
                "reveals": [] if s.action.test else s.reveals,
                "test": s.action.test,
                "reveal_labels": [s.action.name] if s.action.test else [vocab.get(c, {}).get("label", c) for c in s.reveals],
                "cost": float(s.action.cost),
                "risk": float(s.action.risk),
                "voi": round(s.voi, 5),
            }
            for s in dec.actions
        ]
        taken = state.get("actions_taken", []) + [{"turn": turn, "ids": [a["id"] for a in actions], "type": actions[0]["type"],
                                                   "cost": sum(a["cost"] for a in actions), "voi": round(sum(a["voi"] for a in actions), 5)}]
        return {"last_actions": actions, "action_log": action_log, "stall_count": stall, "actions_taken": taken}

    @staticmethod
    def route_after_plan(state: dict[str, Any]) -> str:
        return "report" if state.get("phase") == "committing" else "interview"

    # ----------------------------------------------------------------- interview

    def interview(self, state: dict[str, Any]) -> dict[str, Any]:
        turn = state.get("turn", 0)
        actions = [dict(a) for a in state.get("last_actions", [])]
        first = actions[0]
        if first.get("intake"):
            if first["id"] == "open_hpi" and state.get("presentation_source") == "patient":
                message = OPEN_HPI_FOLLOWUP
            else:
                message = INTAKE_ACTIONS[first["id"]]["message"]
        elif first.get("free"):
            q = self.llm.structured(
                FreeQuestionOutput,
                FREE_QUESTION_SYSTEM,
                free_question_user(state.get("problem_rep", ""), state.get("differential", []), state.get("findings", [])),
            )
            message = q.question
            first["name"] = q.target
            actions[0] = first
        else:
            message = self.llm.structured(
                InterviewerOutput,
                INTERVIEW_SYSTEM,
                interview_user(actions, state.get("problem_rep", ""), state.get("transcript", [])),
            ).message

        transcript = state.get("transcript", []) + [{"role": "doctor", "type": first["type"], "text": message, "turn": turn + 1}]
        return {
            "doctor_message": message,
            "last_actions": actions,
            "transcript": transcript,
            "turn": turn + 1,
            "cost_spent": state.get("cost_spent", 0.0) + sum(a.get("cost", 0.0) for a in actions),
            "done_action_ids": state.get("done_action_ids", []) + [a["id"] for a in actions],
            "asked_codes": sorted(set(state.get("asked_codes", [])) | {c for a in actions for c in a.get("reveals", [])}),
        }

    # --------------------------------------------------------------- await_reply

    def await_reply(self, state: dict[str, Any]) -> dict[str, Any]:
        actions = state.get("last_actions", [])
        reply = interrupt({"message": state.get("doctor_message", ""), "actions": actions, "turn": state.get("turn", 0)})
        kind = actions[0]["type"] if actions else "history"
        role = "patient" if kind == "history" else "clinical_team"
        transcript = state.get("transcript", []) + [{"role": role, "type": kind, "text": str(reply), "turn": state.get("turn", 0)}]
        return {"last_reply": str(reply), "transcript": transcript}

    # -------------------------------------------------------------------- report

    def report(self, state: dict[str, Any]) -> dict[str, Any]:
        diff = state.get("differential", [])
        findings = state.get("findings", [])
        llm_dx, agrees, dissent_name, dissent_reason = "", True, "", ""
        try:
            out = self.llm.structured(
                ReportOutput,
                REPORT_SYSTEM,
                report_user(state.get("problem_rep", ""), diff, findings, state.get("commit_reason", ""), state.get("test_results", [])),
            )
            llm_dx, rationale = out.final_diagnosis, out.rationale
            key, steps = out.key_findings, out.recommended_next_steps
            agrees, dissent_name, dissent_reason = out.agrees_with_top, out.dissent_diagnosis.strip(), out.dissent_reason
        except Exception as exc:
            log.warning("Report LLM failed, falling back to top of differential: %s", exc)
            rationale, key, steps = "Report generation failed; top of computed differential shown.", [], []
        top = diff[0] if diff else None
        top_name = top["name"] if top else (llm_dx or "Undetermined")
        # E1: the answer is the top of the computed differential. The LLM may dissent; a dissent is re-examined
        # by the arithmetic once (the named disease joins the differential), never substituted directly.
        final_dx = top_name if self.s.deterministic_final else (llm_dx or top_name)
        named = dissent_name or ("" if agrees else llm_dx)
        if not named and llm_dx and top and not self._same_disease(llm_dx, top):
            named = llm_dx  # the LLM named something else without flagging it
        dissents = list(state.get("report_dissent", []))
        if named and top and not self._same_disease(named, top):
            entry = {"turn": state.get("turn", 0), "dissent": named, "reason": dissent_reason[:200], "top": top_name}
            hyps = state.get("hypotheses", [])
            card = self.store.get(slug(named)) or self.store.get(named)
            on_list = bool(card and card.id in {h["disease_id"] for h in hyps})
            if (self.s.deterministic_final and self.s.dissent_reevaluate and self.s.dissent_as_objection
                    and not state.get("dissent_checked")):
                # Revised E1: add the named disease if needed, then hold the commit as an objection that only
                # a positive-value separating action can sustain (planner.decide); otherwise commit as before.
                turn = state.get("turn", 0)
                new_hyps = hyps
                if not on_list:
                    prop = ProposedHypothesis(name=named, disease_id=slug(named),
                                              prior_bucket=((card.prevalence or card.base_prevalence or "uncommon") if card else "uncommon"),
                                              cant_miss=bool(card and card.cant_miss), rationale=dissent_reason[:120])
                    new_hyps, _ = self._merge_hypotheses(hyps, [prop], turn, "report_dissent")
                alt_id = card.id if card else slug(named)
                entry["outcome"] = "objection" + ("" if on_list else " (added)")
                return {"hypotheses": new_hyps, "dissent_checked": True, "phase": "gathering",
                        "report_dissent": dissents + [entry],
                        "veto": {"alternative": alt_id, "turn": turn, "source": "report_dissent"}}
            if (self.s.deterministic_final and self.s.dissent_reevaluate and not state.get("dissent_checked")
                    and not on_list):
                prop = ProposedHypothesis(name=named, disease_id=slug(named),
                                          prior_bucket=((card.prevalence or card.base_prevalence or "uncommon") if card else "uncommon"),
                                          cant_miss=bool(card and card.cant_miss), rationale=dissent_reason[:120])
                new_hyps, _ = self._merge_hypotheses(hyps, [prop], state.get("turn", 0), "report_dissent")
                entry["outcome"] = "re-evaluated"
                return {"hypotheses": new_hyps, "dissent_checked": True, "phase": "gathering",
                        "report_dissent": dissents + [entry]}
            entry["outcome"] = "already on the differential" if on_list else "logged"
            dissents.append(entry)

        report = {
            "final_diagnosis": final_dx,
            "rationale": rationale,
            "key_findings": key,
            "recommended_next_steps": steps,
            # v0.4.1: the evidence behind the top 3 and the knowledge each value came from
            "evidence": {d["disease_id"]: evidence_breakdown(d["disease_id"], [f for f in findings if not f.get("chief_complaint")],
                                                               state.get("test_results", []), self.store) for d in diff[:3]},
            "n_findings": len(findings), "n_tests": len(state.get("test_results", [])),
            "differential": [{"name": d["name"], "prob": round(d["prob"], 4), "cant_miss": d["cant_miss"], "disease_id": d["disease_id"],
                              "card_status": d.get("card_status", ""), "tree_parent": d.get("tree_parent")} for d in diff[:8]],
            "other_prob": round(state.get("other_prob", 0.0), 4),
            "commit_reason": state.get("commit_reason", ""),
            "turns": state.get("turn", 0),
            "cost_spent": state.get("cost_spent", 0.0),
            "cards_used": {d["disease_id"]: d.get("card_status", "none") for d in diff[:8]},
            # v0.4: hypotheses that had no knowledge (no own card, no carded ancestor) when considered
            "coverage_misses": state.get("coverage_log", []),
            "knowledge_base": {k: self.store.kb_info.get(k) for k in ("compiler", "compiled_at", "counts")} if self.store.kb_info else None,
            "llm_final_diagnosis": llm_dx,
            "report_dissent": dissents,
            "unresolved_objection": state.get("unresolved_objection") or None,
            "vetoes": state.get("vetoes", []),
            # v0.3.1 diagnostics (P1 step 1): how much the prior decides the answer, and which context moved nothing
            "prior_sensitivity": prior_sensitivity(state.get("hypotheses", []), findings, self.store, self.s,
                                                   state.get("test_results", []), self.s.prior_sensitivity_alpha),
            "context_coverage": context_coverage(findings, diff, self.store, len(self.s.intake_script)),
            "prior_context": state.get("prior_context") or None,
            "prior_log": state.get("prior_log", []),
        }
        top3 = "; ".join(f"{d['name']} ({d['prob']:.0%})" for d in diff[:3])
        message = f"My working diagnosis is {final_dx}. {rationale} Differential: {top3}."
        transcript = state.get("transcript", []) + [{"role": "doctor", "type": "diagnosis", "text": message, "turn": state.get("turn", 0)}]
        return {"final_report": report, "phase": "committed", "doctor_message": message, "transcript": transcript,
                "report_dissent": dissents}

    def _same_disease(self, name: str, d: dict[str, Any]) -> bool:
        card = self.store.get(slug(name)) or self.store.get(name)
        if card is not None:
            return card.id == d["disease_id"]
        return norm_key(name) in (norm_key(d["name"]), norm_key(d["disease_id"]))

    @staticmethod
    def route_after_report(state: dict[str, Any]) -> str:
        return "end" if state.get("phase") == "committed" else "load_knowledge"
