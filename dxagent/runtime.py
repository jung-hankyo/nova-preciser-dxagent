"""Runtime: the dependencies every node needs (models, knowledge, settings)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from dxagent.config import Settings, get_settings
from dxagent.knowledge.cards import CardStore
from dxagent.knowledge.complaints import ComplaintRegistry
from dxagent.knowledge.contrast import ContrastService
from dxagent.llm import LLM, LLMLike

log = logging.getLogger(__name__)


def make_store(settings: Settings, fixtures: bool = False) -> CardStore:
    """The run-time knowledge store: registry + compiled kb/ + cache/. `fixtures=True` loads the demo fixture cards
    instead of kb/ (tests and --offline plumbing checks only)."""
    if fixtures:
        return CardStore(settings.registry_dir, None, settings.kb_perturb, settings.kb_perturb_seed,
                         card_dirs=[settings.fixture_dir], inherit_depth=settings.inherit_depth, inherit_decay=settings.inherit_decay)
    return CardStore(settings.registry_dir, settings.cache_dir, settings.kb_perturb, settings.kb_perturb_seed,
                     kb_dir=settings.kb_dir, inherit_depth=settings.inherit_depth, inherit_decay=settings.inherit_decay)


@dataclass
class Runtime:
    settings: Settings
    doctor: LLMLike
    store: CardStore
    contrast: Optional[ContrastService] = None  # v0.4: LLM contrast cards (existing codes only, capped confidence)
    fallback: Optional[Any] = None  # deprecated single fallback (kept for harnesses); use `fallbacks`
    fallbacks: list = field(default_factory=list)  # v0.4.1: tried in order: LocalFallback, then LiveFallback
    prior_llm: Optional[LLMLike] = None  # v0.3.1 (P1): the S-only prior assessor (defaults to the doctor model)
    complaints: Optional[ComplaintRegistry] = None  # v0.3.1 (P1): canonical complaints, learned entries cached

    @classmethod
    def create(
        cls,
        settings: Optional[Settings] = None,
        doctor: Optional[LLMLike] = None,
        card_llm: Optional[LLMLike] = None,
        store: Optional[CardStore] = None,
    ) -> "Runtime":
        settings = settings or get_settings()
        doctor = doctor or LLM(settings.doctor_model, settings.temperature, settings.max_tokens, role="doctor")
        store = store or make_store(settings)
        if len(store) == 0:
            log.warning("No compiled knowledge base at %s: run `dxagent kb fetch` and `dxagent kb compile`.", settings.kb_dir)
        scripted = not isinstance(doctor, LLM)  # tests and --offline: every role uses the injected stand-in
        if card_llm is None:
            card_llm = doctor if scripted else LLM(settings.card_model, settings.temperature, settings.max_tokens, role="knowledge")
        contrast = ContrastService(card_llm, store, settings) if settings.contrast_cards else None
        fallbacks: list = []
        if settings.local_fallback:
            from dxagent.knowledge.compiler.local import LocalFallback

            fallbacks.append(LocalFallback(settings, card_llm, store))
        if settings.live_fallback:
            from dxagent.knowledge.compiler.online import LiveFallback

            fallbacks.append(LiveFallback(settings, card_llm, store))
        prior_llm = doctor
        if not scripted and settings.prior_model and settings.prior_model != settings.doctor_model:
            prior_llm = LLM(settings.prior_model, settings.temperature, settings.max_tokens, role="prior")
        complaints = ComplaintRegistry(settings.registry_dir, store.cache_dir, llm=doctor,
                                       auto=settings.dedup_auto, confirm=settings.dedup_confirm)
        return cls(settings=settings, doctor=doctor, store=store, contrast=contrast, fallbacks=fallbacks,
                   prior_llm=prior_llm, complaints=complaints)
