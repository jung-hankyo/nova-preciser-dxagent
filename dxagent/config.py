"""Central configuration. Every tunable number lives here, loaded from env vars.

Probabilities are 0-1. Costs are in US-dollar-like units, converted into loss
units by `cost_weight`. Change values via environment variables (see .env.example)
rather than editing code, so experiments stay reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent

ACTION_TYPES = ("history", "exam", "poc", "lab", "imaging", "advanced_imaging", "procedure")
TEST_TYPES = ("poc", "lab", "imaging", "advanced_imaging", "procedure")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _envf(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _envi(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


# Default cost per action type when a card does not specify one.
DEFAULT_ACTION_COST = {
    "history": 0.0,
    "exam": 5.0,
    "poc": 20.0,
    "lab": 40.0,
    "imaging": 300.0,
    "advanced_imaging": 900.0,
    "procedure": 1500.0,
}

# Default procedural risk (loss units) per action type.
DEFAULT_ACTION_RISK = {
    "history": 0.0,
    "exam": 0.0,
    "poc": 0.0,
    "lab": 0.005,
    "imaging": 0.02,
    "advanced_imaging": 0.3,  # radiation, contrast
    "procedure": 1.5,  # invasive risk
}

# Prerequisite gating: an action of this type is only considered when some disease it
# discriminates already has at least this posterior probability (plan section 5, rule 1).
MIN_TARGET_PROB = {"advanced_imaging": 0.05, "procedure": 0.25}

# How reliably each channel reports the true state of a finding.
CHANNEL_RELIABILITY = {
    "history": 0.85,
    "exam": 0.90,
    "poc": 0.95,
    "lab": 0.97,
    "imaging": 0.97,
    "advanced_imaging": 0.98,
    "procedure": 0.99,
}

# Frequency levels: P(finding | disease), P(test outcome | disease), and background rates
# P(finding | patients presenting with this complaint). Knowledge is stored per disease;
# likelihood ratios are computed at run time between the diseases actually on the differential.
FREQ = {"very_common": 0.90, "common": 0.65, "occasional": 0.35, "uncommon": 0.15, "rare": 0.05, "very_rare": 0.01}
FREQ_LEVELS = list(FREQ)  # ordered, most to least frequent

# Pretest probability buckets, given the presenting complaint and demographics.
PRIOR_BUCKET = {"very_common": 0.25, "common": 0.08, "uncommon": 0.02, "rare": 0.005, "very_rare": 0.001}

# Harm profile levels (used by the pairwise confusion loss in "clinical" objective mode).
MISS_HARM = {"lethal": 3.0, "serious": 1.5, "self_limited": 0.3}
URGENCY = {"hours": 2.0, "days": 1.3, "weeks": 1.0}
OVERTREAT_HARM = {"high": 1.0, "moderate": 0.4, "low": 0.1}


@dataclass
class Settings:
    # Models, in LangChain "provider:model" form.
    # v0.4.2 default profile: DeepSeek for the agent, OpenAI for the environment and the judge (different families,
    # so the patient never shares the doctor's model and the judge never grades its own family). Spec options after
    # "?" (llm.py): DeepSeek thinking=off|low|high|max, OpenAI reasoning effort=minimal|low|medium|high.
    doctor_model: str = field(default_factory=lambda: _env("DX_DOCTOR_MODEL", "deepseek:deepseek-v4-pro?thinking=low"))
    card_model: str = field(default_factory=lambda: _env("DX_CARD_MODEL", "deepseek:deepseek-v4-pro"))
    patient_model: str = field(default_factory=lambda: _env("DX_PATIENT_MODEL", "openai:gpt-5-mini?effort=minimal"))
    measurement_model: str = field(default_factory=lambda: _env("DX_MEASUREMENT_MODEL", "openai:gpt-5-mini?effort=minimal"))
    judge_model: str = field(default_factory=lambda: _env("DX_JUDGE_MODEL", "openai:gpt-5-mini?effort=medium"))
    temperature: float | None = field(
        default_factory=lambda: float(os.environ["DX_TEMPERATURE"]) if os.getenv("DX_TEMPERATURE") else None
    )
    max_tokens: int = field(default_factory=lambda: _envi("DX_MAX_TOKENS", 4096))

    # Knowledge layer (v0.4). Compiled knowledge is the default and expected source; nothing is drafted live
    # unless the live fallback is switched on.
    registry_dir: Path = PACKAGE_DIR / "knowledge" / "registry"  # curated registries + local ontology terms
    fixture_dir: Path = PACKAGE_DIR / "knowledge" / "fixtures" / "chest_pain_demo"  # tests and --offline only
    kb_dir: Path = field(default_factory=lambda: Path(_env("DX_KB_DIR", str(PROJECT_DIR / "kb"))))
    sources_dir: Path = field(default_factory=lambda: Path(_env("DX_SOURCES_DIR", str(PROJECT_DIR / "sources"))))
    cache_dir: Path = field(default_factory=lambda: Path(_env("DX_CACHE_DIR", str(PROJECT_DIR / "cache"))))
    # Fallbacks for a resolved disease with no knowledge (no own card, no carded ancestor), tried in order:
    # 1. local (default ON): search the documents downloaded into sources/ and compile an on_demand card through the
    #    same pipeline as `kb compile` (shared extraction cache).
    local_fallback: bool = field(default_factory=lambda: _env("DX_LOCAL_FALLBACK", "1") == "1")
    local_fallback_max_docs: int = field(default_factory=lambda: _envi("DX_LOCAL_FALLBACK_MAX_DOCS", 4))  # per disease
    fallback_max: int = field(default_factory=lambda: _envi("DX_FALLBACK_MAX", 2))  # diseases compiled per turn
    # 2. live / online (default OFF): a resolved disease the local documents do not cover is compiled mid-case from whitelisted
    # online sources (PubMed, publication-type filtered) through the same compiler. Off by default.
    live_fallback: bool = field(default_factory=lambda: _env("DX_LIVE_FALLBACK", "0") == "1")
    live_fallback_conf_cap: float = field(default_factory=lambda: _envf("DX_LIVE_FALLBACK_CONF_CAP", 0.5))
    # Contrast cards stay LLM-elicited (the one remaining LLM-only knowledge path); restricted to existing codes and
    # tests, confidence capped, never admitted to the reverse index.
    contrast_cards: bool = field(default_factory=lambda: _env("DX_CONTRAST", "1") == "1")
    contrast_samples: int = field(default_factory=lambda: _envi("DX_CONTRAST_SAMPLES", 3))
    contrast_conf_cap: float = field(default_factory=lambda: _envf("DX_CONTRAST_CONF_CAP", 0.55))
    strict_quorum: bool = field(default_factory=lambda: _env("DX_STRICT_QUORUM", "1") == "1")
    # Ontology hierarchy. A card-less disease inherits its nearest carded ancestor within `inherit_depth` levels,
    # confidence x `inherit_decay` per level. `tree`: listed ancestors and descendants are a tree, not rivals.
    inherit_depth: int = field(default_factory=lambda: _envi("DX_INHERIT_DEPTH", 3))
    inherit_decay: float = field(default_factory=lambda: _envf("DX_INHERIT_DECAY", 0.9))
    tree: bool = field(default_factory=lambda: _env("DX_TREE", "1") == "1")
    tree_residual_floor: float = field(default_factory=lambda: _envf("DX_TREE_RESIDUAL_FLOOR", 0.10))
    # Compiler.
    target_population: str = field(default_factory=lambda: _env("DX_TARGET_POPULATION", "adult"))
    hpo_scope: str = field(default_factory=lambda: _env("DX_HPO_SCOPE", "all"))  # all | listed | none
    hpo_min_annotations: int = field(default_factory=lambda: _envi("DX_HPO_MIN_ANNOTATIONS", 3))
    hpo_propagate_depth: int = field(default_factory=lambda: _envi("DX_HPO_PROPAGATE", 2))
    hpo_min_term_depth: int = field(default_factory=lambda: _envi("DX_HPO_MIN_TERM_DEPTH", 3))
    derived_bg_min: float = field(default_factory=lambda: _envf("DX_DERIVED_BG_MIN", 0.02))
    derived_bg_max: float = field(default_factory=lambda: _envf("DX_DERIVED_BG_MAX", 0.5))
    # Rare-disease (HPO-only) cards together get at most this share of the weight when backgrounds are derived.
    derived_bg_rare_share: float = field(default_factory=lambda: _envf("DX_DERIVED_BG_RARE_SHARE", 0.05))
    planner_max_codes: int = field(default_factory=lambda: _envi("DX_PLANNER_MAX_CODES", 30))  # per disease
    max_known_in_prompt: int = field(default_factory=lambda: _envi("DX_MAX_KNOWN_IN_PROMPT", 400))
    contrast_min_prob: float = field(default_factory=lambda: _envf("DX_CONTRAST_MIN_PROB", 0.10))
    # Reverse index: known diseases that the evidence supports join the differential even if the LLM missed them.
    reverse_index: bool = field(default_factory=lambda: _env("DX_REVERSE_INDEX", "1") == "1")
    critic: bool = field(default_factory=lambda: _env("DX_CRITIC", "1") == "1")  # v0.4.1: 0 = ablation without the safety critic
    reverse_min_prob: float = field(default_factory=lambda: _envf("DX_REVERSE_MIN_PROB", 0.05))
    reverse_max_per_turn: int = field(default_factory=lambda: _envi("DX_REVERSE_MAX", 2))
    reverse_prior: float = field(default_factory=lambda: _envf("DX_REVERSE_PRIOR", 0.02))
    max_contrasts_per_turn: int = field(default_factory=lambda: _envi("DX_MAX_CONTRASTS", 2))
    # v0.3 knowledge fixes.
    canonical_codes: bool = field(default_factory=lambda: _env("DX_CANONICAL_CODES", "1") == "1")  # A3: one code per finding
    dedup_auto: float = field(default_factory=lambda: _envf("DX_DEDUP_AUTO", 0.85))  # A3: merge without asking
    dedup_confirm: float = field(default_factory=lambda: _envf("DX_DEDUP_CONFIRM", 0.60))  # A3: ask the LLM between the two
    # C3 (shipped): an index-added disease starts from its card's prevalence bucket; admission is the v0.2 test.
    reverse_prevalence_prior: bool = field(default_factory=lambda: _env("DX_REVERSE_PREVALENCE", "1") == "1")
    # C3 alternative, rejected (lowered recall on development worlds): admit on a Bayes factor vs the list.
    reverse_evidence_rule: bool = field(default_factory=lambda: _env("DX_REVERSE_EVIDENCE", "0") == "1")
    reverse_bf: float = field(default_factory=lambda: _envf("DX_REVERSE_BF", 5.0))
    reverse_floor: float = field(default_factory=lambda: _envf("DX_REVERSE_FLOOR", 0.01))
    cardless_discovery: bool = field(default_factory=lambda: _env("DX_CARDLESS_DISCOVERY", "0") == "1")  # off: lowered top-1 (v0.3 record)
    cardless_max_queries: int = field(default_factory=lambda: _envi("DX_CARDLESS_MAX", 2))
    discovery_min_conf: float = field(default_factory=lambda: _envf("DX_DISCOVERY_MIN_CONF", 0.9))
    kb_perturb: float = field(default_factory=lambda: _envf("DX_KB_PERTURB", 1.0))  # robustness check: x/÷ factor, 1 = off
    kb_perturb_seed: int = field(default_factory=lambda: _envi("DX_KB_PERTURB_SEED", 0))

    # Belief state.
    other_mass: float = field(default_factory=lambda: _envf("DX_OTHER_MASS", 0.10))
    max_differential: int = field(default_factory=lambda: _envi("DX_MAX_DIFFERENTIAL", 12))
    drop_below: float = field(default_factory=lambda: _envf("DX_DROP_BELOW", 0.002))
    top_k_load: int = field(default_factory=lambda: _envi("DX_TOP_K_LOAD", 8))
    cant_miss_floor: float = field(default_factory=lambda: _envf("DX_CANT_MISS_FLOOR", 0.01))

    # v0.3 hypothesis-list fixes.
    open_world_enforce: bool = field(default_factory=lambda: _env("DX_OPEN_WORLD", "0") == "1")  # C1: off (no measurable gain, +1.9 LLM calls)
    open_world_max: int = field(default_factory=lambda: _envi("DX_OPEN_WORLD_MAX", 2))  # C1: re-prompts per case

    # v0.3 perception and belief fixes.
    primary_complaint_only: bool = field(default_factory=lambda: _env("DX_PRIMARY_COMPLAINT", "1") == "1")  # A1
    lexicon_vocab: bool = field(default_factory=lambda: _env("DX_LEXICON_VOCAB", "1") == "1")  # A2
    lexicon_k: int = field(default_factory=lambda: _envi("DX_LEXICON_K", 60))

    # Planner and stopping.
    misdiagnosis_loss: float = field(default_factory=lambda: _envf("DX_MISDIAGNOSIS_LOSS", 10.0))
    pair_weight: float = field(default_factory=lambda: _envf("DX_PAIR_WEIGHT", 2.0))
    # "accuracy": every confusion costs the same (benchmark). "clinical": confusions weighted by harm.
    objective: str = field(default_factory=lambda: _env("DX_OBJECTIVE", "accuracy"))
    safety_weight: float = field(default_factory=lambda: _envf("DX_SAFETY_WEIGHT", 1.0))
    cost_weight: float = field(default_factory=lambda: _envf("DX_COST_WEIGHT", 1 / 1000))
    turn_cost: float = field(default_factory=lambda: _envf("DX_TURN_COST", 0.03))
    min_commit_prob: float = field(default_factory=lambda: _envf("DX_MIN_COMMIT_PROB", 0.40))
    voi_epsilon: float = field(default_factory=lambda: _envf("DX_VOI_EPSILON", 0.002))
    commit_threshold: float = field(default_factory=lambda: _envf("DX_COMMIT_THRESHOLD", 0.80))
    exclusion_threshold: float = field(default_factory=lambda: _envf("DX_EXCLUSION_THRESHOLD", 0.03))
    min_turns_before_commit: int = field(default_factory=lambda: _envi("DX_MIN_TURNS", 4))
    max_turns: int = field(default_factory=lambda: _envi("DX_MAX_TURNS", 60))  # hard cap per case; the diagnosis itself is the last turn
    stall_turns: int = field(default_factory=lambda: _envi("DX_STALL_TURNS", 3))  # commit after this many turns with no worthwhile option
    max_batch_history: int = field(default_factory=lambda: _envi("DX_MAX_BATCH_HISTORY", 3))
    max_batch_tests: int = field(default_factory=lambda: _envi("DX_MAX_BATCH_TESTS", 3))

    # D1: build a turn's batch greedily by marginal value given the actions already chosen, scoring joint
    # outcomes with the same cluster pooling the belief update applies.
    conditional_batches: bool = field(default_factory=lambda: _env("DX_CONDITIONAL_BATCHES", "0") == "1")  # D1: off (no measurable gain)

    # Safety critic.
    # D2: a veto must name an alternative; it is resolved by the arithmetic (add it, or separate it from the
    # leader with a positive-value action), is not repeated without new evidence, and has a failsafe ceiling.
    structured_veto: bool = field(default_factory=lambda: _env("DX_STRUCTURED_VETO", "1") == "1")
    veto_ceiling: int = field(default_factory=lambda: _envi("DX_VETO_CEILING", 4))
    veto_no_repeat: bool = field(default_factory=lambda: _env("DX_VETO_NO_REPEAT", "1") == "1")
    # E1: the final diagnosis is the top of the computed differential; a report dissent triggers one
    # re-evaluation with the dissenting diagnosis added as a hypothesis.
    deterministic_final: bool = field(default_factory=lambda: _env("DX_DETERMINISTIC_FINAL", "1") == "1")
    dissent_reevaluate: bool = field(default_factory=lambda: _env("DX_DISSENT_REEVALUATE", "1") == "1")
    # E1 (revised): the dissent is an objection resolved by the arithmetic, like a structured veto. Whether or not
    # the named disease was already on the differential, the next commit is held while an action that separates
    # it from the leader has positive value. Off: re-evaluate only a dissent naming a disease absent from the list.
    dissent_as_objection: bool = field(default_factory=lambda: _env("DX_DISSENT_OBJECTION", "1") == "1")
    critic_entropy_trigger: float = field(default_factory=lambda: _envf("DX_CRITIC_ENTROPY", 0.60))
    critic_every: int = field(default_factory=lambda: _envi("DX_CRITIC_EVERY", 3))

    # v0.3.1 prior context (P1). The prior may condition ONLY on S = {age, sex, care setting, complaint group}.
    # A prior assessor that sees nothing but S assigns every bucket (information hiding, not an instruction);
    # the generator's own bucket is logged for comparison and never used. The primary complaint is chosen by a
    # deterministic rule over LLM-normalized complaint labels; it is the one finding excluded from the update.
    prior_context: bool = field(default_factory=lambda: _env("DX_PRIOR_CONTEXT", "1") == "1")
    prior_model: str = field(default_factory=lambda: _env("DX_PRIOR_MODEL", "deepseek:deepseek-flash"))  # empty: the doctor model
    prior_samples: int = field(default_factory=lambda: _envi("DX_PRIOR_SAMPLES", 3))  # self-consistency, median bucket
    # Card prevalence (keyed by complaint) and the assessor disagree by this many buckets or more: meet in the middle.
    prior_reconcile_steps: int = field(default_factory=lambda: _envi("DX_PRIOR_RECONCILE", 2))
    prior_sensitivity_alpha: float = field(default_factory=lambda: _envf("DX_PRIOR_SENS_ALPHA", 0.5))  # diagnostic only

    # Fixed opening moves before value-of-information planning takes over.
    intake_script: tuple[str, ...] = ("open_hpi", "pmh_meds_social", "vitals_general_exam")


def get_settings() -> Settings:
    return Settings()
