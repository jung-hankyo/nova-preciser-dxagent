"""Data models.

Two families live here:

1. The LangGraph state (`DxState`), a TypedDict of plain JSON-serializable values so
   the checkpointer can persist it.
2. Pydantic models for structured LLM outputs. Every LLM call in the agent returns
   one of these, so no node ever parses free text.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

ActionType = Literal["history", "exam", "poc", "lab", "imaging", "advanced_imaging", "procedure"]
PriorBucket = Literal["very_common", "common", "uncommon", "rare", "very_rare"]
LRBucket = Literal["definitive", "very_strong", "strong", "moderate", "weak", "none"]
HarmBucket = Literal["catastrophic", "high", "moderate", "low"]


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class DxState(TypedDict, total=False):
    # Encounter bookkeeping
    presentation: str  # opening statement or chief complaint
    presentation_source: str  # "patient" (chat) or "referral" (benchmark objective)
    turn: int  # number of doctor turns sent so far
    phase: str  # "gathering" | "committed"
    transcript: list[dict[str, Any]]  # [{role, type, text, turn}]

    # Belief state
    context: dict[str, Any]  # age, sex, setting ...
    problem_rep: str
    findings: list[dict[str, Any]]  # see Finding
    lab_values: list[dict[str, Any]]
    hypotheses: list[dict[str, Any]]  # candidates with prior buckets, rescaled with the list at each update: {disease_id, name, prior, prior_source, cant_miss}
    differential: list[dict[str, Any]]  # posterior-ranked: {disease_id, name, prob, cant_miss}
    other_prob: float
    entropy: float
    unexplained: list[str]
    belief_trace: dict[str, list[str]]
    hypothesis_trace: list[dict[str, Any]]  # per turn: which diseases were on the differential (recall analysis)

    # Planning
    last_actions: list[dict[str, Any]]  # actions sent in the last doctor turn
    done_action_ids: list[str]
    asked_codes: list[str]
    action_log: list[dict[str, Any]]
    cost_spent: float
    critic_notes: list[str]
    critic_block: bool
    critic_turn: int
    critic_added: bool
    commit_reason: str
    stall_count: int
    test_results: list[dict[str, Any]]
    contrast_turn: int
    contrast_added: bool

    # v0.3 state
    unexplained_streak: dict[str, int]  # C3: consecutive turns each clear finding stayed unexplained
    unexplained_turn: int
    cardless_queries: int  # C3: "which diseases cause X?" queries used
    discovery_done: list[str]  # C3: findings already queried
    list_effect_log: list[dict[str, Any]]  # B1 diagnostic: leader change caused by list changes alone
    open_world_reprompts: int  # C1
    vetoes: list[dict[str, Any]]  # D2: every structured veto and how it was resolved
    veto: dict[str, Any]  # D2: the veto honored this turn ({"alternative", "turn"})
    unresolved_objection: dict[str, Any]  # D2: a veto that no positive-value action could resolve
    dissent_checked: bool  # E1
    report_dissent: list[dict[str, Any]]  # E1
    # v0.3.1 prior context (P1)
    prior_context: dict[str, Any]  # S: {age, sex, setting, complaint, complaint_label, quote, rule, acuity, candidates}
    prior_log: list[dict[str, Any]]  # every bucket assignment: route, generator bucket, card bucket, assessor bucket, final
    coverage_log: list[dict[str, Any]]  # v0.4: hypotheses with no knowledge (no own card, no carded ancestor)
    perceive_log: list[dict[str, Any]]  # v0.4.1: extraction diagnostics per reply
    contrast_log: list[dict[str, Any]]  # v0.4.1: contrast pairs requested and whether cards were added
    actions_taken: list[dict[str, Any]]  # v0.4.1: the actions sent each turn

    # I/O with the environment
    doctor_message: str
    last_reply: str
    final_report: dict[str, Any]


class Finding(BaseModel):
    code: str
    label: str = ""
    present: bool
    confidence: float = Field(ge=0.0, le=1.0)
    source: ActionType = "history"
    evidence: str = ""
    turn: int = 0
    chief_complaint: bool = False


# ---------------------------------------------------------------------------
# Structured LLM outputs
# ---------------------------------------------------------------------------


class ExtractedFinding(BaseModel):
    code: str = Field(
        description="snake_case finding code. Reuse a code from the provided vocabulary whenever the meaning matches; "
        "only invent a new code if nothing fits."
    )
    label: str = Field(description="Short human-readable label, e.g. 'pleuritic chest pain'.")
    present: bool = Field(description="True if the finding is present, False if explicitly denied or normal.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How clearly the reply establishes this finding: 0.9+ explicit and unambiguous, "
        "0.6-0.8 probable, below 0.5 vague or hedged.",
    )
    source: ActionType = Field(description="Channel that produced this finding.")
    evidence: str = Field(description="Verbatim fragment of the reply that supports the finding (under 20 words).")
    chief_complaint: bool = Field(
        default=False, description="True only if this finding is the presenting complaint itself."
    )
    primary_complaint: bool = Field(
        default=False,
        description="Opening statement only: true for exactly ONE finding, the single symptom that best defines "
        "why the patient came in (usually the first one mentioned). Every other finding is false.",
    )


class LabValue(BaseModel):
    analyte: str = Field(description="snake_case analyte name, e.g. 'troponin_i', 'd_dimer', 'wbc'.")
    value: float
    unit: str = ""


class ExtractedTest(BaseModel):
    test_id: str = Field(description="Test id from the provided test list.")
    outcome: str = Field(description="One of that test's listed outcomes, exactly.")
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(description="Verbatim fragment of the reply (under 20 words).")


class ExtractedComplaint(BaseModel):
    label: str = Field(description="Short normalized complaint, 1-4 words, as a triage nurse would file it "
                                   "(e.g. 'chest pain', 'shortness of breath', 'double vision', 'weakness').")
    quote: str = Field(description="Verbatim fragment of the opening statement that states this complaint (under 15 words).")
    stated_reason: bool = Field(description="True only if the statement presents this as the reason for the visit "
                                            "('came in because of ...', 'presenting with ...', the first complaint "
                                            "volunteered). Several may be true.")
    finding_code: str = Field(default="", description="Code of the extracted finding that expresses this complaint.")


class ExtractionOutput(BaseModel):
    findings: list[ExtractedFinding] = Field(default_factory=list)
    complaints: list[ExtractedComplaint] = Field(
        default_factory=list,
        description="Opening statement only: every presenting complaint, in order of mention. Empty otherwise.",
    )
    setting: Optional[Literal["emergency", "outpatient", "inpatient", "unknown"]] = Field(
        default=None, description="Care setting if the text states or clearly implies it.")
    test_results: list[ExtractedTest] = Field(default_factory=list)
    lab_values: list[LabValue] = Field(default_factory=list)
    age: Optional[int] = Field(default=None, description="Patient age in years if stated.")
    sex: Optional[Literal["female", "male", "other"]] = None
    problem_representation: str = Field(
        description="One-sentence clinical summary of everything known so far (age, sex, tempo, key findings)."
    )


class ProposedHypothesis(BaseModel):
    name: str = Field(description="Specific diagnosis name, as a clinician would write it.")
    disease_id: str = Field(
        description="snake_case id. If the disease appears in the provided known-disease list, reuse that id exactly."
    )
    prior_bucket: PriorBucket = Field(
        description="Pretest probability GIVEN ONLY demographics and the presenting complaint, before other findings."
    )
    cant_miss: bool = Field(description="True if missing this diagnosis could cause death or serious harm soon.")
    rationale: str = Field(description="One short sentence.")


class HypothesisOutput(BaseModel):
    hypotheses: list[ProposedHypothesis] = Field(
        description="6-10 hypotheses covering common causes, can't-miss causes, and at least one outside the current list."
    )


class CriticOutput(BaseModel):
    missing_hypotheses: list[ProposedHypothesis] = Field(
        default_factory=list, description="Up to 3 diagnoses that are plausible but absent from the differential."
    )
    concerns: list[str] = Field(default_factory=list, description="Anchoring, unexplained findings, or red flags.")
    block_commit: bool = Field(
        default=False, description="True if committing now would be unsafe (a dangerous alternative is not excluded)."
    )
    veto_alternative: str = Field(
        default="",
        description="Only when the agent is about to commit: the ONE specific diagnosis that has not been reasonably "
        "excluded and would make committing now unsafe or wrong. Empty if there is none.",
    )
    veto_reason: Literal["none", "cant_miss_not_excluded", "unexplained_finding", "contradicting_finding",
                         "insufficient_workup"] = Field(default="none", description="Why the alternative blocks the commit.")
    veto_evidence: str = Field(default="", description="The findings the objection rests on (one short sentence).")


class InterviewerOutput(BaseModel):
    message: str = Field(description="The single conversational turn to send to the patient or the clinical team.")


class FreeQuestionOutput(BaseModel):
    question: str = Field(description="One focused history question.")
    target: str = Field(description="What finding or distinction the question is meant to settle.")


class ReportOutput(BaseModel):
    final_diagnosis: str = Field(description="The single most likely specific diagnosis.")
    agrees_with_top: bool = Field(
        default=True, description="False only if the findings clearly contradict the top of the computed differential."
    )
    dissent_diagnosis: str = Field(default="", description="If you disagree: the specific diagnosis you believe instead.")
    dissent_reason: str = Field(default="", description="If you disagree: the findings that contradict the top diagnosis.")
    rationale: str = Field(description="2-4 sentences citing the decisive findings.")
    key_findings: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Knowledge drafting (frequencies, never likelihood ratios)
# ---------------------------------------------------------------------------

FreqLevel = Literal["very_common", "common", "occasional", "uncommon", "rare", "very_rare"]
FREQ_HELP = "very_common ~90%, common ~65%, occasional ~35%, uncommon ~15%, rare ~5%, very_rare ~1%"
TestType = Literal["poc", "lab", "imaging", "advanced_imaging", "procedure"]


class OutcomeLevel(BaseModel):
    outcome: str = Field(description="Outcome name (snake_case). For existing tests use the listed outcomes exactly.")
    level: FreqLevel


class ContrastDraftFinding(BaseModel):
    code: str
    label: str
    source: Literal["history", "exam"]
    freq_in_a: FreqLevel
    freq_in_b: FreqLevel
    background_freq: FreqLevel
    cluster: str


class ContrastDraftTest(BaseModel):
    test_id: str
    name: str
    type: TestType
    outcomes_a: list[OutcomeLevel]
    outcomes_b: list[OutcomeLevel]
    background: list[OutcomeLevel] = Field(default_factory=list, description="Only for a NEW test.")


class ContrastDraft(BaseModel):
    findings: list[ContrastDraftFinding] = Field(description="3-8 history or exam findings whose frequency differs most between A and B.")
    tests: list[ContrastDraftTest] = Field(description="1-4 tests that separate A from B, with outcome distributions in each.")
    management_differs: bool = Field(description="True if A and B are treated differently.")


class LeakTermsOutput(BaseModel):
    """v0.3 (G2): names a simulator could use to give the diagnosis away."""

    terms: list[str] = Field(default_factory=list, description="Synonyms, abbreviations, eponyms, lay names (max 12).")


# ---------------------------------------------------------------------------
# v0.4 knowledge compiler: claims extracted from cited sources (values are computed and verified by code)
# ---------------------------------------------------------------------------

Population = Literal["adult", "pediatric", "mixed", "unknown"]
ClaimCategory = Literal["symptom", "sign", "vital", "risk_factor", "exposure", "medication", "family_history", "lab", "imaging"]


class ExtractedFindingClaim(BaseModel):
    finding: str = Field(description="The finding as named in the source, e.g. 'pleuritic chest pain'.")
    category: ClaimCategory = "symptom"
    quote: str = Field(description="8-60 consecutive words copied exactly from the passage.")
    locator: str = Field(default="", description="The [page/section] tag of the passage.")
    percent: Optional[float] = Field(default=None, description="Percent of patients WITH THIS DISEASE who have it (0-100), as stated.")
    n: Optional[int] = Field(default=None, description="Number of patients with the finding, if stated as n of m.")
    m: Optional[int] = Field(default=None, description="Number of patients in that group, if stated as n of m.")
    sensitivity: Optional[float] = Field(default=None, description="0-100, as stated.")
    specificity: Optional[float] = Field(default=None, description="0-100, as stated.")
    lr_positive: Optional[float] = None
    lr_negative: Optional[float] = None
    population: Population = "unknown"


class ExtractedTestClaim(BaseModel):
    test_id: str = Field(description="A registry test id exactly as listed.")
    positive_outcomes: list[str] = Field(default_factory=list, description="Listed outcome names that count as positive.")
    quote: str
    locator: str = ""
    percent: Optional[float] = Field(default=None, description="Percent of patients with this disease who have the positive outcome(s).")
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    lr_positive: Optional[float] = None
    lr_negative: Optional[float] = None
    population: Population = "unknown"


class ExtractedHarm(BaseModel):
    untreated_course: Optional[Literal["lethal", "serious", "self_limited"]] = None
    urgency: Optional[Literal["hours", "days", "weeks"]] = None
    treatments: list[str] = Field(default_factory=list, description="Treatment classes, snake_case.")
    treatment_risk: Optional[Literal["high", "moderate", "low"]] = None
    quote: str = ""
    locator: str = ""


class ExtractedMimic(BaseModel):
    name: str
    quote: str = ""
    locator: str = ""


class ClaimExtraction(BaseModel):
    findings: list[ExtractedFindingClaim] = Field(default_factory=list)
    tests: list[ExtractedTestClaim] = Field(default_factory=list)
    harm: Optional[ExtractedHarm] = None
    mimics: list[ExtractedMimic] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Benchmark judge
# ---------------------------------------------------------------------------


class JudgeOutput(BaseModel):
    top1_correct: bool = Field(description="Does the doctor's final diagnosis name the same disease as the reference?")
    match_rank: Optional[int] = Field(
        default=None, description="1-based rank of the first differential entry matching the reference, or null."
    )
    rationale: str = ""


# --------------------------------------------------------------------------- v0.3.1 prior context (P1)


class ComplaintMatchOutput(BaseModel):
    same_as: str = Field(default="", description="The listed existing complaint that means the same thing, or empty.")
    acuity: int = Field(default=3, ge=1, le=5, description="Typical triage acuity of this complaint in adults "
                                                         "(1 most urgent, 5 least), as in KTAS/CTAS.")


class PriorItem(BaseModel):
    disease_id: str
    prior_bucket: PriorBucket


class PriorAssessment(BaseModel):
    items: list[PriorItem] = Field(default_factory=list)
