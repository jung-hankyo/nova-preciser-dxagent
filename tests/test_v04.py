"""v0.4 knowledge compiler and ontology tree. Tiny inline ontologies; no network, no API key."""

import gzip
import json
import math

import pytest

from dxagent.knowledge.cards import (CardStore, DiseaseCard, EntryProvenance, FindingEntry, HarmProfile, Provenance,
                                     card_to_row, row_to_card)
from dxagent.knowledge.compiler.assemble import VocabularyMapper, build_card, derive_backgrounds, reconcile
from dxagent.knowledge.compiler.claims import Claim, numeric_value, qualitative_value, sens_spec_from_lr
from dxagent.knowledge.compiler.compiler import KnowledgeCompiler
from dxagent.knowledge.compiler.fetch import fetch
from dxagent.knowledge.compiler.hpo import hpo_claims, parse_frequency
from dxagent.knowledge.compiler.sources import Manifest, SourceDoc, entry_conf
from dxagent.knowledge.compiler.verify import norm_text, span_found, verify_claim
from dxagent.knowledge.ontology import OntologyIndex, OntologySet, build_index
from dxagent.llm import ScriptedLLM
from dxagent.reasoning.belief import compute_posterior, prune_hypotheses, tree_parents, tree_priors
from dxagent.reasoning.planner import view_groups
from dxagent.schemas import ClaimExtraction, ExtractedFindingClaim, ExtractedTestClaim

MONDO_OBO = """format-version: 1.2
data-version: releases/test

[Term]
id: MONDO:0000001
name: disease

[Term]
id: MONDO:0000100
name: lung disorder
is_a: MONDO:0000001 ! disease

[Term]
id: MONDO:0005249
name: pneumonia
synonym: "lung infection" EXACT []
is_a: MONDO:0000100 ! lung disorder

[Term]
id: MONDO:0005824
name: Legionnaires' disease
synonym: "Legionella pneumonia" EXACT []
is_a: MONDO:0005249 ! pneumonia

[Term]
id: MONDO:0009688
name: myasthenia gravis
synonym: "MG" RELATED []
xref: OMIM:254200 {source="MONDO:equivalentTo"}
xref: Orphanet:589 {source="MONDO:equivalentTo"}
is_a: MONDO:0000001 ! disease

[Term]
id: MONDO:0009999
name: old pneumonia term
is_obsolete: true
replaced_by: MONDO:0005249
"""

HP_OBO = """format-version: 1.2
data-version: hp/releases/test

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0000001
name: Abnormality of the neuromuscular system
is_a: HP:0000118

[Term]
id: HP:0000002
name: Abnormality of muscle physiology
is_a: HP:0000001

[Term]
id: HP:0003324
name: Generalized muscle weakness
is_a: HP:0000002

[Term]
id: HP:0001324
name: Muscle weakness
synonym: "weakness of muscles" EXACT []
is_a: HP:0003324

[Term]
id: HP:0000508
name: Ptosis
synonym: "drooping eyelid" EXACT []
is_a: HP:0000002

[Term]
id: HP:0000651
name: Diplopia
synonym: "double vision" EXACT []
is_a: HP:0000002

[Term]
id: HP:0001945
name: Fever
is_a: HP:0000118
"""

HPOA = """#description: test
#version: 2026-09-02
database_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\tonset\tfrequency\tsex\tmodifier\taspect\tbiocuration
ORPHA:589\tMyasthenia gravis\t\tHP:0001324\tORPHA:589\tTAS\t\tHP:0040281\t\t\tP\tx
ORPHA:589\tMyasthenia gravis\t\tHP:0000508\tORPHA:589\tTAS\t\tHP:0040282\t\t\tP\tx
ORPHA:589\tMyasthenia gravis\t\tHP:0000651\tORPHA:589\tTAS\t\tHP:0040282\t\t\tP\tx
ORPHA:589\tMyasthenia gravis\tNOT\tHP:0001945\tORPHA:589\tTAS\t\t\t\t\tP\tx
OMIM:254200\tMyasthenia gravis\t\tHP:0000508\tPMID:1\tPCS\t\t2/3\t\t\tP\tx
OMIM:254200\tMyasthenia gravis\t\tHP:0000001\tPMID:1\tPCS\t\t\t\t\tI\tx
"""

DOC = """Community-acquired pneumonia: guideline summary.

Clinical features. Fever was present in 78% of adults with pneumonia in the derivation cohort.
Pleuritic chest pain occurs frequently in pneumonia, while hemoptysis is rare.
A chest radiograph showed consolidation with a sensitivity of 80% for pneumonia.
"""


@pytest.fixture
def onto(tmp_path):
    (tmp_path / "mondo.obo").write_text(MONDO_OBO)
    (tmp_path / "hp.obo").write_text(HP_OBO)
    build_index(tmp_path / "mondo.obo", tmp_path / "mondo_index.json.gz", "MONDO", "MONDO:0000001")
    build_index(tmp_path / "hp.obo", tmp_path / "hpo_index.json.gz", "HP", "HP:0000118")
    return OntologySet.load_dir(tmp_path)


# ------------------------------------------------------------------ ontology


def test_ontology_resolution_ancestry_and_redirects(onto):
    d = onto.disease
    assert d.resolve("Pneumonia") == ("MONDO:0005249", "exact", 1.0)
    assert d.resolve("Legionella pneumonia")[0] == "MONDO:0005824"
    assert d.resolve("MG")[:2] == ("MONDO:0009688", "synonym")  # RELATED synonym, unique
    assert d.resolve("MONDO:0009999")[0] == "MONDO:0005249"  # obsolete id redirects
    assert d.resolve("zebra fever")[0] is None
    assert d.ancestors("MONDO:0005824") == {"MONDO:0005249": 1, "MONDO:0000100": 2, "MONDO:0000001": 3}
    assert d.xref_map("Orphanet")["Orphanet:589"] == ["MONDO:0009688"]
    assert onto.phenotype.depth("HP:0001324") == 4


def test_local_terms_extend_the_disease_index(onto, settings, tmp_path):
    lt = tmp_path / "local.yaml"
    lt.write_text('- {id: "DXA:0000099", name: aspiration pneumonitis, synonyms: [aspiration pneumonia], parents: ["MONDO:0000100"]}\n')
    store = CardStore(settings.registry_dir, ontology=onto, local_terms=lt)
    assert store.resolve_disease("aspiration pneumonia") == ("dxa_0000099", "DXA:0000099", "exact")
    assert "mondo_0000100" in store.ancestors("dxa_0000099")


# ------------------------------------------------------------------ claims and verification


def test_values_are_computed_from_stated_numbers_only():
    assert numeric_value(percent=78)[:3] == (0.78, None, "percent")
    p, _, how, size = numeric_value(n=1, m=1)
    assert how == "n_of_m" and p == 0.75 and size == 1  # 1/1 never becomes certainty
    p, bg, how, _ = numeric_value(sensitivity=96, specificity=40)
    assert (p, round(bg, 2), how) == (0.96, 0.6, "sensitivity")
    sens, spec = sens_spec_from_lr(5.0, 0.2)
    assert abs(sens / (1 - spec) - 5.0) < 1e-9 and abs((1 - sens) / spec - 0.2) < 1e-9
    assert qualitative_value("hemoptysis is rare")[0] == 0.05
    assert qualitative_value("this is not uncommon")[0] is None  # negated
    assert qualitative_value("rarely seen; usually common in smokers")[0] is None  # conflicting
    q = "Pleuritic chest pain occurs frequently in pneumonia, while hemoptysis is rare"
    assert qualitative_value(q, ["hemoptysis"])[0] == 0.05 and qualitative_value(q, ["pleuritic chest pain"])[0] == 0.60


def test_verification_rejects_paraphrase_wrong_numbers_and_wrong_subject():
    src = norm_text("Fever was present in 78% of adults with pneumo-\nnia in the derivation cohort.")
    assert span_found("fever was present in 78% of adults with pneumonia", src)
    assert verify_claim("Fever was present in 78% of adults with pneumonia", src, ["fever"], {"percent": 78}) is None
    assert verify_claim("Fever was present in 0.78 of adults with pneumonia", src, ["fever"], {"percent": 78}) is not None
    assert "not stated" in verify_claim("Fever was present in 78% of adults with pneumonia", src, ["fever"], {"percent": 87})
    assert "verbatim" in verify_claim("Most adults with pneumonia had a fever on admission", src, ["fever"], {})
    assert "mention" in verify_claim("Fever was present in 78% of adults with pneumonia", src, ["hemoptysis"], {"percent": 78})


def test_confidence_comes_from_tier_and_precision():
    assert entry_conf(1, "numeric", 1.0) == 0.95 and entry_conf(3, "qualitative", 1.0) == 0.55
    assert entry_conf(2, "class", 0.8) == 0.6


# ------------------------------------------------------------------ HPO


def test_hpo_annotations_become_claims(onto, tmp_path):
    path = tmp_path / "p.hpoa"
    path.write_text(HPOA)
    src = SourceDoc(id="hpo_annotations", kind="curated_db")
    out, stats = hpo_claims(path, src, onto.disease, onto.phenotype, min_annotations=3, propagate_depth=2, min_term_depth=2)
    cs = out["MONDO:0009688"]
    by = {}
    for c in cs:
        by.setdefault(c.hpo_id, []).append(c)
    assert by["HP:0001324"][0].p == 0.90 and by["HP:0001324"][0].precision == "class"
    assert by["HP:0001945"][0].p == 0.01  # NOT: a pertinent negative
    ptosis = sorted(by["HP:0000508"], key=lambda c: c.locator)
    assert {c.precision for c in ptosis} == {"class"}  # 2/3 is a small series: treated as a class
    assert "HP:0003324" in by and by["HP:0003324"][0].extra["propagated_from"] == "HP:0001324"  # true-path, depth 1
    assert "HP:0000001" not in by  # inheritance-aspect row skipped; too general to propagate into
    assert parse_frequency("12%") == (0.12, "numeric", None)


# ------------------------------------------------------------------ reconciliation and assembly


def _claim(**kw):
    base = dict(disease="MONDO:0005249", kind="finding", source_id="s", source_kind="guideline", tier=1, code="fever",
                label="fever", p=0.7, precision="numeric", edition="2020")
    base.update(kw)
    return Claim(**base)


def test_higher_tier_wins_and_conflicts_are_recorded_not_averaged(settings):
    best = _claim(source_id="guideline_a", tier=1, p=0.7)
    weak = _claim(source_id="case_series", source_kind="primary", tier=3, p=0.1)
    newer = _claim(source_id="guideline_b", tier=1, p=0.72, edition="2023")
    chosen = reconcile([weak, best, newer])
    c, conflicts = chosen[("finding", "fever")]
    assert c.source_id == "guideline_b"  # same tier and precision: newer edition
    assert [x.source_id for x in conflicts] == ["case_series"]
    store = CardStore(settings.registry_dir)
    card, notes = build_card("MONDO:0005249", None, [weak, best, newer], VocabularyMapper(store, None), store, "t")
    fe = card.finding_map["fever"]
    assert fe.p == 0.72 and fe.prov.conflicts[0]["source_id"] == "case_series"
    assert fe.conf == entry_conf(1, "numeric", 1.0)  # a conflict from a WORSE tier does not discount


def test_test_claims_build_an_outcome_distribution(settings):
    store = CardStore(settings.registry_dir)
    c = _claim(kind="test", test_id="img_chest_xray", positive=["consolidation"], p=0.8, code="")
    card, _ = build_card("MONDO:0005249", None, [c, _claim()], VocabularyMapper(store, None), store, "t")
    dist = card.tests["img_chest_xray"].dist
    assert abs(dist["consolidation"] - 0.8) < 1e-9 and abs(sum(dist.values()) - 1) < 1e-9


def test_derived_background_caps_rare_cards(settings):
    store = CardStore(settings.registry_dir)
    mapper = VocabularyMapper(store, None)
    code = mapper.for_label("new rare sign", "sign")
    def card(cid, kind, p, prev):
        return DiseaseCard(id=cid, name=cid, findings=[FindingEntry(code=code, p=p, conf=0.8)], base_prevalence=prev,
                           provenance=Provenance(sources=[{"id": "x", "kind": kind}]))
    rare = [card(f"r{i}", "curated_db", 0.9, "very_rare") for i in range(100)]
    common = [card("c1", "guideline", 0.0, "common"), card("c2", "guideline", 0.0, "common")]
    bg = derive_backgrounds(rare + common, mapper.new_defs, rare_share=0.05, bg_min=0.02, bg_max=0.5)[code]
    assert abs(bg - 0.05 * 0.9) < 2e-3  # 100 rare cards together weigh 5%, not 100x a common card
    assert mapper.new_defs[code].background_source == "derived"


def test_compact_rows_round_trip(settings):
    store = CardStore(settings.registry_dir)
    prov = EntryProvenance(source_id="esc", kind="guideline", tier=1, precision="numeric", edition="2019", locator="p.12",
                           quote="fever in 20%", population="adult")
    card = DiseaseCard(id="mondo_0005279", name="pulmonary embolism", ontology_id="MONDO:0005279",
                       findings=[FindingEntry(code="fever", p=0.2, conf=0.95, prov=prov)], harm=HarmProfile(untreated_course="lethal", evidence="esc"),
                       provenance=Provenance(sources=[{"id": "esc", "edition": "2019", "kind": "guideline"}]))
    back = row_to_card(json.loads(json.dumps(card_to_row(card))), store._findings)
    f = back.finding_map["fever"]
    assert (f.p, f.conf, f.prov.locator, f.prov.population, f.prov.edition) == (0.2, 0.95, "p.12", "adult", "2019")
    assert back.cant_miss is True and back.harm.evidence == "esc" and back.ontology_id == "MONDO:0005279"


# ------------------------------------------------------------------ compiler end to end


def test_compile_end_to_end_with_verified_document_claims(tmp_path, settings, onto):
    src_dir = tmp_path / "sources"
    for name, text in (("mondo.obo", MONDO_OBO), ("hp.obo", HP_OBO), ("p.hpoa", HPOA)):
        (tmp_path / name).write_text(text)
    fetch(src_dir, settings.kb_dir, local={"mondo": tmp_path / "mondo.obo", "hpo": tmp_path / "hp.obo",
                                            "hpo_annotations": tmp_path / "p.hpoa"}, progress=lambda *a: None)
    (src_dir / "docs").mkdir()
    (src_dir / "docs" / "cap.txt").write_text(DOC)
    m = Manifest.load(src_dir / "manifest.yaml")
    m.sources.append(SourceDoc(id="cap_guideline", kind="guideline", edition="2019", path="docs/cap.txt",
                               diseases=["pneumonia"], population="adult"))
    m.save(src_dir / "manifest.yaml")
    settings.sources_dir, settings.hpo_min_annotations, settings.hpo_min_term_depth = src_dir, 3, 2
    calls = []

    def extract(system, user):
        calls.append(user)
        return ClaimExtraction(
            findings=[ExtractedFindingClaim(finding="fever", category="symptom", percent=78, locator="sec.1", population="adult",
                                            quote="Fever was present in 78% of adults with pneumonia in the derivation cohort"),
                      ExtractedFindingClaim(finding="hemoptysis", category="symptom", locator="sec.1",
                                            quote="Pleuritic chest pain occurs frequently in pneumonia, while hemoptysis is rare"),
                      ExtractedFindingClaim(finding="cough", category="symptom", percent=90, locator="sec.1",
                                            quote="Cough was present in 90% of adults with pneumonia"),  # fabricated
                      ExtractedFindingClaim(finding="fever", category="symptom", percent=87, locator="sec.1",
                                            quote="Fever was present in 78% of adults with pneumonia in the derivation cohort")],
            tests=[ExtractedTestClaim(test_id="img_chest_xray", positive_outcomes=["consolidation"], sensitivity=80, locator="sec.1",
                                      quote="A chest radiograph showed consolidation with a sensitivity of 80% for pneumonia")])

    res = KnowledgeCompiler(settings, ScriptedLLM({ClaimExtraction: extract}), progress=lambda *a: None).compile()
    assert res["build"]["counts"]["cards"] == 2  # pneumonia (document) and myasthenia gravis (HPO)
    reasons = {r["item"]: r["reason"] for r in res["report"]["rejected"]}
    assert "verbatim" in reasons["cough"]
    with gzip.open(settings.kb_dir / "cards.jsonl.gz", "rt") as fh:
        assert len(fh.readlines()) == 2
    store = CardStore(settings.registry_dir, kb_dir=settings.kb_dir)
    pna = store.get("pneumonia")
    fever = pna.finding_map["fever"]
    assert fever.p == 0.78 and fever.prov.source_id == "cap_guideline" and fever.prov.tier == 1 and fever.conf == 0.95
    assert pna.finding_map["hemoptysis"].p == 0.05 and pna.finding_map["hemoptysis"].prov.precision == "qualitative"
    assert abs(pna.tests["img_chest_xray"].dist["consolidation"] - 0.8) < 1e-4  # rows store 5 decimals
    mg = store.get("myasthenia gravis")
    assert mg.base_prevalence == "very_rare" and mg.provenance.status == "compiled"
    build = json.loads((settings.kb_dir / "build.json").read_text())
    assert {s["id"] for s in build["sources"] if s["used"]} == {"cap_guideline", "hpo_annotations"}
    # recompiling without an LLM reuses the cached extraction: same cards, no new call
    n = len(calls)
    res2 = KnowledgeCompiler(settings, None, progress=lambda *a: None).compile()
    assert len(calls) == n and res2["build"]["counts"]["cards"] == 2


# ------------------------------------------------------------------ inheritance and the tree


@pytest.fixture
def tree_store(settings, onto):
    pna = DiseaseCard(id="mondo_0005249", name="pneumonia", ontology_id="MONDO:0005249",
                      findings=[FindingEntry(code="fever", p=0.8, conf=1.0), FindingEntry(code="productive_cough", p=0.7, conf=1.0)])
    leg = DiseaseCard(id="mondo_0005824", name="Legionnaires' disease", ontology_id="MONDO:0005824",
                      findings=[FindingEntry(code="fever", p=0.95, conf=1.0)])
    store = CardStore(settings.registry_dir, ontology=onto, inherit_depth=3, inherit_decay=0.9)
    store._register(pna)
    return store, leg


def test_cardless_child_inherits_and_own_entries_override(tree_store):
    store, leg = tree_store
    assert store.get("mondo_0005824") is None and store.inherited_from("mondo_0005824") == "mondo_0005249"
    p, conf = store.finding_p("mondo_0005824", "productive_cough")
    assert p == 0.7 and abs(conf - 0.9) < 1e-9  # one level: x0.9
    store._register(leg)
    assert store.finding_p("mondo_0005824", "fever") == (0.95, 1.0)  # own entry wins
    assert store.finding_p("mondo_0005824", "productive_cough")[0] == 0.7  # gap still inherited


def test_tree_priors_parent_and_child_share_one_budget(tree_store, settings):
    store, leg = tree_store
    store._register(leg)
    hyps = [{"disease_id": "mondo_0005249", "name": "pneumonia", "prior": 0.08, "cant_miss": False},
            {"disease_id": "mondo_0005824", "name": "LD", "prior": 0.08, "cant_miss": False},
            {"disease_id": "x_other", "name": "other", "prior": 0.08, "cant_miss": False}]
    parents = tree_parents(hyps, store)
    assert parents == {"mondo_0005249": None, "mondo_0005824": "mondo_0005249", "x_other": None}
    pri = tree_priors(hyps, parents, floor=0.1)
    assert abs(pri["mondo_0005249"] + pri["mondo_0005824"] - 0.08) < 1e-12 and pri["mondo_0005249"] >= 0.008 - 1e-12
    res = compute_posterior(hyps, [], store, settings)
    d = {x["disease_id"]: x for x in res["differential"]}
    assert d["mondo_0005824"]["tree_parent"] == "mondo_0005249"
    assert abs(d["mondo_0005249"]["subtree_prob"] - (d["mondo_0005249"]["prob"] + d["mondo_0005824"]["prob"])) < 1e-12
    settings.tree = False
    flat = compute_posterior(hyps, [], store, settings)["differential"]
    assert all(x["tree_parent"] is None for x in flat)


def test_prune_keeps_the_parent_of_a_kept_child(settings):
    diff = [{"disease_id": "child", "prob": 0.5, "cant_miss": False, "tree_parent": "parent"},
            {"disease_id": "parent", "prob": 0.0001, "cant_miss": False, "tree_parent": None}]
    kept = prune_hypotheses([{"disease_id": "child"}, {"disease_id": "parent"}], diff, settings)
    assert {h["disease_id"] for h in kept} == {"child", "parent"}


def test_planner_view_separates_siblings_only_once_parent_leads():
    diff = [{"disease_id": "a1", "prob": 0.25, "tree_parent": "A"}, {"disease_id": "a2", "prob": 0.25, "tree_parent": "A"},
            {"disease_id": "B", "prob": 0.35, "tree_parent": None}, {"disease_id": "A", "prob": 0.05, "tree_parent": None}]
    groups = view_groups(diff)  # subtree A = 0.55 leads: expanded into A, a1, a2
    assert sorted(map(sorted, groups)) == [[0], [1], [2], [3]]
    diff[2]["prob"] = 0.70
    for i in (0, 1):
        diff[i]["prob"] = 0.1
    groups = view_groups(diff)  # B leads: subtree A stays collapsed
    assert sorted(map(sorted, groups)) == [[0, 1, 3], [2]]
    flat = [{"disease_id": x, "prob": p, "tree_parent": None} for x, p in (("a", 0.5), ("b", 0.3))]
    assert view_groups(flat) == [[0], [1]]  # no ontology relations: the v0.3 objective


# ------------------------------------------------------------------ run time


def test_live_fallback_is_off_by_default_and_provisional_when_on(settings, tmp_path, onto):
    from dxagent.config import Settings
    from dxagent.knowledge.compiler.online import LiveFallback

    assert Settings().live_fallback is False
    store = CardStore(settings.registry_dir, tmp_path / "cache", ontology=onto)
    abstract = "Fever was present in 78% of adults with pneumonia in the derivation cohort."

    def http(url):
        if "esearch" in url:
            return json.dumps({"esearchresult": {"idlist": ["123"]}}).encode()
        return (f"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article><ArticleTitle>T</ArticleTitle>"
                f"<Abstract><AbstractText>{abstract}</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>"
                f"</PubmedArticleSet>").encode()

    llm = ScriptedLLM({ClaimExtraction: lambda s, u: ClaimExtraction(findings=[ExtractedFindingClaim(
        finding="fever", percent=78, quote="Fever was present in 78% of adults with pneumonia")])})
    card = LiveFallback(settings, llm, store, http=http).compile_one("mondo_0005249", "MONDO:0005249")
    assert card.provenance.status == "provisional" and card.finding_map["fever"].conf <= settings.live_fallback_conf_cap
    assert not store.index_eligible(card) and (tmp_path / "cache" / "provisional" / "mondo_0005249.yaml").exists()


def test_hypotheses_resolve_through_the_ontology_and_misses_are_logged(settings, onto):
    from dxagent.graph.nodes import Nodes
    from dxagent.runtime import Runtime
    from dxagent.schemas import ProposedHypothesis

    store = CardStore(settings.registry_dir, ontology=onto)
    store._register(DiseaseCard(id="mondo_0005249", name="pneumonia", ontology_id="MONDO:0005249",
                                findings=[FindingEntry(code="fever", p=0.8, conf=1.0)]))
    n = Nodes(Runtime(settings=settings, doctor=ScriptedLLM({}), store=store))
    kw = {"cant_miss": False, "rationale": "x"}
    props = [ProposedHypothesis(name="Legionella pneumonia", disease_id="legionella_pneumonia", prior_bucket="uncommon", **kw),
             ProposedHypothesis(name="Lung infection", disease_id="lung_infection", prior_bucket="common", **kw),
             ProposedHypothesis(name="Myasthenia gravis", disease_id="mg", prior_bucket="rare", **kw),
             ProposedHypothesis(name="Zebra fever", disease_id="zebra_fever", prior_bucket="rare", **kw)]
    hyps, added = n._merge_hypotheses([], props, 1, "generator")
    ids = {h["disease_id"]: h for h in hyps}
    assert set(ids) == {"mondo_0005824", "mondo_0005249", "mondo_0009688", "zebra_fever"}  # 'lung infection' = pneumonia
    assert ids["zebra_fever"]["resolution"] == "unresolved"
    settings.prior_context = False
    upd = n.load_knowledge({"hypotheses": hyps, "turn": 1})
    missed = {c["disease_id"] for c in upd["coverage_log"]}
    assert missed == {"mondo_0009688", "zebra_fever"}  # Legionnaires' inherits pneumonia's card


# ------------------------------------------------------------------ v0.4.1 local-document fallback


def _local_setup(tmp_path, settings, onto, scoped):
    from dxagent.knowledge.compiler.local import LocalDocumentIndex, LocalFallback

    src = tmp_path / "sources"
    (src / "docs").mkdir(parents=True)
    (src / "docs" / "ref.txt").write_text(DOC)
    m = Manifest()
    m.sources.append(SourceDoc(id="ref_text", kind="reference", edition="2024", path="docs/ref.txt",
                               diseases=["pneumonia"] if scoped else [], population="adult"))
    m.save(src / "manifest.yaml")
    settings.sources_dir = src
    store = CardStore(settings.registry_dir, tmp_path / "cache", ontology=onto)
    calls = []

    def extract(system, user):
        calls.append(user)
        return ClaimExtraction(findings=[ExtractedFindingClaim(
            finding="fever", percent=78, locator="sec.1", population="adult",
            quote="Fever was present in 78% of adults with pneumonia in the derivation cohort")])

    fb = LocalFallback(settings, ScriptedLLM({ClaimExtraction: extract}), store,
                       LocalDocumentIndex(src, tmp_path / "pc"))
    return store, fb, calls


@pytest.mark.parametrize("scoped", [True, False])
def test_local_fallback_compiles_an_on_demand_card_from_downloaded_sources(tmp_path, settings, onto, scoped):
    store, fb, calls = _local_setup(tmp_path, settings, onto, scoped)
    card = fb.compile_one("mondo_0005249", "MONDO:0005249")
    assert card is not None and card.provenance.status == "on_demand"
    f = card.finding_map["fever"]
    assert f.p == 0.78 and f.conf == entry_conf(2, "numeric", 1.0)  # tier-table confidence, no cap
    assert store.index_eligible(card) and (tmp_path / "cache" / "on_demand" / "mondo_0005249.yaml").exists()
    assert fb.compile_one("mondo_0005249", "MONDO:0005249") is None and len(calls) == 1  # searched once per run
    # a disease the documents never name finds nothing and makes no LLM call
    assert fb.compile_one("mondo_0009688", "MONDO:0009688") is None and len(calls) == 1


def test_fallbacks_run_local_first_then_online(settings, onto):
    from dxagent.graph.nodes import Nodes
    from dxagent.runtime import Runtime

    order = []

    class FB:
        def __init__(self, name, card):
            self.name, self.card = name, card

        def compile_one(self, did, oid):
            order.append(self.name)
            return self.card

    store = CardStore(settings.registry_dir, ontology=onto)
    hit = DiseaseCard(id="mondo_0005249", name="pneumonia", provenance=Provenance(status="on_demand"),
                      findings=[FindingEntry(code="fever", p=0.8, conf=0.9)])
    rt = Runtime(settings=settings, doctor=ScriptedLLM({}), store=store, fallbacks=[FB("local", hit), FB("online", None)])
    settings.prior_context = False
    upd = Nodes(rt).load_knowledge({"hypotheses": [{"disease_id": "mondo_0005249", "name": "pneumonia", "prior": 0.08,
                                                     "cant_miss": False, "ontology_id": "MONDO:0005249"}], "turn": 1})
    assert order == ["local"]  # the online fallback is not reached when the local one succeeds
    assert upd["coverage_log"][0]["fallback"] == "on_demand"
