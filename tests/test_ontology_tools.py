"""Executable checks against the real supplied ontology plus isolated fixtures."""

import hashlib
from pathlib import Path
import unittest

from rdflib import Literal, RDF, RDFS
from ontology_tools import (DEFAULT_ONTOLOGY, NS, TOPIC_IRIS, OntologyError, OntologyTools, limited_closure,
                            pipeline_base_graph)
from fakes import demo_fixture
from run_queries import with_test_assertions

SOURCE_EVIDENCE = Path(__file__).resolve().parents[1] / "queries" / "source_evidence.rq"


def labels(result):
    return [{key: term.get("label", term["value"]) for key, term in row.items()}
            for row in result["rows"]]


class OntologyTests(unittest.TestCase):
    def setUp(self):
        self.tools = OntologyTools()

    def test_q1_needs_inverse_inference(self):
        self.assertEqual(labels(self.tools.run_query("Q1", use_inference=False)), [])
        self.assertEqual(labels(self.tools.run_query("Q1")), [{"person": "chef1"}, {"person": "waitress1"}])

    def test_q2_and_q3_match_report(self):
        self.assertEqual(labels(self.tools.run_query("Q2")), [{"company": "OtherCompany"}])
        self.assertEqual(labels(self.tools.run_query("Q3")),
                         [{"winner": "team1", "loser": "team2", "ritual": "HalftimeShow"}])

    def test_q4_needs_symmetric_inference(self):
        self.assertEqual(labels(self.tools.run_query("Q4", use_inference=False)), [])
        self.assertEqual(labels(self.tools.run_query("Q4")), [{"occupation": "president", "friend": "Person1"}])

    def test_empty_assessments_remain_unknown(self):
        self.assertEqual(labels(self.tools.run_query("Q5")), [])
        self.assertEqual(self.tools.get_claim_assessments("UnassessedClaim")["status"], "no_assessment")

    def test_assessments_preserve_disagreeing_sources(self):
        # Synthetic test data stays in memory; it is not evidence about real tweets.
        graph = self.tools.asserted
        graph.add((NS.TestClaim, NS.hasText, Literal("Synthetic claim")))
        for name, source, valid in [("AssessmentA", "SourceA", False), ("AssessmentB", "SourceB", True)]:
            graph.add((NS[name], RDF.type, NS.ClaimAssessment))
            graph.add((NS[name], NS.assessmentOf, NS.TestClaim))
            graph.add((NS[name], NS.assessedBy, NS[source]))
            graph.add((NS[name], NS.isValid, Literal(valid)))
        self.tools.inferred = limited_closure(graph)
        self.assertEqual(labels(self.tools.run_query("Q5")),
                         [{"claim": "TestClaim", "text": "Synthetic claim", "source": "SourceA"}])
        assessments = labels(self.tools.get_claim_assessments("TestClaim"))
        self.assertEqual({row["valid"] for row in assessments}, {"true", "false"})

    def test_q6_isolated_conflict_and_no_mutation(self):
        before_bytes = hashlib.sha256(DEFAULT_ONTOLOGY.read_bytes()).hexdigest()
        before_graph = set(self.tools.asserted)
        self.assertEqual(labels(self.tools.run_query("Q6")), [])
        candidates = [{"entity": "ExampleCondition", "class": cls} for cls in ["Allergy", "Injury"]]
        result = self.tools.check_candidate_types(candidates)
        self.assertEqual(result["status"], "conflicts_found")
        self.assertEqual(labels(result), [{"entity": "ExampleCondition", "class1": "Allergy", "class2": "Injury"}])
        self.assertEqual(self.tools.check_candidate_types(candidates[:1])["rows"], [])
        self.assertEqual(set(self.tools.asserted), before_graph)
        self.assertEqual(hashlib.sha256(DEFAULT_ONTOLOGY.read_bytes()).hexdigest(), before_bytes)

    def test_q6_checks_subclasses_and_transitivity_is_supported(self):
        graph = self.tools.asserted
        graph.add((NS.TestAllergy, RDFS.subClassOf, NS.Allergy))
        graph.add((NS.TestCondition, RDF.type, NS.TestAllergy))
        graph.add((NS.TestCondition, RDF.type, NS.Injury))
        graph.add((NS.A, NS.partOf, NS.B))
        graph.add((NS.B, NS.partOf, NS.C))
        self.tools.inferred = limited_closure(graph)
        self.assertIn((NS.A, NS.partOf, NS.C), self.tools.inferred)
        self.assertIn((NS.TestCondition, RDF.type, NS.Allergy), self.tools.inferred)
        self.assertEqual(labels(self.tools.run_query("Q6"))[0]["entity"], "TestCondition")

    def test_q7_and_q8_need_transitive_and_subclass_inference_on_a_copy(self):
        before = hashlib.sha256(DEFAULT_ONTOLOGY.read_bytes()).hexdigest()
        examples = with_test_assertions()
        self.assertEqual(labels(self.tools.run_query("Q7")), [])
        self.assertEqual(labels(examples.run_query("Q7", use_inference=False)), [])
        self.assertEqual(labels(examples.run_query("Q7")), [{"branch": "branch1", "place": "Netherlands"}])
        self.assertEqual(labels(examples.run_query("Q8", use_inference=False)), [])
        self.assertEqual(labels(examples.run_query("Q8")), [
            {"ad": "FishBurgerAd", "product": "FishBurger", "company": "McDonalds"},
            {"ad": "TacoBellAd", "product": "TacoBellProduct", "company": "TacoBell", "event": "SuperBowl1"}])
        self.assertEqual(hashlib.sha256(DEFAULT_ONTOLOGY.read_bytes()).hexdigest(), before)

    def test_q9_finds_sports_claims_through_subclass(self):
        self.assertEqual(labels(self.tools.run_query("Q9", use_inference=False)), [{"claim": "claim1"}])
        self.assertEqual(labels(self.tools.run_query("Q9")), [{"claim": "claim1"}, {"claim": "claim2"}])

    def test_entity_evidence_marks_inferred_relations(self):
        result = self.tools.get_entity_facts("chef1")
        works_for = [t for t in result["triples"] if t["predicate"].get("label") == "worksFor"]
        self.assertEqual(works_for[0]["origin"], "inferred")
        self.assertEqual(self.tools.get_entity_facts("UnknownPerson")["status"], "unknown_entity")

    def test_pipeline_base_excludes_example_evidence_and_never_mutates_the_file(self):
        before = hashlib.sha256(DEFAULT_ONTOLOGY.read_bytes()).hexdigest()
        base = pipeline_base_graph(self.tools)
        for cls in (NS.Tweet, NS.Claim, NS.ClaimAssessment, NS.Summary):
            for subject in self.tools.inferred.subjects(RDF.type, cls):
                self.assertFalse(set(base.triples((subject, None, None))), subject)
        self.assertLess(len(base), len(self.tools.asserted))
        self.assertIn((NS.Person, RDF.type, self.tools.asserted.value(NS.Person, RDF.type)), base)
        self.assertEqual(hashlib.sha256(DEFAULT_ONTOLOGY.read_bytes()).hexdigest(), before)

    def test_entity_names_are_validated_not_interpolated(self):
        self.assertEqual(OntologyTools.entity_iri("Obama"), NS.Obama)
        self.assertEqual(OntologyTools.entity_iri(str(NS.Obama)), NS.Obama)
        for bad in ("Obama } DROP", "1abc", 5):
            with self.subTest(bad=bad), self.assertRaises(OntologyError):
                OntologyTools.entity_iri(bad)


class SourceEvidenceQueryTests(unittest.TestCase):
    def test_returns_every_tweet_of_the_topic_with_its_claims(self):
        graph, _ = demo_fixture()
        rows = list(graph.query(SOURCE_EVIDENCE.read_text(encoding="utf-8"),
                                initBindings={"topic": TOPIC_IRIS["Obama"]}))
        self.assertEqual(len(rows), 9)
        self.assertEqual([str(row.dialect_label) for row in rows], ["AA"] * 3 + ["Hisp"] * 3 + ["White"] * 3)
        self.assertEqual(str(rows[0].claim_id), "AA_01_C01")
        self.assertEqual(str(rows[0].claim_text), "The author likes Obama's speech.")
        self.assertEqual(list(graph.query(SOURCE_EVIDENCE.read_text(encoding="utf-8"),
                                          initBindings={"topic": TOPIC_IRIS["McDonalds"]})), [])


if __name__ == "__main__":
    unittest.main()
