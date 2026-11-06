"""Controlled evidence and mocked model verdicts; not a model accuracy evaluation.

assess() runs the fact agent's finalize path on retrieved evidence: lookup_fact,
prepare_assessment, compact_evidence, then expand_verdict and apply_verdict on
the model's verdict.
"""

from copy import deepcopy
import json
import unittest

from rdflib import Literal

from data_loader import Tweet, load_tweets
from fact_check import FactCheckError, apply_verdict, compact_evidence, expand_verdict, prepare_assessment
from fact_tools import lookup_entity_facts, lookup_fact
from ontology_tools import NS
from populate_ontology import populate_ontology

TWEET_URI = "http://www.semanticweb.org/group2/runs/fact_check_fixture/tweet/Obama_fixture"


def controlled_fixture():
    """One synthetic tweet with complete evidence for its one cited summary claim."""
    tweet = Tweet("Obama_fixture", "Obama", "AA", "I like Obama's speech.", "fixture.txt", 1)
    records = [{"tweet_id": tweet.id, "dialect_label": tweet.dialect_label,
                "status": "extracted", "claims": [{
                    "claim_id": "Obama_fixture_C01", "tweet_id": tweet.id,
                    "dialect_label": tweet.dialect_label, "source_quote": tweet.text,
                    "claim": "The author likes Obama's speech.",
                    "target": "Obama's speech", "stance": "positive",
                }]}]
    graph = populate_ontology([tweet], records, run_id="fact_check_fixture")
    claim = {"claim_id": "S01_C01", "claim": "The author likes Obama's speech.",
             "summary_quote": "The author likes Obama's speech.", "target": "Obama's speech",
             "stance": "positive", "cited_source_ids": [tweet.id], "verification_status": "unverified"}
    return graph, claim


def verdict(value="SUPPORTED", references=None):
    if references is None:
        references = [{"evidence_id": TWEET_URI, "quote": "I like Obama's speech."}]
    return {"verdict": value, "references": references, "explanation": (
        "The evidence is insufficient to establish the claim's scope." if value == "UNKNOWN"
        else "The cited author explicitly says they like the speech.")}


def assess(graph, claim, data, entity_id=None):
    retrieval = lookup_fact(graph, claim)
    ontology = lookup_entity_facts(graph, entity_id) if entity_id is not None else None
    assessment = prepare_assessment(claim, retrieval, ontology)
    view = compact_evidence(assessment)
    return apply_verdict(assessment, expand_verdict(data, view, assessment)), view


class ClaimCheckTests(unittest.TestCase):
    def setUp(self):
        self.graph, self.claim = controlled_fixture()

    def test_supported_verdict_keeps_unreviewed_evidence(self):
        before_graph, before_claim = set(self.graph), deepcopy(self.claim)
        result, view = assess(self.graph, self.claim, verdict())
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertEqual(result["review_status"], "unreviewed")
        self.assertEqual(result["decision_source"], "model")
        self.assertEqual(result["assessment_scope"], "retrieved_cited_evidence_only")
        self.assertEqual(len(view["evidence_catalogue"]), 2)  # the tweet and its extracted claim
        tweet = next(item for item in view["evidence_catalogue"].values() if item["kind"] == "tweet")
        self.assertEqual(tweet["quote_text"], "I like Obama's speech.")
        self.assertEqual(next(item for item in result["evidence"].values() if item["kind"] == "tweet")["source_file"],
                         "fixture.txt")
        self.assertEqual(set(self.graph), before_graph)
        self.assertEqual(self.claim, before_claim)
        json.dumps(result)

    def test_contradicted_claim(self):
        result, _ = assess(self.graph, {**self.claim, "claim": "The author dislikes Obama's speech."},
                           verdict("CONTRADICTED"))
        self.assertEqual(result["verdict"], "CONTRADICTED")
        self.assertEqual(result["review_status"], "unreviewed")

    def test_insufficient_evidence_can_return_unknown_without_references(self):
        result, _ = assess(self.graph, {**self.claim, "claim": "Most authors like Obama."}, verdict("UNKNOWN", []))
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["reason"], "model_uncertainty")

    def test_missing_explicit_ontology_entity_is_unknown_not_false(self):
        result, _ = assess(self.graph, self.claim, verdict("CONTRADICTED"), entity_id="UnknownPerson")
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["reason"], "missing_ontology_entity")
        self.assertIn("not evidence of falsehood", result["explanation"])

    def test_explicit_ontology_assertion_can_be_referenced_as_a_stored_assertion(self):
        self.graph.remove((NS.Obama, None, None))
        self.graph.add((NS.Obama, NS.hasAge, Literal(55)))
        claim = {**self.claim, "claim": "The supplied ontology records Obama's age as 55."}
        result, _ = assess(self.graph, claim, verdict(references=[{"evidence_id": "ontology:001", "quote": "55"}]),
                           entity_id="Obama")
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertEqual(result["ontology"]["entity"], str(NS.Obama))
        entry = result["evidence"]["ontology:001"]
        self.assertEqual(entry["kind"], "ontology_assertion")
        self.assertEqual(entry["object"]["value"], "55")
        self.assertEqual(entry["provenance"]["origin"], "asserted")

    def test_source_claim_references_require_their_actual_source_quotation(self):
        uri = "http://www.semanticweb.org/group2/runs/fact_check_fixture/claim/Obama_fixture_C01"
        result, _ = assess(self.graph, self.claim, verdict(references=[{"evidence_id": uri, "quote": "I like Obama's speech."}]))
        self.assertEqual(result["verdict"], "SUPPORTED")
        with self.assertRaises(FactCheckError):
            assess(self.graph, self.claim, verdict(references=[{"evidence_id": uri, "quote": "The author likes Obama's speech."}]))

    def test_incomplete_five_tweet_sample_never_gets_a_definitive_verdict(self):
        tweets = load_tweets("Obama")[:5]
        # Synthetic no-claim records isolate retrieval completeness, not extraction quality.
        records = [{"tweet_id": t.id, "dialect_label": t.dialect_label,
                    "status": "no_claim", "claims": []} for t in tweets]
        graph = populate_ontology(tweets, records, run_id="five_tweets")
        claim = {**self.claim, "cited_source_ids": [t.id for t in tweets] + ["Obama_006"]}
        cited = f"http://www.semanticweb.org/group2/runs/five_tweets/tweet/{tweets[0].id}"
        result, _ = assess(graph, claim, verdict(references=[{"evidence_id": cited, "quote": tweets[0].text[:20]}]))
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["reason"], "incomplete_cited_evidence")
        self.assertEqual(result["retrieval"]["missing_tweet_ids"], ["Obama_006"])
        self.assertIn("full 90-tweet", result["explanation"])

    def test_no_citations_missing_tweets_and_missing_text_are_never_complete(self):
        for ids in ([], ["Obama_missing"]):
            with self.subTest(ids=ids):
                assessment = prepare_assessment(self.claim, lookup_fact(self.graph, {**self.claim, "cited_source_ids": ids}))
                self.assertFalse(assessment["evidence_complete"])
                self.assertEqual(apply_verdict(assessment, verdict("UNKNOWN", []))["verdict"], "UNKNOWN")
        self.graph.remove((None, NS.tweetText, None))
        self.graph.remove((None, NS.tweetHasText, None))
        assessment = prepare_assessment(self.claim, lookup_fact(self.graph, self.claim))
        self.assertFalse(assessment["evidence_complete"])
        self.assertEqual(assessment["reason"], "incomplete_cited_evidence")

    def test_fabricated_evidence_ids_and_quotations_are_rejected(self):
        for reference in (
            {"evidence_id": "invented", "quote": "I like Obama's speech."},
            {"evidence_id": TWEET_URI, "quote": "I dislike Obama's speech."},
            {"evidence_id": "ontology:999", "quote": "55"},
        ):
            with self.subTest(reference=reference), self.assertRaises(FactCheckError):
                assess(self.graph, self.claim, verdict(references=[reference]))

    def test_malformed_verdicts_are_rejected(self):
        for data in (
            {"verdict": "TRUE", "explanation": "Text", "references": []},
            {"verdict": "SUPPORTED", "explanation": "Text", "references": []},
            {"verdict": "UNKNOWN", "explanation": " ", "references": []},
            {"verdict": "UNKNOWN", "explanation": "Text", "references": "wrong"},
            {"verdict": "UNKNOWN", "explanation": "Text", "references": [{"evidence_id": "x"}]},
            {"verdict": "UNKNOWN", "explanation": "Text", "references": [], "review_status": "reviewed"},
        ):
            with self.subTest(data=data), self.assertRaises(FactCheckError):
                assess(self.graph, self.claim, data)


if __name__ == "__main__":
    unittest.main()
