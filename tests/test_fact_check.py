"""Controlled evidence and mocked Ollama responses; not a model accuracy evaluation."""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

from rdflib import Literal

from data_loader import Tweet, load_tweets
from fact_check import FactCheckError, check_claim
from fact_tools import lookup_entity_facts
from ontology_tools import NS
from populate_ontology import populate_ontology


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


def mock_verdict(verdict="SUPPORTED", references=None):
    client = Mock()
    if references is None:
        references = [{"evidence_id": "http://www.semanticweb.org/group2/runs/fact_check_fixture/tweet/Obama_fixture",
                       "quote": "I like Obama's speech."}]
    client.chat.return_value = {"output": [{"content": [{"text": json.dumps({
        "verdict": verdict, "explanation": (
            "The evidence is insufficient to establish the claim's scope." if verdict == "UNKNOWN"
            else "The cited author explicitly says they like the speech."
        ),
        "references": references,
    })}]}]}
    return client


class ClaimCheckTests(unittest.TestCase):
    def setUp(self):
        self.graph, self.claim = controlled_fixture()

    def test_supported_and_model_payload_preserve_unreviewed_evidence(self):
        client = mock_verdict()
        before_graph, before_claim = set(self.graph), deepcopy(self.claim)
        with patch("fact_check.lookup_entity_facts") as entity_lookup:
            result = check_claim(self.graph, self.claim, client)
            entity_lookup.assert_not_called()
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertEqual(result["review_status"], "unreviewed")
        self.assertEqual(result["decision_source"], "model")
        self.assertEqual(result["assessment_scope"], "retrieved_cited_evidence_only")
        client.chat.assert_called_once()
        payload = client.chat.call_args.args[0]
        self.assertEqual(payload["model"], "qwen3.5:4b")
        self.assertEqual(payload["format"]["properties"]["verdict"]["enum"],
                         ["SUPPORTED", "CONTRADICTED", "UNKNOWN"])
        supplied = json.loads(payload["messages"][1]["content"])
        self.assertEqual(supplied["summary_claim"], self.claim)
        self.assertEqual(supplied["ontology_status"], "not_requested")
        self.assertEqual(len(supplied["evidence"]), 2)
        tweet = next(item for item in supplied["evidence"].values() if item["kind"] == "tweet")
        self.assertEqual(tweet["source_file"], "fixture.txt")
        self.assertEqual(tweet["text"], "I like Obama's speech.")
        self.assertEqual(set(self.graph), before_graph)
        self.assertEqual(self.claim, before_claim)
        json.dumps(result)

    def test_contradicted_claim(self):
        claim = {**self.claim, "claim": "The author dislikes Obama's speech."}
        client = mock_verdict("CONTRADICTED")
        result = check_claim(self.graph, claim, client)
        self.assertEqual(result["verdict"], "CONTRADICTED")
        self.assertEqual(result["review_status"], "unreviewed")

    def test_insufficient_evidence_can_return_unknown_without_references(self):
        client = mock_verdict("UNKNOWN", [])
        result = check_claim(self.graph, {**self.claim, "claim": "Most authors like Obama."}, client)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["reason"], "model_uncertainty")

    def test_missing_explicit_ontology_entity_is_unknown_not_false(self):
        client = mock_verdict("CONTRADICTED")
        with patch("fact_check.lookup_entity_facts", wraps=lookup_entity_facts) as lookup:
            result = check_claim(self.graph, self.claim, client, entity_id="UnknownPerson")
            lookup.assert_called_once_with(self.graph, "UnknownPerson")
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["reason"], "missing_ontology_entity")
        self.assertIn("not evidence of falsehood", result["explanation"])
        client.chat.assert_not_called()

    def test_explicit_ontology_assertion_can_be_referenced_as_a_stored_assertion(self):
        self.graph.remove((NS.Obama, None, None))
        self.graph.add((NS.Obama, NS.hasAge, Literal(55)))
        claim = {**self.claim, "claim": "The supplied ontology records Obama's age as 55."}
        client = mock_verdict(references=[{"evidence_id": "ontology:001", "quote": "55"}])
        result = check_claim(self.graph, claim, client, entity_id="Obama")
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertEqual(result["ontology"]["entity"], str(NS.Obama))
        entry = result["evidence"]["ontology:001"]
        self.assertEqual(entry["kind"], "ontology_assertion")
        self.assertEqual(entry["object"]["value"], "55")
        self.assertEqual(entry["provenance"]["origin"], "asserted")

    def test_source_claim_references_require_their_actual_source_quotation(self):
        uri = "http://www.semanticweb.org/group2/runs/fact_check_fixture/claim/Obama_fixture_C01"
        client = mock_verdict(references=[{"evidence_id": uri, "quote": "I like Obama's speech."}])
        self.assertEqual(check_claim(self.graph, self.claim, client)["verdict"], "SUPPORTED")
        client = mock_verdict(references=[{"evidence_id": uri, "quote": "The author likes Obama's speech."}])
        with self.assertRaises(FactCheckError):
            check_claim(self.graph, self.claim, client)

    def test_incomplete_five_tweet_sample_never_gets_a_definitive_verdict(self):
        tweets = load_tweets("Obama")[:5]
        # Synthetic no-claim records isolate retrieval completeness, not extraction quality.
        records = [{"tweet_id": t.id, "dialect_label": t.dialect_label,
                    "status": "no_claim", "claims": []} for t in tweets]
        graph = populate_ontology(tweets, records)
        claim = {**self.claim, "cited_source_ids": [t.id for t in tweets] + ["Obama_006"]}
        client = mock_verdict()
        result = check_claim(graph, claim, client)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["reason"], "incomplete_cited_evidence")
        self.assertEqual(result["retrieval"]["missing_tweet_ids"], ["Obama_006"])
        self.assertIn("full 90-tweet", result["explanation"])
        client.chat.assert_not_called()

    def test_all_missing_no_citations_and_missing_tweet_text_are_unknown(self):
        for ids in ([], ["Obama_missing"]):
            with self.subTest(ids=ids):
                client = mock_verdict()
                result = check_claim(self.graph, {**self.claim, "cited_source_ids": ids}, client)
                self.assertEqual(result["verdict"], "UNKNOWN")
                client.chat.assert_not_called()
        self.graph.remove((None, NS.tweetText, None))
        client = mock_verdict()
        self.assertEqual(check_claim(self.graph, self.claim, client)["verdict"], "UNKNOWN")
        client.chat.assert_not_called()

    def test_fabricated_evidence_ids_and_quotations_are_rejected(self):
        valid_id = "http://www.semanticweb.org/group2/runs/fact_check_fixture/tweet/Obama_fixture"
        for reference in (
            {"evidence_id": "invented", "quote": "I like Obama's speech."},
            {"evidence_id": valid_id, "quote": "I dislike Obama's speech."},
            {"evidence_id": "ontology:999", "quote": "55"},
        ):
            with self.subTest(reference=reference), self.assertRaises(FactCheckError):
                check_claim(self.graph, self.claim, mock_verdict(references=[reference]))

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
                client = mock_verdict()
                client.chat.return_value["output"][0]["content"][0]["text"] = json.dumps(data)
                check_claim(self.graph, self.claim, client)
        for response in ({}, {"output": []}, {"output": [{"content": [{"text": "not JSON"}]}]}):
            with self.subTest(response=response), self.assertRaises(FactCheckError):
                client = Mock()
                client.chat.return_value = response
                check_claim(self.graph, self.claim, client)


if __name__ == "__main__":
    unittest.main()
