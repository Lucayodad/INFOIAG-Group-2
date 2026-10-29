"""Offline checks of citation retrieval; no factual judgments or model calls."""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock

from rdflib import Graph, Literal, Namespace, RDF

from fact_tools import lookup_fact
from ontology_tools import NS
from summary_claims import extract_sentence


class FactRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.run = Namespace("https://example.org/retrieval-test/")
        for index, status in ((1, "extracted"), (2, "no_claim"), (3, "unclear")):
            tweet = self.run[f"tweet/{index}"]
            self.graph.add((tweet, RDF.type, NS.Tweet))
            self.graph.add((tweet, NS.tweetId, Literal(f"Obama_{index:03d}")))
            self.graph.add((tweet, NS.tweetText, Literal(f"Synthetic source {index}.\nExact spacing.")))
            self.graph.add((tweet, NS.extractionStatus, Literal(status)))
            self.graph.add((tweet, NS.dialectLabel, Literal("AA")))
            self.graph.add((tweet, NS.sourceFile, Literal("test.txt")))
            self.graph.add((tweet, NS.sourceLine, Literal(index)))
            self.graph.add((tweet, NS.aboutTopic, NS.Topic_Obama))
        claim = self.run["claim/1"]
        self.graph.add((self.run["tweet/1"], NS.containsClaim, claim))
        self.graph.add((claim, RDF.type, NS.Claim))
        self.graph.add((claim, NS.claimId, Literal("Obama_001_C01")))
        self.graph.add((claim, NS.hasText, Literal("Synthetic interpretation.")))
        self.graph.add((claim, NS.sourceQuote, Literal("source 1.\nExact spacing.")))
        self.graph.add((claim, NS.targetText, Literal("Synthetic target")))
        self.graph.add((claim, NS.stance, Literal("neutral")))
        self.claim = {
            "claim_id": "S01_C01", "claim": "Synthetic summary claim.",
            "cited_source_ids": ["Obama_001"], "verification_status": "unverified",
        }

    def test_retrieves_evidence_and_provenance_without_mutating_inputs(self):
        before_graph, before_claim = set(self.graph), deepcopy(self.claim)
        result = lookup_fact(self.graph, self.claim)
        self.assertEqual(result["retrieval_status"], "complete")
        self.assertEqual(result["summary_claim_id"], "S01_C01")
        self.assertEqual(result["summary_claim_text"], self.claim["claim"])
        self.assertEqual(result["missing_tweet_ids"], [])
        tweet = result["tweets"][0]
        self.assertEqual(tweet["tweet_id"], "Obama_001")
        self.assertEqual(tweet["text"], "Synthetic source 1.\nExact spacing.")
        self.assertEqual(tweet["source_file"], "test.txt")
        self.assertEqual(tweet["source_line"], 1)
        self.assertEqual(tweet["topic_uri"], str(NS.Topic_Obama))
        self.assertEqual(tweet["dialect_label"], "AA")
        self.assertEqual(tweet["source_claims"][0], {
            "claim_id": "Obama_001_C01", "claim_uri": str(self.run["claim/1"]),
            "text": "Synthetic interpretation.", "source_quote": "source 1.\nExact spacing.",
            "target": "Synthetic target", "stance": "neutral",
        })
        json.dumps(result)
        self.assertEqual(set(self.graph), before_graph)
        self.assertEqual(self.claim, before_claim)

    def test_keeps_tweets_with_no_claim_or_unclear_extraction(self):
        self.claim["cited_source_ids"] = ["Obama_002", "Obama_003"]
        result = lookup_fact(self.graph, self.claim)
        self.assertEqual(result["retrieval_status"], "complete")
        self.assertEqual([t["extraction_status"] for t in result["tweets"]], ["no_claim", "unclear"])
        self.assertEqual([t["source_claims"] for t in result["tweets"]], [[], []])

    def test_reports_partial_missing_and_empty_citations(self):
        cases = (
            (["Obama_099", "Obama_001"], "partial", ["Obama_099"]),
            (["Obama_099", "Obama_098"], "not_found", ["Obama_099", "Obama_098"]),
            ([], "no_citations", []),
        )
        for ids, status, missing in cases:
            with self.subTest(status=status):
                result = lookup_fact(self.graph, {**self.claim, "cited_source_ids": ids})
                self.assertEqual(result["retrieval_status"], status)
                self.assertEqual(result["missing_tweet_ids"], missing)
                self.assertEqual(len(result["tweets"]), int(status == "partial"))

    def test_deduplicates_citations_and_preserves_requested_order(self):
        self.claim["cited_source_ids"] = ["Obama_003", "Obama_001", "Obama_003"]
        result = lookup_fact(self.graph, self.claim)
        self.assertEqual(result["cited_source_ids"], ["Obama_003", "Obama_001"])
        self.assertEqual([t["tweet_id"] for t in result["tweets"]], ["Obama_003", "Obama_001"])

    def test_optional_provenance_and_claim_fields_do_not_hide_tweets_or_claims(self):
        self.graph.remove((None, NS.sourceFile, None))
        self.graph.remove((None, NS.hasText, None))
        result = lookup_fact(self.graph, self.claim)
        self.assertEqual(result["retrieval_status"], "complete")
        self.assertIsNone(result["tweets"][0]["source_file"])
        self.assertIsNone(result["tweets"][0]["source_claims"][0]["text"])
        self.assertEqual(result["tweets"][0]["source_claims"][0]["claim_id"], "Obama_001_C01")

    def test_bound_citation_is_data_and_cannot_inject_sparql(self):
        self.claim["cited_source_ids"] = ['Obama_001" . } UNION { ?s ?p ?o } #']
        result = lookup_fact(self.graph, self.claim)
        self.assertEqual(result["retrieval_status"], "not_found")
        self.assertEqual(result["tweets"], [])

    def test_rejects_invalid_input(self):
        for field, value in (("claim_id", ""), ("claim", None), ("cited_source_ids", "Obama_001"),
                             ("cited_source_ids", [None]), ("cited_source_ids", [" "])):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                lookup_fact(self.graph, {**self.claim, field: value})
        with self.assertRaises(ValueError):
            lookup_fact(self.graph, {"claim_id": "S01_C01", "claim": "Text"})
        with self.assertRaises(TypeError):
            lookup_fact(None, self.claim)
        with self.assertRaises(TypeError):
            lookup_fact(self.graph, [])

    def test_actual_summary_claim_record_format(self):
        # Exercise the existing extractor with a fake response, never a model call.
        client = Mock()
        client.chat.return_value = {"output": [{"content": [{"text": json.dumps({
            "status": "extracted", "claims": [{
                "quote": "Synthetic summary claim.", "claim": "Synthetic summary claim.",
                "target": "Synthetic target", "stance": "neutral",
            }],
        })}]}]}
        record = extract_sentence({"text": "Synthetic summary claim.",
                                   "source_ids": ["Obama_001", "Obama_099"]}, 1, client)["claims"][0]
        result = lookup_fact(self.graph, record)
        self.assertEqual(result["summary_claim_id"], record["claim_id"])
        self.assertEqual(result["retrieval_status"], "partial")
        self.assertIn("Obama_001", [t["tweet_id"] for t in result["tweets"]])


if __name__ == "__main__":
    unittest.main()
