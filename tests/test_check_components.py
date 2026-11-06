"""Fact checking of a five-sentence summary: synthetic complete evidence, mocked model outputs."""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock

from data_loader import Tweet
from fact_check import fact_check_summary
from populate_ontology import populate_ontology
from summary_claims import flatten_summary_claims
from test_fact_agent import loop_response


def complete_check_fixture():
    source = [
        ("AA_01", "AA", "I like Obama's speech.", "The author likes Obama's speech.", "speech", "positive"),
        ("AA_02", "AA", "Obama's speech is good.", "The author approves of Obama's speech.", "speech", "positive"),
        ("AA_03", "AA", "I dislike the policy.", "The author dislikes the policy.", "policy", "negative"),
        ("Hisp_01", "Hisp", "I like Obama's speech.", "The author likes Obama's speech.", "speech", "positive"),
        ("Hisp_02", "Hisp", "I like Obama's speech.", "The author likes Obama's speech.", "speech", "positive"),
        ("Hisp_03", "Hisp", "I dislike the policy.", "The author dislikes the policy.", "policy", "negative"),
        ("White_01", "White", "I like Obama's speech.", "The author likes Obama's speech.", "speech", "positive"),
        ("White_02", "White", "I dislike the policy.", "The author dislikes the policy.", "policy", "negative"),
        ("White_03", "White", "I want more detail.", "The author wants more detail.", "detail", "neutral"),
    ]
    tweets, extractions = [], []
    for line, (key, group, text, claim, target, stance) in enumerate(source, 1):
        tweets.append(Tweet(key, "Obama", group, text, "controlled.txt", line))
        extractions.append({"tweet_id": key, "dialect_label": group, "status": "extracted", "claims": [{
            "claim_id": key + "_C01", "tweet_id": key, "dialect_label": group,
            "source_quote": text, "claim": claim, "target": target, "stance": stance,
        }]})
    graph = populate_ontology(tweets, extractions, run_id="checks_fixture")
    sentences = ["The author likes Obama's speech.", "The author dislikes Obama's speech.",
                 "Millions watched the speech.", "End of briefing.", "Further context follows."]
    summary = []
    for index, text in enumerate(sentences, 1):
        claims = [{"claim_id": f"S{index:02d}_C01", "claim": text, "summary_quote": text,
                   "cited_source_ids": ["AA_01"], "verification_status": "unverified"}] if index <= 3 else []
        summary.append({"sentence_id": f"S{index:02d}", "sentence": text,
                        "status": "extracted" if claims else "no_claim", "claims": claims})
    return graph, summary


def factual_client():
    client = Mock()
    responses = []
    for verdict in ("SUPPORTED", "CONTRADICTED", "UNKNOWN"):
        refs = [{"evidence_id": "http://www.semanticweb.org/group2/runs/checks_fixture/tweet/AA_01",
                 "quote": "I like Obama's speech."}] if verdict != "UNKNOWN" else []
        responses.extend([loop_response("lookup_fact"), loop_response(
            "finalize", verdict=verdict, references=refs,
            explanation="Original tweet explicitly says likes." if verdict != "UNKNOWN" else "The cited tweet does not establish viewer counts.")])
    client.chat.side_effect = responses
    return client


class FactSummaryTests(unittest.TestCase):
    def setUp(self):
        self.graph, self.summary = complete_check_fixture()

    def test_summary_reuses_agent_and_documented_issue_severities(self):
        before_graph, before_summary = set(self.graph), deepcopy(self.summary)
        result = fact_check_summary(self.graph, self.summary, factual_client())
        self.assertEqual(result["claims_examined"], 3)
        self.assertEqual(result["verdict_counts"], {"SUPPORTED": 1, "CONTRADICTED": 1, "UNKNOWN": 1})
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["tool_call_count"], 3)
        self.assertEqual(result["model_call_count"], 6)
        self.assertEqual([(i["claim_id"], i["sentence_id"], i["severity"]) for i in result["issues"]],
                         [("S02_C01", "S02", "serious"), ("S03_C01", "S03", "unresolved")])
        self.assertTrue(result["issues"][0]["evidence"])
        self.assertTrue(all(a["execution_trace"] for a in result["assessments"]))
        self.assertEqual(set(self.graph), before_graph)
        self.assertEqual(self.summary, before_summary)
        json.dumps(result)

    def test_unknown_and_unclear_are_unresolved_not_serious(self):
        for record in self.summary[:3]:
            record.update(status="unclear", claims=[])
        client = Mock()
        result = fact_check_summary(self.graph, self.summary, client)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(len(result["issues"]), 3)
        self.assertTrue(all(item["severity"] == "unresolved" for item in result["issues"]))
        client.chat.assert_not_called()

    def test_summary_format_rejects_missing_sentences_duplicate_ids_and_invented_quotes(self):
        for records in (self.summary[:4], self.summary + [self.summary[0]]):
            with self.assertRaises(ValueError):
                flatten_summary_claims(records)
        for field, value in (("summary_quote", "Not in the sentence"), ("claim_id", ""),
                             ("cited_source_ids", [None])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                records = deepcopy(self.summary)
                records[0]["claims"][0][field] = value
                flatten_summary_claims(records)
        records = deepcopy(self.summary)
        records[1]["claims"][0]["claim_id"] = records[0]["claims"][0]["claim_id"]
        with self.assertRaises(ValueError):
            flatten_summary_claims(records)

    def test_zero_tool_budget_bounds_every_claim(self):
        client = Mock()
        result = fact_check_summary(self.graph, self.summary, client, max_tool_calls=0)
        self.assertEqual(result["verdict_counts"]["UNKNOWN"], 3)
        self.assertEqual(result["tool_call_count"], 0)
        client.chat.assert_not_called()

    def test_all_supported_summary_passes(self):
        client = Mock()
        responses = []
        for record in self.summary[:3]:
            text = "The author likes Obama's speech."
            record["sentence"] = text
            record["claims"][0].update(claim=text, summary_quote=text)
            responses.extend([loop_response("lookup_fact"), loop_response(
                "finalize", verdict="SUPPORTED", explanation="Original tweet expresses liking the speech.",
                references=[{"evidence_id": "http://www.semanticweb.org/group2/runs/checks_fixture/tweet/AA_01",
                             "quote": "I like Obama's speech."}])])
        client.chat.side_effect = responses
        result = fact_check_summary(self.graph, self.summary, client)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["issues"], [])


if __name__ == "__main__":
    unittest.main()
