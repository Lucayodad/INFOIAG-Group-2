"""Three-group synthetic complete evidence; mocked model outputs, not accuracy scores."""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock

from rdflib import Literal, RDF, URIRef

from data_loader import Tweet
from fact_check import fact_check_summary
from ontology_tools import NS
from populate_ontology import populate_ontology
from representation_check import RepresentationError, check_summary_fairness, match_summary_claims, representation_check
from representation_tools import get_group_claims
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


def matching_response(p_status="MATCHED", q_status="NOT_MATCHED", r_status="NOT_MATCHED", equivalence_complete=True):
    clusters = []
    specs = [(["K001", "K002"], p_status,
              [("AA_01_C01", "I like Obama's speech."), ("AA_02_C01", "Obama's speech is good.")]),
             (["K003"], q_status, [("AA_03_C01", "I dislike the policy.")]),
             (["K004"], r_status, [("White_03_C01", "I want more detail.")])]
    for keys, status, evidence in specs:
        matched = ["S01_C01"] if status == "MATCHED" else []
        clusters.append({
            "candidate_ids": keys, "status": status, "summary_claim_ids": matched,
            "source_evidence": [{"claim_id": key, "quote": quote} for key, quote in evidence],
            "summary_evidence": [{"claim_id": "S01_C01", "quote": "The author likes Obama's speech."}] if matched else [],
            "explanation": "Synthetic equivalence and matching judgment for deterministic tests.",
        })
    return {"output": [{"content": [{"text": json.dumps({"equivalence_complete": equivalence_complete,
                                                          "clusters": clusters})}]}]}


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


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.graph, self.summary = complete_check_fixture()
        self.expected = {label: 3 for label in ("AA", "Hisp", "White")}

    def groups(self):
        return {label: get_group_claims(self.graph, "Obama", label) for label in self.expected}

    def matching(self, **kwargs):
        client = Mock()
        client.chat.return_value = matching_response(**kwargs)
        return match_summary_claims(self.groups(), self.summary, client)

    def test_sparql_retrieval_preserves_group_and_original_evidence(self):
        result = get_group_claims(self.graph, "Obama", "Hisp")
        self.assertEqual(result["tweet_count"], 3)
        self.assertEqual(len(result["claims"]), 3)
        self.assertTrue(all(item["dialect_label"] == "Hisp" for item in result["claims"]))
        self.assertTrue(all(item["source_file"] == "controlled.txt" for item in result["claims"]))
        self.assertEqual(result["integrity_issues"], [])
        with self.assertRaises(ValueError):
            get_group_claims(self.graph, "Obama", "InventedGroup")

    def test_exact_and_paraphrase_aggregation_use_distinct_supporting_tweets(self):
        matching = self.matching()
        self.assertEqual(len(matching["clusters"]), 3)
        positive = matching["clusters"][0]
        self.assertEqual(positive["candidate_ids"], ["K001", "K002"])
        self.assertEqual(positive["weights"], {"AA": 2, "Hisp": 2, "White": 1})
        self.assertEqual(len(positive["source_claim_ids"]), 5)
        self.assertEqual(positive["summary_claim_ids"], ["S01_C01"])
        self.assertTrue(positive["source_evidence"])

    def test_multiple_claim_records_in_one_tweet_do_not_inflate_weights(self):
        original = URIRef("http://www.semanticweb.org/group2/runs/checks_fixture/claim/AA_01_C01")
        duplicate = URIRef("http://www.semanticweb.org/group2/runs/checks_fixture/claim/AA_01_C02")
        tweet = URIRef("http://www.semanticweb.org/group2/runs/checks_fixture/tweet/AA_01")
        for _, predicate, obj in list(self.graph.triples((original, None, None))):
            self.graph.add((duplicate, predicate, Literal("AA_01_C02") if predicate == NS.claimId else obj))
        self.graph.add((tweet, NS.containsClaim, duplicate))
        positive = self.matching()["clusters"][0]
        self.assertEqual(len(positive["source_claim_ids"]), 6)
        self.assertEqual(positive["weights"]["AA"], 2)

    def test_one_rdf_claim_can_have_two_supporting_tweets(self):
        tweet = URIRef("http://www.semanticweb.org/group2/runs/checks_fixture/tweet/Hisp_02")
        claim = URIRef("http://www.semanticweb.org/group2/runs/checks_fixture/claim/Hisp_01_C01")
        self.graph.remove((tweet, NS.containsClaim, None))
        self.graph.add((tweet, NS.containsClaim, claim))
        group = get_group_claims(self.graph, "Obama", "Hisp")
        self.assertEqual(group["integrity_issues"], [])
        self.assertEqual(self.matching()["clusters"][0]["weights"]["Hisp"], 2)

    def test_weighted_coverage_disparity_and_omissions(self):
        report = check_summary_fairness(self.matching(), 0.25, data_complete=True)
        self.assertAlmostEqual(report["coverage_by_group"]["AA"]["coverage"], 2 / 3)
        self.assertAlmostEqual(report["coverage_by_group"]["Hisp"]["coverage"], 2 / 3)
        self.assertAlmostEqual(report["coverage_by_group"]["White"]["coverage"], 1 / 3)
        self.assertAlmostEqual(report["disparity"], 1 / 3)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(len(report["omitted_claims"]), 2)
        self.assertTrue(report["representation_issues"])
        self.assertEqual(report["metric"], "internal_weighted_claim_coverage")

    def test_pass_and_exact_threshold_boundary(self):
        matching = self.matching()
        threshold = 2 / 3 - 1 / 3
        self.assertEqual(check_summary_fairness(matching, threshold, data_complete=True)["status"], "PASS")

    def test_unknown_is_an_interval_not_an_omission(self):
        matching = self.matching(r_status="UNKNOWN")
        result = check_summary_fairness(matching, 0.25, data_complete=True)
        white = result["coverage_by_group"]["White"]
        self.assertIsNone(white["coverage"])
        self.assertAlmostEqual(white["lower"], 1 / 3)
        self.assertAlmostEqual(white["upper"], 2 / 3)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(len(result["omitted_claims"]), 1)
        self.assertEqual(len(result["uncertain_claims"]), 1)
        self.assertIsNone(result["disparity"])
        self.assertEqual(check_summary_fairness(matching, 1, data_complete=True)["status"], "PASS")

    def test_uncertain_equivalence_prevents_definitive_coverage(self):
        result = check_summary_fairness(self.matching(equivalence_complete=False), 0.25, data_complete=True)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIn("uncertain_claim_equivalence", result["reasons"])

    def test_shared_unknown_claim_moves_groups_together(self):
        # Policy claim Q has weight 1/3 in every group. Its unknown m_c cancels
        # from pairwise coverage differences, leaving disparity exactly 1/3.
        result = check_summary_fairness(self.matching(q_status="UNKNOWN"), 0.25, data_complete=True)
        self.assertIsNone(result["coverage_by_group"]["AA"]["coverage"])
        self.assertEqual(result["status"], "FAIL")
        self.assertAlmostEqual(result["disparity_bounds"]["lower"], 1 / 3)
        self.assertAlmostEqual(result["disparity_bounds"]["upper"], 1 / 3)
        self.assertAlmostEqual(result["disparity"], 1 / 3)

    def test_missing_groups_and_sample_counts_skip_model_and_return_unknown(self):
        client = Mock()
        result = representation_check(self.graph, self.summary, client, "Obama", 0.25)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertFalse(result["data_complete"])
        self.assertEqual(result["expected_group_counts"], {"AA": 30, "Hisp": 30, "White": 30})
        client.chat.assert_not_called()
        for tweet in list(self.graph.subjects(NS.dialectLabel, Literal("Hisp"))):
            self.graph.remove((tweet, None, None))
        result = representation_check(self.graph, self.summary, client, "Obama", 0.25,
                                      expected_group_counts=self.expected)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIsNone(result["coverage_by_group"]["Hisp"]["coverage"])
        self.assertIsNone(result["coverage_by_group"]["Hisp"]["lower"])

    def test_no_fabricated_group_labels_are_accepted_or_returned(self):
        tweet = next(self.graph.subjects(NS.dialectLabel, Literal("AA")))
        self.graph.add((tweet, NS.dialectLabel, Literal("Invented")))
        client = Mock()
        result = representation_check(self.graph, self.summary, client, "Obama", 0.25,
                                      expected_group_counts=self.expected)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(set(result["coverage_by_group"]), {"AA", "Hisp", "White"})
        client.chat.assert_not_called()

    def test_complete_representation_interface_and_shared_graph_are_json_serializable(self):
        before = set(self.graph)
        client = Mock()
        client.chat.return_value = matching_response()
        report = representation_check(self.graph, self.summary, client, "Obama", 0.25,
                                      expected_group_counts=self.expected)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["model_call_count"], 1)
        self.assertTrue(report["data_complete"])
        self.assertTrue(report["supporting_source_evidence"])
        self.assertEqual(set(self.graph), before)
        json.dumps(report, allow_nan=False)

    def test_missing_threshold_does_not_invent_a_paper_threshold(self):
        result = check_summary_fairness(self.matching(), data_complete=True)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIsNone(result["disparity_threshold"])
        self.assertIn("threshold_not_configured", result["reasons"])
        for threshold in (-1, 2, True, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                check_summary_fairness(self.matching(), threshold, data_complete=True)

    def test_fabricated_quotes_ids_partitions_and_entity_only_merges_are_rejected(self):
        mutations = [
            lambda data: data["clusters"][0]["source_evidence"][0].update(quote="Fabricated quote"),
            lambda data: data["clusters"][0]["summary_evidence"][0].update(claim_id="S99_C01"),
            lambda data: data["clusters"][0]["candidate_ids"].append("K999"),
            lambda data: data["clusters"].pop(),
            lambda data: data["clusters"][1].update(candidate_ids=["K001"]),
            lambda data: data["clusters"][0].update(dialect_label="Invented"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                response = matching_response()
                data = json.loads(response["output"][0]["content"][0]["text"])
                mutate(data)
                response["output"][0]["content"][0]["text"] = json.dumps(data)
                client = Mock()
                client.chat.return_value = response
                with self.assertRaises(RepresentationError):
                    match_summary_claims(self.groups(), self.summary, client)
        response = matching_response()
        data = json.loads(response["output"][0]["content"][0]["text"])
        data["clusters"][0]["candidate_ids"].append("K003")
        data["clusters"][0]["source_evidence"].extend(data["clusters"][1]["source_evidence"])
        data["clusters"].pop(1)
        response["output"][0]["content"][0]["text"] = json.dumps(data)
        client = Mock()
        client.chat.return_value = response
        with self.assertRaises(RepresentationError):
            match_summary_claims(self.groups(), self.summary, client)
        # Exact grouping uses the full statement, never just entity/target or stance.
        groups = self.groups()
        for claim in groups["AA"]["claims"]:
            claim.update(target="Obama", stance="positive")
        fallback = match_summary_claims(groups, self.summary, Mock(), max_model_calls=0)
        ids = [{member["claim_id"] for member in item["members"]} for item in fallback["clusters"]]
        self.assertFalse(any({"AA_01_C01", "AA_03_C01"}.issubset(group) for group in ids))

    def test_bad_model_output_returns_unknown_and_keeps_original_evidence(self):
        client = Mock()
        client.chat.return_value = {}
        result = representation_check(self.graph, self.summary, client, "Obama", 0.25,
                                      expected_group_counts=self.expected)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["model_call_count"], 1)
        self.assertTrue(result["supporting_source_evidence"])
        self.assertEqual(result["execution_trace"][0]["status"], "rejected")

    def test_ambiguous_source_identities_return_unknown_without_model_calls(self):
        uri = URIRef("http://www.semanticweb.org/group2/runs/checks_fixture/claim/Hisp_01_C01")
        self.graph.set((uri, NS.claimId, Literal("AA_01_C01")))
        client = Mock()
        result = representation_check(self.graph, self.summary, client, "Obama", 0.25,
                                      expected_group_counts=self.expected)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertFalse(result["data_complete"])
        self.assertEqual(result["matching"]["reason"], "invalid_source_identity")
        client.chat.assert_not_called()

    def test_input_and_model_budgets_do_not_truncate_or_guess(self):
        client = Mock()
        for options in ({"max_model_calls": 0}, {"max_input_bytes": 1}):
            result = match_summary_claims(self.groups(), self.summary, client, **options)
            self.assertEqual(result["aggregation_status"], "UNKNOWN")
            self.assertTrue(all(item["status"] == "UNKNOWN" for item in result["clusters"]))
            self.assertEqual(result["model_call_count"], 0)
        client.chat.assert_not_called()

    def test_unclear_extraction_prevents_silent_omissions(self):
        summary = deepcopy(self.summary)
        summary[2].update(status="unclear", claims=[])
        client = Mock()
        result = representation_check(self.graph, summary, client, "Obama", 0.25,
                                      expected_group_counts=self.expected)
        self.assertEqual(result["status"], "UNKNOWN")
        client.chat.assert_not_called()

    def test_coverage_bounds_can_prove_failure_despite_some_unknown_matches(self):
        def cluster(key, status, counts):
            return {"canonical_claim_id": key, "status": status,
                    "supporting_tweet_ids": {g: [f"{g}_{key}_{i}" for i in range(counts.get(g, 0))]
                                             for g in self.expected}, "weights": {g: counts.get(g, 0) for g in self.expected}}
        matching = {"aggregation_status": "ready", "clusters": [
            cluster("A", "MATCHED", {"AA": 4, "Hisp": 1, "White": 1}),
            cluster("B", "NOT_MATCHED", {"AA": 1, "Hisp": 3, "White": 3}),
            cluster("U", "UNKNOWN", {"White": 1}),
        ]}
        result = check_summary_fairness(matching, 0.3, data_complete=True)
        self.assertEqual(result["status"], "FAIL")
        self.assertAlmostEqual(result["disparity_bounds"]["lower"], 0.55)
        self.assertAlmostEqual(result["disparity_bounds"]["upper"], 0.6)
        self.assertIsNone(result["coverage_by_group"]["White"]["coverage"])

    def test_zero_denominator_is_undefined_not_zero_coverage(self):
        matching = self.matching()
        for item in matching["clusters"]:
            item["supporting_tweet_ids"]["AA"] = []
        result = check_summary_fairness(matching, 0.25, data_complete=True)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIsNone(result["coverage_by_group"]["AA"]["lower"])
        self.assertIn("zero_claim_weight_for_a_group", result["reasons"])


if __name__ == "__main__":
    unittest.main()
