"""Bounded batching mechanics with gold-labelled mocked decisions, not model accuracy."""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock

from data_loader import Tweet
from populate_ontology import populate_ontology
from representation_check import representation_check
from representation_tools import get_group_claims


def topic_fixture(unique=False, duplicate=False):
    phrases = {
        "speech": ["The author likes Obama's speech.", "The author approves of Obama's speech.",
                   "The author expresses support for Obama's speech."],
        "policy": ["The author likes Obama's policy.", "The author approves of Obama's policy.",
                   "The author expresses support for Obama's policy."],
        "detail": ["The author wants more detail from Obama.", "The author requests further explanation from Obama.",
                   "The author asks Obama for additional detail."],
    }
    tweets, records, gold = [], [], {}
    for group in ("AA", "Hisp", "White"):
        for index in range(30):
            category = ("speech" if index < (20 if group == "AA" else 10) else
                        "policy" if group == "AA" or group == "Hisp" and index < 20 else "detail")
            key = f"{group}_{index + 1:03d}"
            claim_id = key + "_C01"
            original = (f"I request item {key} from Obama." if unique else
                        "I like Obama's speech." if category == "speech" else
                        "I like Obama's policy." if category == "policy" else "I want more detail from Obama.")
            text = f"Original source {key}: {original} " + "Source context. " * (2 if unique else 12)
            meaning = key if unique else category
            claim = f"The author requests the distinct item {key} from Obama." if unique else phrases[category][index % 3]
            stance = "neutral" if unique or category == "detail" else "positive"
            tweet = Tweet(key, "Obama", group, text, "synthetic_90.txt", len(tweets) + 1)
            entry = {"claim_id": claim_id, "tweet_id": key, "dialect_label": group,
                     "source_quote": text, "claim": claim, "target": "Obama", "stance": stance}
            tweets.append(tweet)
            records.append({"tweet_id": key, "dialect_label": group, "status": "extracted", "claims": [entry]})
            gold[claim_id] = meaning
    if duplicate:
        extra = {**records[0]["claims"][0], "claim_id": "AA_001_C02",
                 "claim": "The author is favorable about Obama's speech."}
        records[0]["claims"].append(extra)
        gold[extra["claim_id"]] = "speech"
    graph = populate_ontology(tweets, records, run_id="batch_fixture")
    sentence = "The author likes Obama's speech."
    summary = [{"sentence_id": "S01", "sentence": sentence, "status": "extracted", "claims": [{
        "claim_id": "S01_C01", "claim": sentence, "summary_quote": sentence,
        "cited_source_ids": ["AA_001"],
    }]}]
    summary.extend({"sentence_id": f"S{i:02d}", "sentence": f"End of section {i}.",
                    "status": "no_claim", "claims": []} for i in range(2, 6))
    return graph, summary, gold


def oracle_client(gold, unknown_matching=(), unknown_equivalence=False):
    """Use fixture gold labels only; every response references the actual supplied data."""
    client = Mock()

    def category(item):
        return gold[item["source_evidence"][0]["claim_id"]]

    def quotes(item):
        return [{"claim_id": item["source_evidence"][0]["claim_id"],
                 "quote": item["source_evidence"][0]["source_quote"]}]

    def respond(payload):
        body = json.loads(payload["messages"][1]["content"])
        if "candidates" in body:
            groups = {}
            for key, item in body["candidates"].items():
                groups.setdefault(category(item), []).append(key)
            clusters = []
            for meaning, keys in groups.items():
                status = "UNKNOWN" if meaning in unknown_matching else "MATCHED" if meaning == "speech" else "NOT_MATCHED"
                summary_ids = ["S01_C01"] if status == "MATCHED" else []
                clusters.append({
                    "candidate_ids": keys, "status": status, "summary_claim_ids": summary_ids,
                    "source_evidence": [q for key in keys for q in quotes(body["candidates"][key])],
                    "summary_evidence": [{"claim_id": "S01_C01", "quote": body["summary_claims"][0]["summary_quote"]}]
                                        if summary_ids else [],
                    "explanation": "Gold-labelled synthetic local matching decision.",
                })
            data = {"equivalence_complete": True, "clusters": clusters}
        else:
            decisions = []
            for key, item in body["incoming"].items():
                equivalent = next((old for old, value in body["registry"].items()
                                   if category(item) == category(value)), None)
                status = "UNKNOWN" if unknown_equivalence else "EQUIVALENT" if equivalent else "DISTINCT"
                target = equivalent if status == "EQUIVALENT" else None
                evidence = quotes(item) + (quotes(body["registry"][target]) if target else [])
                decisions.append({"candidate_id": key, "status": status, "equivalent_to": target,
                                  "source_evidence": evidence, "explanation": "Gold-labelled cross-batch equivalence decision."})
            data = {"decisions": decisions}
        return {"output": [{"content": [{"text": json.dumps(data)}]}]}

    client.chat.side_effect = respond
    return client


class BatchedRepresentationTests(unittest.TestCase):
    def setUp(self):
        self.graph, self.summary, self.gold = topic_fixture()

    def test_complete_90_tweet_topic_cross_batch_equivalence_and_weights(self):
        client = oracle_client(self.gold)
        before, summary_before = set(self.graph), deepcopy(self.summary)
        report = representation_check(self.graph, self.summary, client, "Obama", 0.2, batch_size=2)
        self.assertTrue(report["data_complete"])
        self.assertEqual(report["source_tweet_counts"], {"AA": 30, "Hisp": 30, "White": 30})
        self.assertEqual(report["matching"]["aggregation_status"], "ready")
        self.assertEqual(report["status"], "FAIL")
        clusters = report["matching"]["clusters"]
        self.assertEqual(len(clusters), 3)
        self.assertEqual(sum(len(c["members"]) for c in clusters), 90)
        positive = report["matched_claims"][0]
        self.assertEqual(positive["weights"], {"AA": 20, "Hisp": 10, "White": 10})
        self.assertGreater(len(positive["candidate_ids"]), 2)
        self.assertAlmostEqual(report["coverage_by_group"]["AA"]["coverage"], 2 / 3)
        self.assertAlmostEqual(report["coverage_by_group"]["Hisp"]["coverage"], 1 / 3)
        self.assertAlmostEqual(report["coverage_by_group"]["White"]["coverage"], 1 / 3)
        self.assertAlmostEqual(report["disparity"], 1 / 3)
        self.assertGreater(sum(e["step"] == "aggregation_and_matching" for e in report["execution_trace"]), 1)
        self.assertTrue(any(e["step"] == "cross_batch_reconciliation" for e in report["execution_trace"]))
        self.assertEqual(report["model_call_count"], client.chat.call_count)
        self.assertLessEqual(report["model_call_count"], 64)
        self.assertTrue(all(e["input_bytes"] <= 12000 for e in report["execution_trace"]))
        # The old all-supporters prompt exceeded the limit; all evidence is still retained.
        self.assertGreater(len(json.dumps(report["supporting_source_evidence"]).encode()), 12000)
        self.assertEqual({m["claim_id"] for c in clusters for m in c["members"]}, set(self.gold))
        self.assertTrue(all(m["source_file"] == "synthetic_90.txt" for c in clusters for m in c["members"]))
        self.assertEqual(set(self.graph), before)
        self.assertEqual(self.summary, summary_before)
        json.dumps(report, allow_nan=False)

    def test_distinct_claims_sharing_entity_and_words_do_not_merge(self):
        report = representation_check(self.graph, self.summary, oracle_client(self.gold), "Obama", 0.2, batch_size=1)
        for cluster in report["matching"]["clusters"]:
            meanings = {self.gold[key] for key in cluster["source_claim_ids"]}
            self.assertEqual(len(meanings), 1)
            self.assertTrue(all(member["target"] == "Obama" for member in cluster["members"]))
        self.assertEqual(len(report["omitted_claims"]), 2)

    def test_duplicate_extractions_across_batches_do_not_inflate_weights(self):
        graph, summary, gold = topic_fixture(duplicate=True)
        report = representation_check(graph, summary, oracle_client(gold), "Obama", 0.2, batch_size=1)
        positive = report["matched_claims"][0]
        self.assertEqual(positive["weights"], {"AA": 20, "Hisp": 10, "White": 10})
        self.assertIn("AA_001_C02", positive["source_claim_ids"])
        self.assertEqual(len(positive["source_claim_ids"]), 41)

    def test_unknown_matching_remains_separate_and_uses_global_bounds(self):
        client = oracle_client(self.gold, unknown_matching={"speech"})
        report = representation_check(self.graph, self.summary, client, "Obama", 0.2, batch_size=2)
        self.assertEqual(report["matching"]["aggregation_status"], "ready")
        self.assertEqual(report["status"], "UNKNOWN")
        self.assertEqual(len(report["uncertain_claims"]), 1)
        self.assertEqual(len(report["omitted_claims"]), 2)
        self.assertIsNone(report["coverage_by_group"]["AA"]["coverage"])
        self.assertAlmostEqual(report["coverage_by_group"]["AA"]["upper"], 2 / 3)
        self.assertAlmostEqual(report["disparity_bounds"]["upper"], 1 / 3)

    def test_unknown_equivalence_is_semantic_uncertainty_not_missing_input(self):
        report = representation_check(self.graph, self.summary, oracle_client(self.gold, unknown_equivalence=True),
                                      "Obama", 0.2, batch_size=1)
        self.assertTrue(report["data_complete"])
        self.assertEqual(report["status"], "UNKNOWN")
        self.assertEqual(report["matching"]["aggregation_status"], "UNKNOWN")
        self.assertIn("uncertain_claim_equivalence", report["reasons"])
        self.assertEqual(report["disparity_bounds"], {"lower": 0.0, "upper": 1.0})
        self.assertIsNone(report["disparity"])
        self.assertEqual(report["coverage_by_group"]["AA"]["lower"], 0.0)
        self.assertEqual(report["coverage_by_group"]["AA"]["upper"], 1.0)

    def test_incomplete_dataset_is_reported_separately_from_semantic_uncertainty(self):
        from rdflib import URIRef
        self.graph.remove((URIRef("http://www.semanticweb.org/group2/runs/batch_fixture/tweet/AA_001"), None, None))
        client = oracle_client(self.gold)
        report = representation_check(self.graph, self.summary, client, "Obama", 0.2)
        self.assertFalse(report["data_complete"])
        self.assertEqual(report["source_tweet_counts"]["AA"], 29)
        self.assertEqual(report["matching"]["reason"], "incomplete_input_data")
        self.assertIn("semantic_assessment_not_run", report["reasons"])
        self.assertNotIn("uncertain_claim_equivalence", report["reasons"])
        client.chat.assert_not_called()

    def test_call_budget_is_enforced_without_dropping_source_claims(self):
        client = oracle_client(self.gold)
        report = representation_check(self.graph, self.summary, client, "Obama", 0.2,
                                      max_model_calls=1, batch_size=1)
        self.assertEqual(client.chat.call_count, 1)
        self.assertEqual(report["model_call_count"], 1)
        self.assertEqual(report["status"], "UNKNOWN")
        self.assertTrue(report["data_complete"])
        self.assertEqual(report["matching"]["reason"], "model_budget_exhausted")
        self.assertEqual({m["claim_id"] for c in report["matching"]["clusters"] for m in c["members"]}, set(self.gold))

    def test_malformed_later_response_preserves_progress_and_exact_call_count(self):
        client = oracle_client(self.gold)
        oracle = client.chat.side_effect
        counter = 0
        def respond(payload):
            nonlocal counter
            counter += 1
            return {} if counter == 2 else oracle(payload)
        client.chat.side_effect = respond
        report = representation_check(self.graph, self.summary, client, "Obama", 0.2, batch_size=1)
        self.assertEqual(report["status"], "UNKNOWN")
        self.assertEqual(report["model_call_count"], 2)
        self.assertEqual(report["execution_trace"][-1]["status"], "rejected")
        self.assertTrue(report["matched_claims"])
        self.assertTrue(report["uncertain_claims"])
        self.assertFalse(report["omitted_claims"])
        self.assertEqual({m["claim_id"] for c in report["matching"]["clusters"] for m in c["members"]}, set(self.gold))

    def test_cross_batch_fabricated_registry_reference_is_rejected(self):
        client = oracle_client(self.gold)
        oracle = client.chat.side_effect
        def respond(payload):
            response = oracle(payload)
            if "registry" in json.loads(payload["messages"][1]["content"]):
                data = json.loads(response["output"][0]["content"][0]["text"])
                data["decisions"][0].update(status="EQUIVALENT", equivalent_to="K999")
                response["output"][0]["content"][0]["text"] = json.dumps(data)
            return response
        client.chat.side_effect = respond
        report = representation_check(self.graph, self.summary, client, "Obama", 0.2, batch_size=1)
        self.assertEqual(report["status"], "UNKNOWN")
        self.assertEqual(report["matching"]["reason"], "invalid_or_failed_model_assessment")
        self.assertIn("registry reference", report["execution_trace"][-1]["error"])

    def test_valid_ids_with_mismatched_quote_pairs_are_rejected(self):
        client = oracle_client(self.gold)
        oracle = client.chat.side_effect
        def respond(payload):
            response = oracle(payload)
            body = json.loads(payload["messages"][1]["content"])
            if "registry" in body:
                data = json.loads(response["output"][0]["content"][0]["text"])
                target = data["decisions"][0]["equivalent_to"]
                if target:
                    data["decisions"][0]["source_evidence"][0]["quote"] = body["registry"][target]["source_evidence"][0]["source_quote"]
                    response["output"][0]["content"][0]["text"] = json.dumps(data)
            return response
        client.chat.side_effect = respond
        report = representation_check(self.graph, self.summary, client, "Obama", 0.2, batch_size=1)
        self.assertEqual(report["status"], "UNKNOWN")
        self.assertEqual(report["execution_trace"][-1]["status"], "rejected")
        self.assertIn("quotation is absent", report["execution_trace"][-1]["error"])

    def test_inconsistent_matching_across_equivalent_batches_remains_unknown(self):
        client = oracle_client(self.gold)
        oracle = client.chat.side_effect
        local_calls = 0
        def respond(payload):
            nonlocal local_calls
            response = oracle(payload)
            body = json.loads(payload["messages"][1]["content"])
            if "candidates" in body:
                local_calls += 1
                if local_calls == 2:
                    data = json.loads(response["output"][0]["content"][0]["text"])
                    data["clusters"][0].update(status="NOT_MATCHED", summary_claim_ids=[], summary_evidence=[])
                    response["output"][0]["content"][0]["text"] = json.dumps(data)
            return response
        client.chat.side_effect = respond
        report = representation_check(self.graph, self.summary, client, "Obama", 0.2, batch_size=1)
        self.assertEqual(report["matching"]["aggregation_status"], "ready")
        self.assertEqual(report["status"], "UNKNOWN")
        self.assertEqual(len(report["uncertain_claims"]), 1)
        self.assertEqual(len(report["omitted_claims"]), 2)
        self.assertIn("Conflicting matching", report["uncertain_claims"][0]["explanation"])

    def test_real_but_unshown_supporter_reference_is_rejected(self):
        client = oracle_client(self.gold)
        oracle = client.chat.side_effect
        unseen = next(claim for claim in get_group_claims(self.graph, "Obama", "AA")["claims"]
                      if claim["claim_id"] == "AA_004_C01")
        def respond(payload):
            response = oracle(payload)
            data = json.loads(response["output"][0]["content"][0]["text"])
            if "clusters" in data:
                data["clusters"][0]["source_evidence"][0].update(claim_id=unseen["claim_id"], quote=unseen["source_quote"])
                response["output"][0]["content"][0]["text"] = json.dumps(data)
            return response
        client.chat.side_effect = respond
        report = representation_check(self.graph, self.summary, client, "Obama", 0.2, batch_size=1)
        self.assertEqual(report["status"], "UNKNOWN")
        self.assertEqual(report["model_call_count"], 1)
        self.assertEqual(report["execution_trace"][0]["status"], "rejected")

    def test_deterministic_configurable_batches_and_canonical_ids(self):
        a, b = oracle_client(self.gold), oracle_client(self.gold)
        first = representation_check(self.graph, self.summary, a, "Obama", 0.2, batch_size=2)
        second = representation_check(self.graph, self.summary, b, "Obama", 0.2, batch_size=2)
        self.assertEqual(first, second)
        self.assertEqual([call.args[0] for call in a.chat.call_args_list], [call.args[0] for call in b.chat.call_args_list])
        sizes = [len(json.loads(call.args[0]["messages"][1]["content"]).get("candidates", {}))
                 for call in a.chat.call_args_list]
        self.assertTrue(all(size <= 2 for size in sizes))
        for invalid in (0, -1, True, "8"):
            with self.subTest(batch_size=invalid), self.assertRaises(ValueError):
                representation_check(self.graph, self.summary, Mock(), "Obama", 0.2, batch_size=invalid)

    def test_ninety_distinct_candidates_require_no_individual_pair_calls(self):
        graph, summary, gold = topic_fixture(unique=True)
        client = oracle_client(gold)
        report = representation_check(graph, summary, client, "Obama", 0.2)
        self.assertTrue(report["data_complete"])
        self.assertEqual(report["matching"]["aggregation_status"], "ready")
        self.assertEqual(len(report["matching"]["clusters"]), 90)
        self.assertEqual(report["status"], "PASS")
        self.assertLess(report["model_call_count"], 64)
        self.assertTrue(all(item["total_weight"] == 30 for item in report["coverage_by_group"].values()))
        self.assertTrue(all(e["input_bytes"] <= 12000 for e in report["execution_trace"]))


if __name__ == "__main__":
    unittest.main()
