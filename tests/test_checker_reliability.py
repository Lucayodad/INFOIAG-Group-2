"""Pilot failure regressions with real validators and deterministic model replies."""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

from fact_agent import run_fact_agent, _loop_schema
from fact_check import FactCheckError, compact_evidence, expand_verdict, prepare_assessment
from fact_tools import lookup_fact
from representation_check import match_summary_claims
from representation_tools import get_group_claims
from test_fact_agent import loop_response
from test_fact_check import controlled_fixture
from test_representation_batches import oracle_client, topic_fixture


def response(data):
    return {"output": [{"content": [{"text": json.dumps(data)}]}]}


class FactualReliabilityTests(unittest.TestCase):
    def test_observed_evidence_schema_requires_verdict_and_exact_reference_pairs(self):
        catalogue = {"E001": {"quote_text": "I like Obama's speech."},
                     "E002": {"quote_text": "Needs rest and inspiration IMO"}}
        properties = _loop_schema([], catalogue)["properties"]
        self.assertEqual(properties["action"]["enum"], ["finalize"])
        self.assertNotIn("anyOf", properties["verdict"])
        choices = properties["references"]["items"]["anyOf"]
        self.assertEqual([(c["properties"]["evidence_id"]["enum"][0], c["properties"]["quote"]["enum"][0])
                          for c in choices], [(key, value["quote_text"]) for key, value in catalogue.items()])
        self.assertEqual(_loop_schema(["Obama"], catalogue)["properties"]["action"]["enum"],
                         ["finalize", "lookup_entity_facts"])
        ontology = {"E001": {"kind": "ontology_triple", "quote_text": "{}"}}
        self.assertIn("lookup_fact", _loop_schema(["Obama"], ontology)["properties"]["action"]["enum"])

    def setUp(self):
        self.graph, self.claim = controlled_fixture()
        self.assessment = prepare_assessment(self.claim, lookup_fact(self.graph, self.claim))

    def test_short_source_claim_id_restores_full_rdf_provenance(self):
        client = Mock()
        def reply(payload):
            if len(payload["messages"]) == 2:
                return loop_response("lookup_fact")
            catalogue = json.loads(payload["messages"][-1]["content"])["observation"]["evidence_catalogue"]
            short, item = next((k, v) for k, v in catalogue.items() if v["kind"] == "source_claim")
            return loop_response("finalize", verdict="SUPPORTED", explanation="The original author likes the speech.",
                                 references=[{"evidence_id": short, "quote": item["quote_text"]}])
        client.chat.side_effect = reply
        result = run_fact_agent(self.graph, self.claim, client)
        original = result["references"][0]["evidence_id"]
        self.assertTrue(original.endswith("/claim/Obama_fixture_C01"))
        self.assertEqual(result["supporting_evidence"][original]["source_quote"], "I like Obama's speech.")
        self.assertIn(original, result["evidence_id_map"].values())

    def test_missing_explanation_gets_exactly_one_repair_without_retrieval(self):
        client = Mock()
        client.chat.side_effect = [loop_response("lookup_fact"), loop_response("finalize", verdict="UNKNOWN"),
            loop_response("finalize", verdict="UNKNOWN", explanation="One tweet cannot establish how many users agree.")]
        result = run_fact_agent(self.graph, self.claim, client)
        self.assertEqual(result["model_call_count"], 3)
        self.assertEqual(result["tool_call_count"], 1)
        self.assertEqual(result["stop_reason"], "explicit_unknown")
        self.assertTrue(result["execution_trace"][-2]["format_repair"])
        self.assertTrue(result["explanation"].strip())

    def test_format_repair_cannot_flip_to_opposing_verdict(self):
        client = Mock()
        reference = [{"evidence_id": "E001", "quote": "I like Obama's speech."}]
        client.chat.side_effect = [loop_response("lookup_fact"),
            loop_response("finalize", verdict="CONTRADICTED", explanation="An opposing assertion.",
                          references=[{"evidence_id": "E001", "quote": "invented quote"}]),
            loop_response("finalize", verdict="SUPPORTED", explanation="Changed conclusion during format repair.",
                          references=reference)]
        result = run_fact_agent(self.graph, self.claim, client)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["model_call_count"], 3)
        self.assertIn("opposing verdict", result["explanation"])

    def test_persistent_missing_explanation_is_unknown_after_one_repair(self):
        client = Mock()
        client.chat.side_effect = [loop_response("lookup_fact")] + [loop_response("finalize", verdict="UNKNOWN")] * 2
        result = run_fact_agent(self.graph, self.claim, client)
        self.assertEqual(result["stop_reason"], "invalid_verdict")
        self.assertEqual(result["model_call_count"], 3)
        self.assertEqual(result["references"], [])

    def test_initial_reference_schema_has_no_empty_enum(self):
        client = Mock()
        client.chat.return_value = loop_response("no_tool", reason="No appropriate evidence source.")
        run_fact_agent(self.graph, self.claim, client)
        def inspect(value):
            if isinstance(value, dict):
                if "enum" in value:
                    self.assertTrue(value["enum"])
                for child in value.values():
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)
        inspect(client.chat.call_args.args[0]["format"])

    def test_large_evidence_is_bounded_with_auditable_selection_and_stable_ids(self):
        original = self.assessment["evidence"]
        item = next(v for v in original.values() if v["kind"] == "tweet")
        large = {**self.assessment, "evidence": {f"urn:tweet:{i:03d}": {**item, "tweet_id": f"T{i:03d}"} for i in range(300)}}
        view = compact_evidence(large)
        self.assertLessEqual(len(view["evidence_catalogue"]), 16)
        self.assertLessEqual(len(json.dumps(view["evidence_catalogue"], ensure_ascii=False).encode()), 8000)
        self.assertEqual(view["selection"]["omitted"], 284)
        later = compact_evidence({**large, "evidence": {**large["evidence"], "urn:extra": item}}, view["all_evidence_id_map"])
        for key, value in view["all_evidence_id_map"].items():
            self.assertEqual(later["all_evidence_id_map"][key], value)
        self.assertEqual(len(large["evidence"]), 300)
        hidden = next(key for key in large["evidence"] if key not in view["evidence_id_map"].values())
        for evidence_id in ("E999", hidden, "invented"):
            with self.subTest(evidence_id=evidence_id), self.assertRaises(FactCheckError):
                expand_verdict({"verdict": "SUPPORTED", "explanation": "Claimed support.",
                    "references": [{"evidence_id": evidence_id, "quote": item["quote_text"]}]}, view, large)

    def test_large_tool_result_is_not_repeated_in_model_context(self):
        retrieval = deepcopy(self.assessment["retrieval"])
        tweet = retrieval["tweets"][0]
        retrieval["tweets"] = [{**tweet, "tweet_id": f"T{i}", "tweet_uri": f"urn:T{i}", "source_claims": []} for i in range(200)]
        client = Mock()
        client.chat.side_effect = [loop_response("lookup_fact"), loop_response("finalize", verdict="UNKNOWN",
            explanation="The selected tweets do not establish corpus frequency.")]
        with patch("fact_agent.lookup_fact", return_value=retrieval):
            result = run_fact_agent(self.graph, self.claim, client)
        self.assertEqual(len(result["retrieval"]["tweets"]), 200)
        payload = client.chat.call_args.args[0]
        self.assertNotIn('"result":', payload["messages"][-1]["content"])
        self.assertLessEqual(len(json.dumps(payload["messages"], ensure_ascii=False).encode()), 16000)
        self.assertEqual(result["evidence_selection"]["shown"], 16)

    def test_insufficiency_is_not_an_automatic_contradiction(self):
        for text in ("Many users like Obama.", "Some users acknowledge his popularity."):
            client = Mock()
            client.chat.side_effect = [loop_response("lookup_fact"), loop_response("finalize", verdict="UNKNOWN",
                explanation="These tweets do not establish the claimed scope or an opposing fact.")]
            result = run_fact_agent(self.graph, {**self.claim, "claim": text}, client)
            self.assertEqual(result["verdict"], "UNKNOWN")
            self.assertIn("Negative sentiment", client.chat.call_args.args[0]["messages"][0]["content"])


class RepresentationReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.graph, self.summary, self.gold = topic_fixture()
        self.groups = {g: get_group_claims(self.graph, "Obama", g) for g in ("AA", "Hisp", "White")}

    def run_with_mutation(self, mutation, *, persistent=False, **options):
        client = oracle_client(self.gold)
        oracle = client.chat.side_effect
        failures = 0
        def respond(payload):
            nonlocal failures
            reply = oracle(payload)
            body = json.loads(payload["messages"][1]["content"])
            if "candidates" in body and "summary_claims" not in body and (persistent or failures == 0):
                data = json.loads(reply["output"][0]["content"][0]["text"])
                mutation(data, body)
                failures += 1
                return response(data)
            return reply
        client.chat.side_effect = respond
        return match_summary_claims(self.groups, self.summary, client, **options), client

    def test_incomplete_partition_is_retried_and_repaired(self):
        report, client = self.run_with_mutation(lambda data, body: data["clusters"].pop(), batch_size=20)
        self.assertEqual(report["aggregation_status"], "ready")
        rejected = [e for e in report["execution_trace"] if e["status"] == "rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertIn("exactly once", rejected[0]["error"])
        self.assertEqual(report["model_call_count"], client.chat.call_count)

    def test_persistent_failed_batches_do_not_abort_independent_coverage(self):
        client = oracle_client(self.gold)
        oracle = client.chat.side_effect
        def respond(payload):
            body = json.loads(payload["messages"][1]["content"])
            if "incoming" in body and "K002" in body["incoming"]:
                return response({"decisions": []})
            return oracle(payload)
        client.chat.side_effect = respond
        report = match_summary_claims(self.groups, self.summary, client, batch_size=2)
        self.assertEqual(report["aggregation_status"], "UNKNOWN")
        self.assertEqual(len([e for e in report["execution_trace"] if e["status"] == "rejected"]), 2)
        self.assertTrue(any(e["step"] == "summary_coverage" and e["status"] == "completed" for e in report["execution_trace"]))
        self.assertTrue(any(c["status"] != "UNKNOWN" for c in report["clusters"]))
        self.assertEqual({m["claim_id"] for c in report["clusters"] for m in c["members"]}, set(self.gold))
        self.assertTrue(all(c["status"] == "UNKNOWN" for c in report["clusters"] if "K001" in c["candidate_ids"]))

    def test_invalid_stance_merge_is_rejected_without_accepting_bad_partition(self):
        def mutate(data, body):
            data["clusters"] = [{"candidate_ids": list(body["candidates"]), "source_evidence": [
                {"claim_id": q["claim_id"], "quote": q["source_quote"]}
                for c in body["candidates"].values() for q in c["source_evidence"]], "explanation": "Same Obama topic."}]
        report, client = self.run_with_mutation(mutate, persistent=True, batch_size=20)
        self.assertEqual(report["aggregation_status"], "UNKNOWN")
        self.assertEqual(client.chat.call_count, 2)
        self.assertTrue(all(c["status"] == "UNKNOWN" for c in report["clusters"]))
        self.assertTrue(all("stances cannot be merged" in e["error"] for e in report["execution_trace"]))

    def test_equivalence_and_coverage_have_separate_bounded_inputs(self):
        client = oracle_client(self.gold)
        report = match_summary_claims(self.groups, self.summary, client, batch_size=2)
        for event in report["execution_trace"]:
            self.assertLessEqual(event["input_bytes"], 12000)
            self.assertLessEqual(event["request_bytes"], 30000)
            if event["step"] == "semantic_equivalence":
                self.assertNotIn("summary_claims", event["input"])
            if event["step"] == "summary_coverage":
                data = json.loads(event["raw_response"])
                self.assertTrue(all(len(c["candidate_ids"]) == 1 for c in data["clusters"]))
        self.assertTrue(any(e["step"] == "summary_coverage" for e in report["execution_trace"]))

    def test_retry_counts_toward_global_budget_and_unknown_is_conservative(self):
        report, client = self.run_with_mutation(lambda data, body: data["clusters"].clear(), persistent=True,
                                                max_model_calls=1, batch_size=2)
        self.assertLessEqual(client.chat.call_count, 1)
        self.assertEqual(report["reason"], "model_budget_exhausted")
        self.assertEqual(report["aggregation_status"], "UNKNOWN")
        self.assertTrue(all(c["status"] == "UNKNOWN" for c in report["clusters"]))


if __name__ == "__main__":
    unittest.main()
