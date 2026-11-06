"""Fact-checker failure regressions from the pilot, with real validators and deterministic model replies."""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

from fact_agent import run_fact_agent, _loop_schema
from fact_check import MAX_INPUT_BYTES, FactCheckError, compact_evidence, expand_verdict, prepare_assessment
from fact_tools import lookup_fact
from test_fact_agent import loop_response
from test_fact_check import controlled_fixture


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
        self.assertLessEqual(len(view["evidence_catalogue"]), 40)
        self.assertLessEqual(len(json.dumps(view["evidence_catalogue"], ensure_ascii=False).encode()), 12000)
        self.assertEqual(view["selection"]["omitted"], 300 - len(view["evidence_catalogue"]))
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
        self.assertLessEqual(len(json.dumps({"messages": payload["messages"], "format": payload["format"]},
                                            ensure_ascii=False).encode()), MAX_INPUT_BYTES)
        self.assertEqual(result["evidence_selection"]["shown"], 40)

    def test_insufficiency_is_not_an_automatic_contradiction(self):
        for text in ("Many users like Obama.", "Some users acknowledge his popularity."):
            client = Mock()
            client.chat.side_effect = [loop_response("lookup_fact"), loop_response("finalize", verdict="UNKNOWN",
                explanation="These tweets do not establish the claimed scope or an opposing fact.")]
            result = run_fact_agent(self.graph, {**self.claim, "claim": text}, client)
            self.assertEqual(result["verdict"], "UNKNOWN")
            self.assertIn("never count against a claim", client.chat.call_args.args[0]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
