"""One-step tool selection with mocked model responses; no live model required."""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

from rdflib import Graph, Literal, RDF, URIRef

from fact_agent import FactAgentError, run_fact_agent, run_fact_tool_step
from fact_tools import lookup_entity_facts, lookup_fact
from ontology_tools import NS
from data_loader import load_tweets
from populate_ontology import populate_ontology
from summarizer import GenerationError
from test_fact_check import controlled_fixture


def selection_client(action, entity_id=None, reason="Retrieve the appropriate source evidence."):
    client = Mock()
    client.chat.return_value = {"output": [{"content": [{"text": json.dumps({
        "action": action, "entity_id": entity_id, "reason": reason,
    })}]}]}
    return client


class FactAgentTests(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(identifier=URIRef("https://example.org/agent-fixture"))
        tweet = URIRef("https://example.org/agent-fixture/tweet/Obama_001")
        self.graph.add((tweet, RDF.type, NS.Tweet))
        self.graph.add((tweet, NS.tweetId, Literal("Obama_001")))
        self.graph.add((tweet, NS.tweetText, Literal("I like Obama's speech.")))
        self.graph.add((tweet, NS.extractionStatus, Literal("no_claim")))
        self.graph.add((NS.Obama, RDF.type, NS.Person))
        self.graph.add((NS.Obama, NS.hasName, Literal("Obama")))
        self.claim = {"claim_id": "S01_C01", "claim": "The author likes Obama's speech.",
                      "summary_quote": "The author likes Obama's speech.",
                      "target": "Obama's speech", "stance": "positive",
                      "cited_source_ids": ["Obama_001"], "verification_status": "unverified"}

    def test_lookup_fact_is_selected_and_only_that_tool_executes(self):
        client = selection_client("lookup_fact")
        before_graph, before_claim = set(self.graph), deepcopy(self.claim)
        with patch("fact_agent.lookup_fact", wraps=lookup_fact) as tweets, \
             patch("fact_agent.lookup_entity_facts") as entities:
            result = run_fact_tool_step(self.graph, self.claim, client, ["Obama"])
            tweets.assert_called_once_with(self.graph, self.claim)
            entities.assert_not_called()
        self.assertEqual(result["selected_tool"], "lookup_fact")
        self.assertEqual(result["tool_arguments"], {"summary_claim": self.claim})
        self.assertEqual(result["retrieved_evidence"]["tweets"][0]["text"], "I like Obama's speech.")
        self.assertEqual(result["retrieved_evidence"]["tweets"][0]["source_claims"], [])
        client.chat.assert_called_once()
        self.assertEqual(set(self.graph), before_graph)
        self.assertEqual(self.claim, before_claim)
        json.dumps(result)

    def test_allowed_entity_tool_is_selected_and_only_that_tool_executes(self):
        client = selection_client("lookup_entity_facts", "Obama")
        with patch("fact_agent.lookup_fact") as tweets, \
             patch("fact_agent.lookup_entity_facts", wraps=lookup_entity_facts) as entities:
            result = run_fact_tool_step(self.graph, self.claim, client, ["Obama"])
            tweets.assert_not_called()
            entities.assert_called_once_with(self.graph, "Obama")
        self.assertEqual(result["selected_tool"], "lookup_entity_facts")
        self.assertEqual(result["tool_arguments"], {"entity": "Obama"})
        self.assertEqual(result["retrieved_evidence"]["entity"], str(NS.Obama))
        self.assertEqual(len(result["retrieved_evidence"]["triples"]), 2)

    def test_no_tool_executes_neither_retrieval_and_records_skip(self):
        client = selection_client("no_tool", reason="Neither source is appropriate.")
        with patch("fact_agent.lookup_fact") as tweets, patch("fact_agent.lookup_entity_facts") as entities:
            result = run_fact_tool_step(self.graph, self.claim, client)
            tweets.assert_not_called()
            entities.assert_not_called()
        self.assertEqual(result["tool_arguments"], {})
        self.assertIsNone(result["retrieved_evidence"])
        self.assertEqual(result["selection_reason"], "Neither source is appropriate.")
        self.assertEqual(result["execution_trace"][-1]["status"], "skipped")

    def test_trace_records_selection_validation_and_actual_result(self):
        result = run_fact_tool_step(self.graph, self.claim, selection_client("lookup_fact"))
        trace = result["execution_trace"]
        self.assertEqual([step["step"] for step in trace],
                         ["model_selection", "validation", "tool_execution"])
        self.assertEqual(trace[0]["protocol"], "structured_json_action")
        self.assertEqual(trace[0]["selection"]["action"], result["selected_tool"])
        self.assertEqual(trace[1]["status"], "accepted")
        self.assertEqual(trace[2]["status"], "completed")
        self.assertEqual(trace[2]["result"], result["retrieved_evidence"])

    def test_invalid_actions_and_unauthorized_entities_execute_nothing(self):
        for action, entity, allowed in (
            ("fact_check", None, ["Obama"]), ("__import__", None, []),
            (["lookup_fact"], None, []), ("lookup_entity_facts", "McDonalds", ["Obama"]),
            ("lookup_entity_facts", "Obama", None), ("lookup_entity_facts", None, ["Obama"]),
            ("lookup_entity_facts", str(NS.Obama), ["Obama"]),
            ("lookup_fact", "Obama", ["Obama"]), ("no_tool", "Obama", ["Obama"]),
        ):
            with self.subTest(action=action, entity=entity), \
                 patch("fact_agent.lookup_fact") as tweets, \
                 patch("fact_agent.lookup_entity_facts") as entities:
                with self.assertRaises(FactAgentError):
                    run_fact_tool_step(self.graph, self.claim, selection_client(action, entity), allowed)
                tweets.assert_not_called()
                entities.assert_not_called()

    def test_full_iri_allowed_exactly_and_missing_entity_is_not_a_verdict(self):
        entity = str(NS.UnknownPerson)
        result = run_fact_tool_step(self.graph, self.claim,
                                    selection_client("lookup_entity_facts", entity), [entity])
        self.assertEqual(result["retrieved_evidence"]["status"], "unknown_entity")
        self.assertNotIn("verdict", result)

    def test_malformed_responses_are_rejected_before_execution(self):
        selections = [[], {"action": "lookup_fact"},
                      {"action": "lookup_fact", "entity_id": None, "reason": " "},
                      {"action": "lookup_fact", "entity_id": None, "reason": "Text", "arguments": {}},
                      {"action": "lookup_entity_facts", "entity_id": {}, "reason": "Text"}]
        responses = [{}, {"output": []}, {"output": [{"content": [{"text": "not JSON"}]}]}]
        responses += [{"output": [{"content": [{"text": json.dumps(s)}]}]} for s in selections]
        for response in responses:
            with self.subTest(response=response), \
                 patch("fact_agent.lookup_fact") as tweets, \
                 patch("fact_agent.lookup_entity_facts") as entities:
                client = Mock()
                client.chat.return_value = response
                with self.assertRaises(FactAgentError):
                    run_fact_tool_step(self.graph, self.claim, client, ["Obama"])
                tweets.assert_not_called()
                entities.assert_not_called()

    def test_payload_describes_tools_and_supplies_only_explicit_entity_ids(self):
        client = selection_client("no_tool")
        run_fact_tool_step(self.graph, self.claim, client, ("Obama", "Obama"))
        payload = client.chat.call_args.args[0]
        body = json.loads(payload["messages"][1]["content"])
        self.assertEqual(body["allowed_entity_ids"], ["Obama"])
        self.assertEqual(body["summary_claim"], self.claim)
        self.assertEqual(set(body["tools"]), {"lookup_fact", "lookup_entity_facts", "no_tool"})
        self.assertEqual(payload["format"]["properties"]["entity_id"]["anyOf"][1]["enum"], ["Obama"])
        self.assertNotIn("tools", payload)  # JSON selection, not native tool_calls.

    def test_invalid_inputs_fail_before_model_or_tools_are_called(self):
        client = selection_client("no_tool")
        for allowed in ("Obama", ["Obama speeches"], [None], ["https://example.org/Obama"]):
            with self.subTest(allowed=allowed), self.assertRaises(ValueError):
                run_fact_tool_step(self.graph, self.claim, client, allowed)
        for claim in ({}, {**self.claim, "claim": " "}, {**self.claim, "cited_source_ids": "Obama_001"}):
            with self.subTest(claim=claim), self.assertRaises(ValueError):
                run_fact_tool_step(self.graph, claim, client)
        with self.assertRaises(TypeError):
            run_fact_tool_step(None, self.claim, client)
        with self.assertRaises(TypeError):
            run_fact_tool_step(self.graph, [], client)
        client.chat.assert_not_called()


def loop_response(action, entity_id=None, verdict=None, explanation=None, references=None,
                  reason="Choose the next useful investigation action."):
    return {"output": [{"content": [{"text": json.dumps({
        "action": action, "entity_id": entity_id, "reason": reason,
        "verdict": verdict, "explanation": explanation, "references": references or [],
    })}]}]}


def fixture_verdict(verdict="SUPPORTED", evidence_id=None, quote="I like Obama's speech."):
    evidence_id = evidence_id or "http://www.semanticweb.org/group2/runs/fact_check_fixture/tweet/Obama_fixture"
    return loop_response("finalize", verdict=verdict,
                         explanation="The author's original tweet explicitly expresses liking the speech.",
                         references=[{"evidence_id": evidence_id, "quote": quote}])


class BoundedFactAgentTests(unittest.TestCase):
    def setUp(self):
        self.graph, self.claim = controlled_fixture()

    def test_one_tool_then_verdict_observes_evidence_without_hidden_retrieval(self):
        client = Mock()
        client.chat.side_effect = [loop_response("lookup_fact"), fixture_verdict()]
        before_graph, before_claim = set(self.graph), deepcopy(self.claim)
        with patch("fact_agent.lookup_fact", wraps=lookup_fact) as tweets, \
             patch("fact_check.lookup_fact", side_effect=AssertionError("Hidden retrieval")), \
             patch("fact_agent.lookup_entity_facts") as entities:
            result = run_fact_agent(self.graph, self.claim, client)
            tweets.assert_called_once_with(self.graph, self.claim)
            entities.assert_not_called()
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertEqual(result["stop_reason"], "verdict")
        self.assertEqual(result["tool_call_count"], 1)
        self.assertEqual(result["model_call_count"], 2)
        self.assertTrue(result["evidence_complete"])
        self.assertEqual(result["review_status"], "unreviewed")
        second_payload = client.chat.call_args_list[1].args[0]
        observation = json.loads(second_payload["messages"][-1]["content"])["observation"]
        self.assertEqual(observation["result"]["tweets"][0]["text"], "I like Obama's speech.")
        self.assertIn(result["references"][0]["evidence_id"], observation["evidence_catalogue"])
        self.assertEqual(len(client.chat.call_args_list[0].args[0]["messages"]), 2)
        self.assertEqual([item["action"] for item in result["model_decisions"]], ["lookup_fact", "finalize"])
        self.assertEqual(result["execution_trace"][-1]["step"], "stop")
        self.assertEqual(set(self.graph), before_graph)
        self.assertEqual(self.claim, before_claim)
        json.dumps(result)

    def test_two_different_tools_then_verdict_preserve_both_observations(self):
        client = Mock()
        client.chat.side_effect = [loop_response("lookup_entity_facts", "Obama"),
                                   loop_response("lookup_fact"), fixture_verdict()]
        with patch("fact_agent.lookup_fact", wraps=lookup_fact) as tweets, \
             patch("fact_agent.lookup_entity_facts", wraps=lookup_entity_facts) as entities:
            result = run_fact_agent(self.graph, self.claim, client, ["Obama"])
            tweets.assert_called_once()
            entities.assert_called_once_with(self.graph, "Obama")
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertEqual(result["tool_call_count"], 2)
        self.assertEqual([call["tool"] for call in result["tool_calls"]], ["lookup_entity_facts", "lookup_fact"])
        self.assertEqual(result["ontology"]["entity"], str(NS.Obama))
        self.assertTrue(any(item["kind"] == "ontology_assertion" for item in result["evidence"].values()))
        last_messages = client.chat.call_args_list[-1].args[0]["messages"]
        self.assertEqual(len(last_messages), 6)
        self.assertIn("ontology_assertion", last_messages[-1]["content"])

    def test_explicit_unknown_can_stop_without_any_retrieval(self):
        client = Mock()
        client.chat.return_value = loop_response("finalize", verdict="UNKNOWN",
                                                 explanation="No appropriate supporting evidence is available.")
        with patch("fact_agent.lookup_fact") as tweets, patch("fact_agent.lookup_entity_facts") as entities:
            result = run_fact_agent(self.graph, self.claim, client)
            tweets.assert_not_called()
            entities.assert_not_called()
        self.assertEqual(result["stop_reason"], "explicit_unknown")
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["tool_call_count"], 0)
        self.assertFalse(result["evidence_complete"])

    def test_tool_limit_prevents_a_fourth_execution(self):
        client = Mock()
        client.chat.side_effect = [loop_response("lookup_entity_facts", entity)
                                   for entity in ("Obama", "McDonalds", "Person1")] + [loop_response("lookup_fact")]
        with patch("fact_agent.lookup_fact") as tweets:
            result = run_fact_agent(self.graph, self.claim, client, ["Obama", "McDonalds", "Person1"])
            tweets.assert_not_called()
        self.assertEqual(result["tool_call_count"], 3)
        self.assertEqual(result["model_call_count"], 4)
        self.assertEqual(result["stop_reason"], "tool_call_limit")
        self.assertEqual(result["verdict"], "UNKNOWN")

    def test_identical_calls_stop_without_reexecution(self):
        client = Mock()
        client.chat.side_effect = [loop_response("lookup_fact"), loop_response("lookup_fact")]
        with patch("fact_agent.lookup_fact", wraps=lookup_fact) as tweets:
            result = run_fact_agent(self.graph, self.claim, client)
            tweets.assert_called_once()
        self.assertEqual(result["stop_reason"], "repeated_tool_call")
        self.assertEqual(result["tool_call_count"], 1)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(len(result["tool_selections"]), 2)

    def test_local_and_full_entity_aliases_cannot_bypass_repetition_guard(self):
        client = Mock()
        client.chat.side_effect = [loop_response("lookup_entity_facts", "Obama"),
                                   loop_response("lookup_entity_facts", str(NS.Obama))]
        result = run_fact_agent(self.graph, self.claim, client, ["Obama", str(NS.Obama)])
        self.assertEqual(result["stop_reason"], "repeated_tool_call")
        self.assertEqual(result["tool_call_count"], 1)

    def test_unauthorized_entity_stops_unknown_and_preserves_prior_trace(self):
        client = Mock()
        client.chat.side_effect = [loop_response("lookup_fact"), loop_response("lookup_entity_facts", "McDonalds")]
        with patch("fact_agent.lookup_entity_facts") as entities:
            result = run_fact_agent(self.graph, self.claim, client, ["Obama"])
            entities.assert_not_called()
        self.assertEqual(result["stop_reason"], "invalid_tool_request")
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["tool_call_count"], 1)
        self.assertEqual(result["tool_calls"][0]["result"]["tweets"][0]["tweet_id"], "Obama_fixture")

    def test_fabricated_tweet_claim_entity_and_quote_references_are_rejected(self):
        for evidence_id, quote in (
            ("Obama_nonexistent", "I like"),
            ("http://www.semanticweb.org/group2/runs/fact_check_fixture/claim/Invented_C01", "I like"),
            ("ontology:001", "55"),  # No entity tool was selected.
            ("http://www.semanticweb.org/group2/runs/fact_check_fixture/tweet/Obama_fixture", "I dislike"),
        ):
            with self.subTest(evidence_id=evidence_id, quote=quote):
                client = Mock()
                client.chat.side_effect = [loop_response("lookup_fact"), fixture_verdict(evidence_id=evidence_id, quote=quote)]
                result = run_fact_agent(self.graph, self.claim, client)
                self.assertEqual(result["stop_reason"], "invalid_verdict")
                self.assertEqual(result["verdict"], "UNKNOWN")
                self.assertEqual(result["references"], [])

    def test_source_claim_id_and_exact_original_quote_can_be_referenced(self):
        client = Mock()
        client.chat.side_effect = [loop_response("lookup_fact"), fixture_verdict(
            evidence_id="http://www.semanticweb.org/group2/runs/fact_check_fixture/claim/Obama_fixture_C01")]
        result = run_fact_agent(self.graph, self.claim, client)
        self.assertEqual(result["verdict"], "SUPPORTED")
        supporting = next(iter(result["supporting_evidence"].values()))
        self.assertEqual(supporting["claim_id"], "Obama_fixture_C01")

    def test_missing_source_citations_remain_explicit(self):
        client = Mock()
        client.chat.side_effect = [loop_response("lookup_fact"), loop_response(
            "finalize", verdict="UNKNOWN", explanation="The cited tweet was not retrieved.")]
        result = run_fact_agent(self.graph, {**self.claim, "cited_source_ids": ["Obama_missing"]}, client)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertFalse(result["evidence_complete"])
        self.assertEqual(result["retrieval"]["missing_tweet_ids"], ["Obama_missing"])

    def test_incomplete_five_tweet_sample_cannot_finalize_supported(self):
        tweets = load_tweets("Obama")[:5]
        graph = populate_ontology(tweets, [{"tweet_id": t.id, "dialect_label": t.dialect_label,
                                          "status": "no_claim", "claims": []} for t in tweets], run_id="five_tweet_test")
        claim = {**self.claim, "cited_source_ids": [t.id for t in tweets] + ["Obama_006"]}
        client = Mock()
        client.chat.side_effect = [loop_response("lookup_fact"), fixture_verdict(
            evidence_id="http://www.semanticweb.org/group2/runs/five_tweet_test/tweet/Obama_001",
            quote=tweets[0].text)]
        result = run_fact_agent(graph, claim, client)
        self.assertEqual(result["stop_reason"], "incomplete_evidence")
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["reason"], "incomplete_cited_evidence")
        self.assertIn("full 90-tweet", result["explanation"])
        self.assertFalse(result["evidence_complete"])
        self.assertEqual(result["model_decisions"][-1]["verdict"], "SUPPORTED")

    def test_no_tool_when_evidence_unavailable_executes_nothing(self):
        client = Mock()
        client.chat.return_value = loop_response("no_tool", reason="There are no cited tweets or allowed entities.")
        with patch("fact_agent.lookup_fact") as tweets, patch("fact_agent.lookup_entity_facts") as entities:
            result = run_fact_agent(self.graph, {**self.claim, "cited_source_ids": []}, client)
            tweets.assert_not_called()
            entities.assert_not_called()
        self.assertEqual(result["stop_reason"], "no_tool")
        self.assertEqual(result["tool_call_count"], 0)
        self.assertEqual(result["verdict"], "UNKNOWN")

    def test_missing_requested_entity_never_becomes_falsehood(self):
        client = Mock()
        client.chat.side_effect = [loop_response("lookup_entity_facts", "MissingEntity"),
                                   loop_response("lookup_fact"), fixture_verdict("CONTRADICTED")]
        result = run_fact_agent(self.graph, self.claim, client, ["MissingEntity"])
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["reason"], "missing_ontology_entity")
        self.assertFalse(result["evidence_complete"])

    def test_three_calls_can_be_followed_by_final_verdict_and_stable_reference_ids(self):
        client = Mock()
        client.chat.side_effect = [loop_response("lookup_entity_facts", "Obama"),
                                   loop_response("lookup_entity_facts", "Person1"),
                                   loop_response("lookup_fact"), fixture_verdict()]
        result = run_fact_agent(self.graph, self.claim, client, ["Obama", "Person1"])
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertEqual(result["tool_call_count"], 3)
        first = json.loads(client.chat.call_args_list[1].args[0]["messages"][-1]["content"])["observation"]
        second = json.loads(client.chat.call_args_list[2].args[0]["messages"][-1]["content"])["observation"]
        for key, item in first["evidence_catalogue"].items():
            self.assertEqual(second["evidence_catalogue"][key]["quote_text"], item["quote_text"])
        self.assertEqual(second["result"]["entity"], str(NS.Person1))

    def test_zero_or_invalid_budget_and_malformed_decisions_are_bounded(self):
        client = Mock()
        result = run_fact_agent(self.graph, self.claim, client, max_tool_calls=0)
        self.assertEqual(result["stop_reason"], "tool_call_limit")
        client.chat.assert_not_called()
        for budget in (-1, True, "3"):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                run_fact_agent(self.graph, self.claim, client, max_tool_calls=budget)
        for response in ({}, {"output": [{"content": [{"text": "not JSON"}]}]},
                         selection_client("lookup_fact").chat.return_value):
            with self.subTest(response=response):
                client = Mock()
                client.chat.return_value = response
                result = run_fact_agent(self.graph, self.claim, client)
                self.assertEqual(result["stop_reason"], "invalid_tool_request")
                self.assertEqual(result["tool_call_count"], 0)

    def test_model_transport_failure_preserves_observations_and_stops_unknown(self):
        client = Mock()
        client.chat.side_effect = [loop_response("lookup_fact"), GenerationError("Controlled model timeout")]
        result = run_fact_agent(self.graph, self.claim, client)
        self.assertEqual(result["stop_reason"], "model_error")
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["tool_call_count"], 1)
        self.assertEqual(result["model_call_count"], 2)
        self.assertTrue(result["retrieval"]["tweets"])


if __name__ == "__main__":
    unittest.main()
