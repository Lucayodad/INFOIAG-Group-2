"""Real StateGraph and checkers, with complete synthetic data and mocked Qwen."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from rdflib import Literal
from rdflib.graph import ModificationException

from fakes import demo_fixture, mock_qwen, response
from fact_check import fact_check_summary
from ontology_tools import NS
from summary_agent import print_run_summary, run_summary_agent
from summary_claims import extract_sentence
from summarizer import GenerationError


class SummaryAgentTests(unittest.TestCase):
    def run_agent(self, graph, draft, client, **options):
        options.setdefault("max_revisions", 0)  # check only, unless a test allows revisions
        return run_summary_agent(graph, draft, topic="Obama", prompt=draft["summary_prompt"], client=client, **options)

    def test_a_contradiction_visits_each_node_and_stops_at_the_revision_bound(self):
        graph, draft = demo_fixture()
        before_graph, before_draft = set(graph), deepcopy(draft)
        client = mock_qwen(draft)
        with patch("summary_claims.extract_sentence", wraps=extract_sentence) as extract, \
             patch("summary_agent.fact_check_summary", wraps=fact_check_summary) as facts:
            result = self.run_agent(graph, draft, client)
        self.assertEqual(extract.call_count, 5)
        facts.assert_called_once()
        self.assertEqual(result["status"], "stopped_with_issues")
        self.assertEqual(result["route"], "finalize")
        self.assertEqual(result["stop_reason"], "revision_limit")
        self.assertEqual(result["revision_count"], 0)
        self.assertEqual(result["revisions"], [])
        self.assertIn("factual_uncertainty", result["uncertainty_flags"])
        self.assertEqual(result["factual_results"]["verdict_counts"],
                         {"SUPPORTED": 1, "CONTRADICTED": 1, "UNKNOWN": 1})
        self.assertEqual(result["representation_results"]["status"], "NOT_RUN")
        history = result["execution_history"]
        self.assertEqual([event["node"] for event in history],
                         ["extract_summary_claims", "factual_check", "route_checks", "finalize"])
        for left, right in zip(history, history[1:]):
            self.assertEqual(left["after"], right["before"])
        self.assertEqual(set(graph), before_graph)
        self.assertEqual(draft, before_draft)
        self.assertEqual(result["initial_summary"], draft)
        self.assertEqual(result["current_summary"], draft)
        self.assertEqual(client.chat.call_count, 11)  # 5 extractions, 3 claims x (lookup_fact + finalize)
        client.close.assert_not_called()
        json.dumps(result)
        # The checker receives a read-only view of the evidence.
        evidence = facts.call_args.args[0]
        with self.assertRaises(ModificationException):
            evidence.add((NS.Example, NS.hasText, Literal("No writes")))

    def test_no_serious_issues_finalize_without_revision(self):
        graph, draft = demo_fixture(contradiction=False)
        result = self.run_agent(graph, draft, mock_qwen(draft, ("SUPPORTED",) * 3), max_revisions=3)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["route"], "finalize")
        self.assertEqual(result["stop_reason"], "no_confirmed_serious_issues")
        self.assertEqual(result["uncertainty_flags"], [])
        self.assertEqual(result["revisions"], [])
        self.assertNotIn("retrieve_revision_evidence", [row["node"] for row in result["execution_history"]])

    def test_unknown_alone_finalizes_with_uncertainty(self):
        graph, draft = demo_fixture(contradiction=False)
        result = self.run_agent(graph, draft, mock_qwen(draft, ("UNKNOWN",) * 3), max_revisions=3)
        self.assertEqual(result["factual_results"]["status"], "UNKNOWN")
        self.assertEqual(result["status"], "completed_with_uncertainty")
        self.assertEqual(result["route"], "finalize")
        self.assertEqual(result["stop_reason"], "uncertainty_only")
        self.assertTrue(result["uncertainty_flags"])
        self.assertEqual(result["revisions"], [])

    def test_failed_extraction_remains_unclear(self):
        graph, draft = demo_fixture()
        for malformed in (GenerationError("Local model unavailable"), response({"bad": "output"})):
            with self.subTest(response=type(malformed).__name__):
                client = Mock()
                client.chat.side_effect = [malformed] * 10  # each sentence: attempt + one retry
                result = self.run_agent(graph, draft, client)
                self.assertEqual(len(result["extraction_errors"]), 5)
                self.assertTrue(all(record["status"] == "unclear" for record in result["summary_claims"]))
                self.assertEqual(result["status"], "completed_with_uncertainty")
                self.assertIn("summary_extraction_failed", result["uncertainty_flags"])
                self.assertEqual(client.chat.call_count, 10)

    def test_bad_inputs_fail_before_model_calls(self):
        graph, draft = demo_fixture()
        client = Mock()
        for change in ({"initial_summary": {}}, {"topic": "missing"}, {"prompt": ""},
                       {"max_tool_calls": -1}, {"max_revisions": 4}, {"proportion_check": 1}, {"revision_top_k": 0}):
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                arguments = dict(graph=graph, initial_summary=draft, topic="Obama", prompt="Test", client=client)
                arguments.update(change)
                run_summary_agent(**arguments)
        client.chat.assert_not_called()

    def test_default_client_is_closed_on_success_and_failure(self):
        graph, draft = demo_fixture()
        client = mock_qwen(draft)
        with patch("summary_agent.OllamaClient", return_value=client):
            run_summary_agent(graph, draft, topic="Obama", prompt="Test", max_revisions=0)
        client.close.assert_called_once()
        client.reset_mock()
        with patch("summary_agent.OllamaClient", return_value=client), self.assertRaises(ValueError):
            run_summary_agent(graph, draft, topic="Obama", prompt="Test", max_tool_calls=-1)
        client.close.assert_called_once()

    def test_run_summary_is_concise(self):
        graph, draft = demo_fixture()
        result = self.run_agent(graph, draft, mock_qwen(draft))
        with redirect_stdout(io.StringIO()) as printed:
            print_run_summary(result, Path("run.json"))
        text = printed.getvalue()
        for expected in ("Factual: FAIL", "no group check in the agent", "Revisions: 0 attempted",
                         "Final status: stopped_with_issues", str(Path("run.json").resolve())):
            self.assertIn(expected, text)
        self.assertNotIn("execution_history", text)


if __name__ == "__main__":
    unittest.main()
