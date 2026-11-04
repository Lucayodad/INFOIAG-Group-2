"""Real StateGraph and existing checkers, with complete synthetic data and mocked Qwen."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from rdflib import Literal
from rdflib.graph import ModificationException

from demo_summary_agent import EXPECTED_COUNTS, demo_fixture, mock_qwen, response
from fact_check import fact_check_summary
from ontology_tools import NS
from representation_check import representation_check
from summary_agent import main, run_summary_agent
from summary_claims import extract_sentence
from summarizer import GenerationError


class SummaryAgentTests(unittest.TestCase):
    def run_agent(self, graph, draft, client, **options):
        options.setdefault("max_revisions", 0)  # Preserve check-only coverage of the original graph.
        options.setdefault("representation_metric", "internal")
        return run_summary_agent(graph, draft, topic="Obama", prompt=draft["summary_prompt"],
                                 client=client, expected_group_counts=EXPECTED_COUNTS, **options)

    def test_serious_issues_visit_real_nodes_and_end_pending_revision(self):
        graph, draft = demo_fixture()
        before_graph, before_draft = set(graph), deepcopy(draft)
        client = mock_qwen(draft)
        with patch("summary_agent.extract_sentence", wraps=extract_sentence) as extract, \
             patch("summary_agent.fact_check_summary", wraps=fact_check_summary) as facts, \
             patch("summary_agent.representation_check", wraps=representation_check) as representation:
            result = self.run_agent(graph, draft, client, disparity_threshold=0.25)
        self.assertEqual(extract.call_count, 5)
        facts.assert_called_once()
        representation.assert_called_once()
        self.assertEqual(result["status"], "stopped_with_issues")
        self.assertEqual(result["route"], "finalize")
        self.assertEqual(result["stop_reason"], "revision_limit")
        self.assertEqual(result["revision_count"], 0)
        self.assertEqual(result["revisions"], [])
        self.assertIn("factual_uncertainty", result["uncertainty_flags"])
        self.assertEqual(result["factual_results"]["verdict_counts"],
                         {"SUPPORTED": 1, "CONTRADICTED": 1, "UNKNOWN": 1})
        rep = result["representation_results"]
        self.assertEqual(rep["status"], "FAIL")
        self.assertAlmostEqual(rep["disparity"], 1 / 3)
        self.assertEqual(rep["source_tweet_counts"], EXPECTED_COUNTS)
        history = result["execution_history"]
        self.assertEqual([event["node"] for event in history],
                         ["extract_summary_claims", "factual_check", "representation_check",
                          "route_checks", "finalize"])
        for left, right in zip(history, history[1:]):
            self.assertEqual(left["after"], right["before"])
        self.assertEqual(set(graph), before_graph)
        self.assertEqual(draft, before_draft)
        self.assertEqual(result["initial_summary"], draft)
        self.assertEqual(result["current_summary"], draft)
        self.assertEqual(client.chat.call_count, 13)
        client.close.assert_not_called()
        json.dumps(result)
        # The wrapper supplied to both checking nodes actually rejects writes.
        evidence = facts.call_args.args[0]
        with self.assertRaises(ModificationException):
            evidence.add((NS.Example, NS.hasText, Literal("No writes")))

    def test_no_serious_issues_finalize_without_revision(self):
        graph, draft = demo_fixture(contradiction=False)
        result = self.run_agent(graph, draft, mock_qwen(draft, ("SUPPORTED",) * 3), disparity_threshold=0.5)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["route"], "finalize")
        self.assertEqual(result["stop_reason"], "no_confirmed_serious_issues")
        self.assertEqual(result["uncertainty_flags"], [])
        self.assertEqual(result["revisions"], [])
        self.assertNotIn("retrieve_revision_evidence", [row["node"] for row in result["execution_history"]])

    def test_unknown_alone_finalizes_with_uncertainty(self):
        graph, draft = demo_fixture(contradiction=False)
        result = self.run_agent(graph, draft, mock_qwen(draft, ("UNKNOWN",) * 3, "UNKNOWN"),
                                disparity_threshold=0.25)
        self.assertEqual(result["factual_results"]["status"], "UNKNOWN")
        self.assertEqual(result["representation_results"]["status"], "UNKNOWN")
        self.assertEqual(result["status"], "completed_with_uncertainty")
        self.assertEqual(result["route"], "finalize")
        self.assertEqual(result["stop_reason"], "uncertainty_only")
        self.assertTrue(result["uncertainty_flags"])
        self.assertEqual(result["revisions"], [])

    def test_either_checker_can_independently_require_revision(self):
        for factual_failure in (True, False):
            with self.subTest(factual_failure=factual_failure):
                graph, draft = demo_fixture(contradiction=factual_failure)
                verdicts = ("SUPPORTED", "CONTRADICTED", "UNKNOWN") if factual_failure else ("SUPPORTED",) * 3
                result = self.run_agent(graph, draft, mock_qwen(draft, verdicts),
                                        disparity_threshold=0.5 if factual_failure else 0.25)
                self.assertEqual(result["status"], "stopped_with_issues")
                self.assertEqual(result["factual_results"]["status"], "FAIL" if factual_failure else "PASS")
                self.assertEqual(result["representation_results"]["status"], "PASS" if factual_failure else "FAIL")

    def test_missing_threshold_is_uncertainty_not_revision(self):
        graph, draft = demo_fixture(contradiction=False)
        result = self.run_agent(graph, draft, mock_qwen(draft, ("SUPPORTED",) * 3))
        self.assertEqual(result["status"], "completed_with_uncertainty")
        self.assertIn("threshold_not_configured", result["uncertainty_flags"])

    def test_confirmed_disparity_routes_to_revision_when_group_issue_list_is_empty(self):
        graph, draft = demo_fixture(contradiction=False)
        result = self.run_agent(graph, draft, mock_qwen(draft, ("SUPPORTED",) * 3,
                                                       policy_status="UNKNOWN"), disparity_threshold=0.25)
        rep = result["representation_results"]
        self.assertEqual(rep["status"], "FAIL")
        self.assertAlmostEqual(rep["disparity_bounds"]["lower"], 1 / 3)
        self.assertEqual(rep["representation_issues"], [])
        self.assertEqual(result["route"], "finalize")
        self.assertEqual(result["status"], "stopped_with_issues")
        self.assertIn("uncertain_representation_matches", result["uncertainty_flags"])

    def test_canonical_superbowl_topic_does_not_hide_missing_labels(self):
        graph, draft = demo_fixture(contradiction=False)
        for tweet in list(graph.subjects(NS.aboutTopic, NS.Topic_Obama)):
            graph.remove((tweet, NS.aboutTopic, NS.Topic_Obama))
            graph.add((tweet, NS.aboutTopic, NS.SuperBowl))
        graph.add((NS.InvalidExample, NS.aboutTopic, NS.SuperBowl))
        draft["topic"] = "Superbowl"
        result = run_summary_agent(graph, draft, topic="Superbowl", prompt="Synthetic test",
                                   client=mock_qwen(draft, ("SUPPORTED",) * 3), representation_metric="mnli",
                                   expected_group_counts=EXPECTED_COUNTS, disparity_threshold=0.25)
        rep = result["representation_results"]
        self.assertFalse(rep["data_complete"])
        self.assertEqual(rep["model_call_count"], 0)
        self.assertTrue(any("dialect label" in reason for reason in rep["reasons"]))

    def test_incomplete_evidence_uses_original_completeness_defaults(self):
        graph, draft = demo_fixture(contradiction=False)
        result = run_summary_agent(graph, draft, topic="Obama", prompt=draft["summary_prompt"],
                                   client=mock_qwen(draft, ("SUPPORTED",) * 3), disparity_threshold=0.25,
                                   representation_metric="mnli")
        self.assertFalse(result["representation_results"]["data_complete"])
        self.assertEqual(result["representation_results"]["model_call_count"], 0)
        self.assertEqual(result["route"], "finalize")
        self.assertEqual(result["status"], "completed_with_uncertainty")

    def test_failed_extraction_remains_unclear_and_skips_semantic_matching(self):
        graph, draft = demo_fixture()
        for malformed in (GenerationError("Local model unavailable"), response({"bad": "output"})):
            with self.subTest(response=type(malformed).__name__):
                client = Mock()
                client.chat.side_effect = [malformed] * 10  # each sentence: attempt + one retry
                result = self.run_agent(graph, draft, client, disparity_threshold=0.25)
                self.assertEqual(len(result["extraction_errors"]), 5)
                self.assertTrue(all(record["status"] == "unclear" for record in result["summary_claims"]))
                self.assertEqual(result["status"], "completed_with_uncertainty")
                self.assertIn("summary_extraction_failed", result["uncertainty_flags"])
                self.assertEqual(client.chat.call_count, 10)

    def test_bad_inputs_fail_before_model_calls(self):
        graph, draft = demo_fixture()
        client = Mock()
        for change in ({"initial_summary": {}}, {"topic": "missing"}, {"prompt": ""},
                       {"max_tool_calls": -1}, {"max_model_calls": -1}, {"disparity_threshold": float("nan")},
                       {"expected_group_counts": {"AA": 1}}, {"batch_size": 0}):
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                arguments = dict(graph=graph, initial_summary=draft, topic="Obama", prompt="Test", client=client)
                arguments.update(change)
                run_summary_agent(**arguments)
        client.chat.assert_not_called()

    def test_default_client_is_closed_on_success_and_failure(self):
        graph, draft = demo_fixture()
        client = mock_qwen(draft)
        with patch("summary_agent.OllamaClient", return_value=client):
            run_summary_agent(graph, draft, topic="Obama", prompt="Test",
                               expected_group_counts=EXPECTED_COUNTS, disparity_threshold=0.25, max_revisions=0,
                               representation_metric="internal")
        client.close.assert_called_once()
        client.reset_mock()
        with patch("summary_agent.OllamaClient", return_value=client), self.assertRaises(ValueError):
            run_summary_agent(graph, draft, topic="Obama", prompt="Test", batch_size=0)
        client.close.assert_called_once()

    def test_cli_reads_saved_graph_and_draft_and_writes_structured_result(self):
        graph, draft = demo_fixture()
        # The CLI keeps the real 30-per-group default; this small fixture's
        # representation result must remain UNKNOWN, with its factual failure
        # still routing to pending revision.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_file, draft_file, output = root / "input.ttl", root / "draft.json", root / "run.json"
            graph.serialize(destination=graph_file, format="turtle")
            draft_file.write_text(json.dumps(draft), encoding="utf-8")
            before = graph_file.read_bytes()
            client = mock_qwen(draft)
            arguments = ["summary_agent.py", str(draft_file), "--graph", str(graph_file),
                         "--topic", "Obama", "--disparity-threshold", "0.4", "--max-revisions", "0", "--output", str(output),
                         "--representation-metric", "mnli"]
            with patch("sys.argv", arguments), patch("summary_agent.OllamaClient", return_value=client), \
                 redirect_stdout(io.StringIO()) as printed:
                main()
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("Factual:", printed.getvalue())
            self.assertIn("Coverage AA:", printed.getvalue())
            self.assertIn("Revisions:", printed.getvalue())
            self.assertIn("Final status:", printed.getvalue())
            self.assertIn(str(output.resolve()), printed.getvalue())
            self.assertNotIn('"execution_history":', printed.getvalue())
            self.assertEqual(result["representation_results"]["disparity_threshold"], 0.4)
            self.assertEqual(result["status"], "stopped_with_issues")
            self.assertEqual(result["representation_results"]["status"], "UNKNOWN")
            self.assertEqual(graph_file.read_bytes(), before)
            client.close.assert_called_once()

    def test_cli_pilot_defaults_save_unique_complete_results(self):
        graph, draft = demo_fixture(contradiction=False)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            graph_file, draft_file = folder / "Obama.ttl", folder / "Obama_prompt1.json"
            graph.serialize(destination=graph_file, format="turtle")
            draft_file.write_text(json.dumps(draft), encoding="utf-8")
            args = ["summary_agent.py", str(draft_file), "--graph", str(graph_file), "--topic", "Obama"]
            for _ in range(2):
                client = mock_qwen(draft, ("SUPPORTED",) * 3)
                with patch("sys.argv", args), patch("summary_agent.DEFAULT_RUN_DIR", folder / "output"), \
                     patch("summary_agent.OllamaClient", return_value=client), redirect_stdout(io.StringIO()) as printed:
                    main()
                # Default: no group check, dialect labels are evaluation-only.
                self.assertIn("no group check in the agent", printed.getvalue())
                self.assertNotIn("Coverage AA", printed.getvalue())
            files = list((folder / "output").glob("Obama_*.json"))
            self.assertEqual(len(files), 2)
            result = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertIn("iterations", result)
            self.assertIn("execution_history", result)
            self.assertIn("initial_summary", result)
            self.assertEqual(result["representation_results"]["status"], "NOT_RUN")
            self.assertNotIn("representation_check", [event["node"] for event in result["execution_history"]])


if __name__ == "__main__":
    unittest.main()
