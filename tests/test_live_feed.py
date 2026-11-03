"""Live observation is immediate, bounded, ordered and cannot alter model verdicts."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from demo_revision_loop import fixture, mock_client
from live_feed import LiveFeed, bounded, emit_event, observe_node, request_context
from run_pipeline import ProgressClient
from summary_agent import run_summary_agent


class LiveFeedTests(unittest.TestCase):
    def events(self, path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_jsonl_is_flushed_before_close_and_console_shows_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            path, console = Path(directory) / "feed.jsonl", io.StringIO()
            feed = LiveFeed(path, "run-1", console=console, heartbeat_seconds=0)
            try:
                feed.emit("handoff", "extractor", "checker", "Two claims ready", claim_ids=["C1", "C2"])
                rows = self.events(path)
                self.assertEqual(rows[0]["sequence"], 1)
                self.assertEqual(rows[0]["run_id"], "run-1")
                self.assertEqual(rows[0]["details"]["claim_ids"], ["C1", "C2"])
                self.assertIn("extractor -> checker", console.getvalue())
            finally:
                feed.close()

    def test_retry_appends_with_distinct_run_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feed.jsonl"
            for run_id in ("first", "second"):
                feed = LiveFeed(path, run_id, console=io.StringIO(), heartbeat_seconds=0)
                feed.emit("run_started", "user", "orchestrator", run_id)
                feed.close()
            self.assertEqual([r["run_id"] for r in self.events(path)], ["first", "second"])

    def test_concurrent_heartbeat_and_events_have_ordered_unique_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feed.jsonl"
            feed = LiveFeed(path, "run", console=io.StringIO(), heartbeat_seconds=0)
            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(lambda n: feed.emit("event", "a", "b", str(n)), range(60)))
                self.assertEqual([r["sequence"] for r in self.events(path)], list(range(1, 61)))
            finally:
                feed.close()

    def test_heartbeat_is_only_emitted_while_a_real_call_is_active(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feed.jsonl"
            feed = LiveFeed(path, "run", console=io.StringIO(), heartbeat_seconds=0)
            try:
                feed.heartbeat()
                feed.begin_call("checker", "Qwen", "Claim C1", call_id=1)
                feed.heartbeat()
                feed.end_call()
                feed.heartbeat()
                self.assertEqual([r["kind"] for r in self.events(path)], ["call_started", "waiting"])
            finally:
                feed.close()

    def test_preview_limits_are_explicit_and_json_safe(self):
        result = bounded({"text": "x" * 2000, "items": list(range(100)), "score": float("nan")})
        self.assertIn("truncated", result["text"])
        self.assertEqual(result["items"][-1], {"_omitted_items": 88})
        json.dumps(result, allow_nan=False)

    def test_mock_clients_do_not_invent_observers(self):
        client = Mock()
        emit_event(client, "event", "a", "b", "ignored")
        self.assertNotIn("live_feed", client.__dict__)
        self.assertEqual(client.mock_calls, [])
        self.assertEqual(observe_node("unknown", lambda state: {"value": 2}, client)({}), {"value": 2})

    def test_observer_io_failure_does_not_change_execution(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()) as error:
            feed = LiveFeed(Path(directory) / "feed.jsonl", "run", console=io.StringIO(), heartbeat_seconds=0)
            feed.file.close()
            feed.emit("event", "a", "b", "test")
            self.assertTrue(feed.disabled)
            self.assertIn("agent execution continues", error.getvalue())
            feed.close()

    def test_model_transport_is_unchanged_and_reply_is_pending_validation(self):
        backend = Mock()
        result = {"output": [{"content": [{"text": json.dumps({"action": "finalize", "verdict": "SUPPORTED"})}]}]}
        backend.chat.return_value = result
        payload = {"messages": [{"role": "user", "content": json.dumps({"summary_claim": {"claim_id": "C1"}})}],
                   "format": {"properties": {"action": {}}}}
        original = deepcopy(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feed.jsonl"
            feed = LiveFeed(path, "run", console=io.StringIO(), heartbeat_seconds=0)
            try:
                client = ProgressClient(backend, feed)
                self.assertIs(client.chat(payload), result)
                self.assertEqual(payload, original)
                backend.chat.assert_called_once_with(payload)
                row = self.events(path)[-1]
                self.assertEqual(row["kind"], "model_returned")
                self.assertEqual(row["details"]["validation_status"], "pending")
                self.assertIsNone(feed.active)
            finally:
                feed.close()

    def test_model_error_is_rethrown_and_active_call_is_cleared(self):
        backend = Mock()
        backend.chat.side_effect = RuntimeError("test transport failure")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feed.jsonl"
            feed = LiveFeed(path, "run", console=io.StringIO(), heartbeat_seconds=0)
            try:
                with self.assertRaisesRegex(RuntimeError, "transport failure"):
                    ProgressClient(backend, feed).chat({})
                self.assertEqual(self.events(path)[-1]["kind"], "call_failed")
                self.assertIsNone(feed.active)
            finally:
                feed.close()

    def test_equivalence_and_coverage_have_distinct_labels(self):
        local = {"format": {"properties": {"clusters": {"items": {"properties": {"candidate_ids": {}}}}}}}
        self.assertEqual(request_context(local)[0], "equivalence_checker")
        local["format"]["properties"]["clusters"]["items"]["properties"]["status"] = {}
        self.assertEqual(request_context(local)[0], "coverage_checker")

    def test_real_graph_revision_and_rechecking_emit_handoffs_without_changing_results(self):
        graph, draft, counts = fixture()
        options = dict(topic="Obama", prompt=draft["summary_prompt"], expected_group_counts=counts, disparity_threshold=0.25,
                       representation_metric="internal")
        baseline_client = mock_client()
        baseline = run_summary_agent(graph, draft, client=baseline_client, **options)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feed.jsonl"
            feed = LiveFeed(path, "run", console=io.StringIO(), heartbeat_seconds=0)
            backend = mock_client()
            try:
                observed = run_summary_agent(graph, draft, client=ProgressClient(backend, feed), **options)
                self.assertEqual(observed, baseline)
                self.assertEqual(backend.chat.call_count, baseline_client.chat.call_count)
                rows = self.events(path)
                pairs = {(r["actor"], r["recipient"]) for r in rows if r["kind"] == "handoff"}
                self.assertIn(("route_checks", "retrieve_revision_evidence"), pairs)
                self.assertIn(("revise_summary", "extract_summary_claims"), pairs)
                self.assertTrue(any(r["kind"] == "tool_result" for r in rows))
                self.assertTrue(any(r["kind"] == "decision_validated" for r in rows))
                self.assertTrue(any(r["kind"] == "assessment_validated" for r in rows))
                routed = [r for r in rows if r["actor"] == "route_checks" and r["kind"] == "handoff"]
                self.assertEqual(routed[-1]["details"]["latest_revision"]["status"], "accepted")
            finally:
                feed.close()

    def test_rejected_revision_is_visible_and_remains_rejected(self):
        graph, draft, counts = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feed.jsonl"
            feed = LiveFeed(path, "run", console=io.StringIO(), heartbeat_seconds=0)
            try:
                backend = mock_client(transform_revision=lambda value: {"invalid": True})
                result = run_summary_agent(graph, draft, topic="Obama", prompt=draft["summary_prompt"],
                    expected_group_counts=counts, disparity_threshold=0.25, client=ProgressClient(backend, feed),
                    representation_metric="internal")
                self.assertEqual(result["revisions"][-1]["status"], "rejected")
                row = next(r for r in self.events(path) if r["actor"] == "revise_summary" and r["kind"] == "handoff")
                self.assertEqual(row["details"]["latest_revision"]["status"], "rejected")
            finally:
                feed.close()


if __name__ == "__main__":
    unittest.main()
