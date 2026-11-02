"""Fresh-input pipeline integration; model outputs are deterministic fixtures."""

from argparse import Namespace
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from data_loader import Tweet
from demo_revision_loop import fixture, mock_client
from demo_summary_agent import response
from representation_check import match_summary_claims
from representation_tools import get_group_claims
from run_pipeline import main, run_pipeline, validate_reused_draft
from test_representation_batches import oracle_client, topic_fixture


class ReleasePipelineTests(unittest.TestCase):
    def fixture(self):
        graph, draft, _ = fixture(many=True, per_group=30)
        groups = {g: get_group_claims(graph, "Obama", g) for g in ("AA", "Hisp", "White")}
        tweets = [Tweet(t["tweet_id"], "Obama", g, t["text"], "synthetic_release.txt", t["source_line"])
                  for g, group in groups.items() for t in group["tweets"]]
        source = {c["tweet_id"]: c for group in groups.values() for c in group["claims"]}
        client = mock_client()
        agent_chat = client.chat.side_effect
        client.check_models.return_value = [{"name": "qwen3.5:4b"}, {"name": "all-minilm:latest"}]
        calls = []
        def chat(payload):
            if "sentences" in payload["format"]["properties"]:
                return {"status": "completed", "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": json.dumps({"sentences": draft["sentences"]})}]}]}
            raw = payload["messages"][-1]["content"]
            try:
                body = json.loads(raw)
            except ValueError:
                body = {}
            if isinstance(body, dict) and "id" in body:
                calls.append(body["id"])
                claim = source[body["id"]]
                return response({"status": "extracted", "claims": [{"quote": claim["source_quote"],
                    "claim": claim["claim"], "target": claim["target"], "stance": claim["stance"]}]})
            return agent_chat(payload)
        client.chat.side_effect = chat
        return tweets, client, calls

    def args(self, root, name="result"):
        return Namespace(topic="Obama", prompt_id="1", disparity_threshold=0.25,
            max_revisions=1, output=root / f"{name}.json", source_claims=root / "checkpoints/Obama.json",
            reuse_draft=None, max_model_calls=64, max_tool_calls=3, representation_metric="internal")

    def test_raw_tweets_to_revision_rechecking_and_saved_results_then_resume(self):
        tweets, client, sources = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("data_loader.load_tweets", return_value=tweets), redirect_stdout(io.StringIO()):
                result = run_pipeline(self.args(root), client)
            self.assertEqual(len(sources), 90)
            self.assertEqual(result["revision_count"], 1)
            nodes = [e["node"] for e in result["execution_history"]]
            self.assertIn("revise_summary", nodes)
            self.assertEqual(nodes.count("factual_check"), 2)
            self.assertEqual(nodes.count("representation_check"), 2)
            self.assertEqual(result["representation_results"]["source_tweet_counts"], {g: 30 for g in ("AA", "Hisp", "White")})
            self.assertTrue((root / "result_artifacts/Obama.ttl").exists())
            self.assertTrue((root / "result_artifacts/Obama_prompt1.json").exists())
            saved = json.loads((root / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["stop_reason"], result["stop_reason"])
            feed_path = Path(saved["pipeline"]["live_feed"])
            events = [json.loads(line) for line in feed_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[0]["kind"], "run_started")
            self.assertEqual(events[-1]["kind"], "run_completed")
            self.assertTrue(any(e["kind"] == "tool_result" for e in events))
            checkpoint = root / "checkpoints/Obama.json"
            before = checkpoint.read_bytes()
            sources.clear()
            with patch("data_loader.load_tweets", return_value=tweets), redirect_stdout(io.StringIO()):
                run_pipeline(self.args(root, "resumed"), client)
            self.assertEqual(sources, [])
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_model_preflight_failure_is_clear_and_recorded(self):
        from summarizer import GenerationError
        client = Mock()
        client.check_models.side_effect = GenerationError("Missing Ollama models. Run: ollama pull all-minilm")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            with self.assertRaisesRegex(GenerationError, "ollama pull"):
                run_pipeline(self.args(root), client)
            error = json.loads((root / "result_artifacts/error.json").read_text(encoding="utf-8"))
            self.assertEqual(error["stage"], "preflight")
            client.chat.assert_not_called()

    def test_cli_help_does_not_load_dependencies_or_models(self):
        with patch("run_pipeline.run_pipeline") as run, redirect_stdout(io.StringIO()) as console:
            with self.assertRaises(SystemExit) as error:
                main(["--help"])
            self.assertEqual(error.exception.code, 0)
            run.assert_not_called()
            self.assertIn("--disparity-threshold", console.getvalue())

    def test_missing_python_dependency_has_install_command(self):
        with patch("run_pipeline.run_pipeline", side_effect=ModuleNotFoundError("No module", name="langgraph")), \
             patch("sys.stderr", new_callable=io.StringIO) as error:
            with self.assertRaises(SystemExit) as status:
                main(["--topic", "Obama"])
            self.assertEqual(status.exception.code, 1)
            self.assertIn("pip install -r requirements.txt", error.getvalue())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            args.output.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_pipeline(args, Mock())
            self.assertEqual(args.output.read_text(), "preserve")


class ReusedDraftTests(unittest.TestCase):
    def setUp(self):
        from data_loader import load_tweets
        from prompts import get_prompt
        self.tweets = load_tweets("Obama")
        self.draft = {"topic": "Obama", "summary_prompt": get_prompt("Obama", "1"),
            "sentences": [{"text": f"Sentence {i + 1}.", "source_ids": [self.tweets[i].id]} for i in range(5)]}

    def test_legacy_draft_with_actual_matching_prompt_is_normalized(self):
        result = validate_reused_draft(self.draft, "Obama", "1", self.tweets)
        self.assertEqual(result["prompt_id"], "1")
        self.assertEqual(result["summary"], " ".join(row["text"] for row in self.draft["sentences"]))

    def test_wrong_actual_prompt_without_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "topic/prompt"):
            validate_reused_draft(self.draft, "Obama", "2", self.tweets)

    def test_conflicting_prompt_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "topic/prompt"):
            validate_reused_draft({**self.draft, "prompt_id": "2"}, "Obama", "2", self.tweets)

    def test_missing_prompt_provenance_is_rejected(self):
        self.draft.pop("summary_prompt")
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_reused_draft(self.draft, "Obama", "1", self.tweets)

    def test_matching_id_without_prompt_text_is_supported(self):
        self.draft.pop("summary_prompt")
        result = validate_reused_draft({**self.draft, "prompt_id": "1"}, "Obama", "1", self.tweets)
        self.assertEqual(result["prompt_id"], "1")

    def test_reused_draft_still_validates_source_citations(self):
        self.draft["sentences"][0]["source_ids"] = ["invented"]
        with self.assertRaisesRegex(ValueError, "unknown source"):
            validate_reused_draft(self.draft, "Obama", "1", self.tweets)


class ReleaseRepresentationTests(unittest.TestCase):
    def test_byte_split_small_collection_still_reconciles_global_equivalence(self):
        graph, summary, gold = topic_fixture()
        groups = {g: get_group_claims(graph, "Obama", g) for g in ("AA", "Hisp", "White")}
        client = oracle_client(gold)
        result = match_summary_claims(groups, summary, client, batch_size=16, max_input_bytes=5000)
        self.assertEqual(len(result["clusters"]), 3)
        self.assertEqual(result["aggregation_status"], "ready")
        self.assertEqual(result["equivalence_scope"], "embedding_shortlist_only")
        self.assertTrue(all(e["input_bytes"] <= 5000 for e in result["execution_trace"]))

    def test_166_distinct_claims_reach_all_coverage_batches_below_budget(self):
        graph, summary, gold = topic_fixture(unique=True)
        groups = {g: get_group_claims(graph, "Obama", g) for g in ("AA", "Hisp", "White")}
        original = [c for group in groups.values() for c in group["claims"]]
        for source in original[:76]:
            claim = deepcopy(source)
            claim["claim_id"] = source["claim_id"] + "_extra"
            claim["claim_uri"] += "_extra"
            claim["claim"] = "The author requests the distinct item " + claim["claim_id"] + " from Obama."
            gold[claim["claim_id"]] = claim["claim_id"]
            groups[claim["dialect_label"]]["claims"].append(claim)
        client = oracle_client(gold)
        report = match_summary_claims(groups, summary, client)
        self.assertEqual(len(report["clusters"]), 166)
        self.assertEqual(report["aggregation_status"], "ready")
        self.assertEqual(report["model_call_count"], 21)
        self.assertTrue(all(c["status"] == "NOT_MATCHED" for c in report["clusters"]))
        self.assertEqual(sum(e["candidate_count"] for e in report["execution_trace"] if e["step"] == "summary_coverage"), 166)
        self.assertFalse(report["global_equivalence_exhaustive"])

    def test_equivalence_failures_cannot_consume_reserved_coverage_calls(self):
        graph, summary, gold = topic_fixture()
        groups = {g: get_group_claims(graph, "Obama", g) for g in ("AA", "Hisp", "White")}
        client = oracle_client(gold)
        oracle = client.chat.side_effect
        def fail_equivalence(payload):
            return response({"decisions": []}) if "decisions" in payload["format"]["properties"] else oracle(payload)
        client.chat.side_effect = fail_equivalence
        result = match_summary_claims(groups, summary, client, batch_size=1, max_model_calls=8)
        equivalence = [e for e in result["execution_trace"] if e["step"] == "shortlisted_equivalence"]
        self.assertLessEqual(len(equivalence), 4)
        self.assertLessEqual(result["model_call_count"], 8)
        self.assertEqual(result["aggregation_status"], "UNKNOWN")

    def test_failed_embeddings_preserve_unknown_without_chat_or_lexical_fallback(self):
        graph, summary, gold = topic_fixture()
        groups = {g: get_group_claims(graph, "Obama", g) for g in ("AA", "Hisp", "White")}
        client = oracle_client(gold)
        result = match_summary_claims(groups, summary, client, batch_size=2,
                                      embedder=lambda texts: [[float("nan")]] * len(texts))
        self.assertEqual(result["reason"], "embedding_retrieval_failed")
        self.assertEqual(result["embedding_call_count"], 1)
        self.assertTrue(all(c["status"] == "UNKNOWN" for c in result["clusters"]))
        client.chat.assert_not_called()


class LocalModelPreflightTests(unittest.TestCase):
    def test_missing_model_names_include_download_commands(self):
        from ollama_backend import OllamaClient
        from summarizer import GenerationError
        opener = Mock()
        opener.open.return_value = io.BytesIO(json.dumps({"models": [{"name": "qwen3.5:4b"}]}).encode())
        with self.assertRaisesRegex(GenerationError, "ollama pull all-minilm"):
            OllamaClient(opener=opener).check_models()

    def test_unavailable_server_has_start_instruction(self):
        from ollama_backend import OllamaClient
        from summarizer import GenerationError
        opener = Mock()
        opener.open.side_effect = URLError("offline")
        with self.assertRaisesRegex(GenerationError, "ollama serve"):
            OllamaClient(opener=opener).check_models()


if __name__ == "__main__":
    unittest.main()
