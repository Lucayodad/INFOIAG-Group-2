"""Fresh-input pipeline integration; model outputs are deterministic fixtures."""

from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from fakes import many_tweets, mock_client, response
from run_pipeline import main, run_pipeline, validate_reused_draft


class ReleasePipelineTests(unittest.TestCase):
    def fixture(self):
        """90 synthetic tweets and a draft whose second sentence contradicts the tweet it cites."""
        tweets, records = many_tweets(per_group=30)
        source = {record["tweet_id"]: record["claims"][0] for record in records}
        texts = ["End of briefing 1.", "One author dislikes Obama's speech.", "End of briefing 3.",
                 "End of briefing 4.", "End of briefing 5."]
        draft = {"sentences": [{"text": text, "source_ids": ["AA_00"]} for text in texts]}
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
        return Namespace(topic="Obama", prompt_id="1", max_revisions=1, output=root / f"{name}.json",
                         source_claims=root / "checkpoints/Obama.json", reuse_draft=None, max_tool_calls=3)

    def test_raw_tweets_to_revision_rechecking_and_saved_results_then_resume(self):
        tweets, client, sources = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("data_loader.load_tweets", return_value=tweets), redirect_stdout(io.StringIO()):
                result = run_pipeline(self.args(root), client)
            self.assertEqual(len(sources), 90)
            self.assertEqual(result["revision_count"], 1)
            self.assertEqual([r["status"] for r in result["revisions"]], ["accepted"])
            self.assertEqual(result["factual_results"]["verdict_counts"]["CONTRADICTED"], 0)
            nodes = [e["node"] for e in result["execution_history"]]
            self.assertIn("revise_summary", nodes)
            self.assertEqual(nodes.count("factual_check"), 2)
            self.assertEqual(result["representation_results"]["status"], "NOT_RUN")
            self.assertEqual(result["pipeline"]["group_counts"], {g: 30 for g in ("AA", "Hisp", "White")})
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
            self.assertIn("--max-revisions", console.getvalue())
            self.assertNotIn("--disparity-threshold", console.getvalue())  # accepted only for old callers

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
