"""Offline tests. Synthetic API responses are fixtures, never model evaluations."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx2
from openai import OpenAI

from data_loader import load_tweets
from prompts import get_prompt
from summarizer import DEFAULT_MODEL, GenerationError, build_request, generate_summary, make_client, parse_response, save_result, validate_output


def fixture_output():
    return {"sentences": [
        {"text": f"Synthetic test sentence {i}.", "source_ids": [f"Obama_{i:03d}"]}
        for i in range(1, 6)
    ]}


def fixture_response():
    return {
        "id": "resp_offline_fixture", "object": "response", "created_at": 0,
        "status": "completed", "model": DEFAULT_MODEL,
        "output": [{"id": "msg_fixture", "type": "message", "role": "assistant",
                    "status": "completed", "content": [{"type": "output_text",
                    "text": json.dumps(fixture_output()), "annotations": []}]}],
        "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    }


class SummarizerTests(unittest.TestCase):
    def setUp(self):
        self.tweets = load_tweets("Obama")

    def test_request_contains_exact_tweets_and_both_report_prompts(self):
        for prompt_id in ["1", "2"]:
            prompt = get_prompt("Obama", prompt_id)
            request = build_request(self.tweets, prompt, provider="openai")
            content = request["input"][0]["content"]
            self.assertTrue(content.startswith(prompt))
            records = json.loads(content.split("Source tweets (JSON data):\n", 1)[1])
            self.assertEqual(records, [{"id": t.id, "text": t.text} for t in self.tweets])
            self.assertEqual(len(records), 90)
            self.assertFalse(request["store"])
            self.assertEqual(request["text"]["format"]["schema"]["properties"]["sentences"]["maxItems"], 5)
            self.assertNotIn("reference_summaries", request)

    def test_real_sdk_with_mock_http_and_saved_metadata(self):
        def respond(request):
            self.assertEqual(request.url.path, "/v1/responses")
            sent = json.loads(request.content)
            self.assertEqual(sent["model"], DEFAULT_MODEL)
            self.assertTrue(sent["text"]["format"]["strict"])
            return httpx2.Response(200, json=fixture_response())
        with OpenAI(api_key="offline-fixture-not-a-real-key", max_retries=0,
                    http_client=httpx2.Client(transport=httpx2.MockTransport(respond))) as client:
            result = generate_summary(self.tweets, get_prompt("Obama", "1"), client=client, provider="openai")
        self.assertEqual(result["status"], "draft_unchecked")
        self.assertEqual(len(result["sentences"]), 5)
        self.assertEqual(result["usage"]["total_tokens"], 150)
        self.assertFalse(result["ontology_used"])
        self.assertFalse(result["reference_summaries_used"])
        self.assertNotIn("offline-fixture-not-a-real-key", json.dumps(result))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "draft.json"
            save_result(result, path)
            self.assertEqual(path.with_suffix(".txt").read_text().strip(), result["summary"])
            self.assertEqual(json.loads(path.read_text())["request_sha256"], result["request_sha256"])

    def test_rejects_refusals_and_truncation(self):
        response = fixture_response()
        response["status"] = "incomplete"
        response["incomplete_details"] = {"reason": "max_output_tokens"}
        with self.assertRaises(GenerationError):
            parse_response(response, self.tweets)
        response = fixture_response()
        response["output"][0]["content"] = [{"type": "refusal", "refusal": "Synthetic refusal"}]
        with self.assertRaises(GenerationError):
            parse_response(response, self.tweets)

    def test_rejects_wrong_count_and_unknown_ids(self):
        wrong_count = fixture_output()
        wrong_count["sentences"].pop()
        unknown_id = fixture_output()
        unknown_id["sentences"][0]["source_ids"] = ["McDonalds_001"]
        for data in [wrong_count, unknown_id, {"sentences": "not a list"}]:
            with self.assertRaises(GenerationError):
                validate_output(json.dumps(data), self.tweets)

    def test_missing_key_has_clear_error(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(GenerationError, "OPENAI_API_KEY"):
                make_client("openai")

    def test_api_auth_error_is_reported_without_secret(self):
        transport = httpx2.MockTransport(lambda request: httpx2.Response(
            401, json={"error": {"message": "invalid key", "type": "authentication_error"}}))
        with OpenAI(api_key="offline-fixture-not-a-real-key", max_retries=0,
                    http_client=httpx2.Client(transport=transport)) as client:
            with self.assertRaisesRegex(GenerationError, "HTTP 401"):
                generate_summary(self.tweets, get_prompt("Obama", "1"), client=client, provider="openai")

    def test_cli_dry_run_all_six_combinations_without_key(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            completed = subprocess.run(
                [sys.executable, str(project / "main.py"), "--section", "draft", "--topic", "all",
                 "--prompt-id", "all", "--dry-run", "--output-dir", folder],
                env={**os.environ, "OPENAI_API_KEY": ""}, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(len(list(Path(folder).rglob("*.request.json"))), 6)
            self.assertEqual(len(list(Path(folder).rglob("*.txt"))), 0)
            self.assertIn("No API calls made", completed.stdout)


if __name__ == "__main__":
    unittest.main()
