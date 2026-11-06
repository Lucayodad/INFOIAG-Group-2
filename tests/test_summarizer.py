"""Offline tests. Synthetic responses are fixtures, never model evaluations."""

import json
import unittest

from data_loader import load_tweets
from prompts import get_prompt
from summarizer import DEFAULT_MODEL, GenerationError, build_request, parse_response, validate_output


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
            request = build_request(self.tweets, prompt)
            content = request["messages"][1]["content"]
            self.assertTrue(content.startswith(prompt))
            records = json.loads(content.split("Source tweets (JSON data):\n", 1)[1])
            self.assertEqual(records, [{"id": t.id, "text": t.text} for t in self.tweets])
            self.assertEqual(len(records), 90)
            self.assertFalse(request["stream"])
            self.assertEqual(request["format"]["properties"]["sentences"]["maxItems"], 5)
            self.assertNotIn("reference_summaries", request)

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


if __name__ == "__main__":
    unittest.main()
