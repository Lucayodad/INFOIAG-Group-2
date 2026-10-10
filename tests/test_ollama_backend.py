"""Synthetic local HTTP fixtures; these are not model quality evaluations."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import unittest
from urllib.error import URLError
from unittest.mock import Mock

from data_loader import load_tweets
from ollama_backend import OllamaClient
from prompts import get_prompt
from summarizer import GenerationError, build_request, generate_summary, make_client


def fixture():
    return {"model": "qwen3.5:4b", "done": True, "done_reason": "stop",
            "prompt_eval_count": 100, "eval_count": 50,
            "message": {"role": "assistant", "content": json.dumps({"sentences": [
                {"text": f"Synthetic fixture sentence {i}.", "source_ids": [f"Obama_{i:03d}"]}
                for i in range(1, 6)]})}}


@contextmanager
def server(response, status=200):
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        def log_message(self, *args):
            pass
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    client = OllamaClient()
    client.endpoint = f"http://127.0.0.1:{httpd.server_port}/api/chat"
    try:
        yield client, received
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


class OllamaTests(unittest.TestCase):
    def setUp(self):
        self.tweets = load_tweets("Obama")
        self.prompt = get_prompt("Obama", "1")

    def test_local_request_has_all_tweets_and_schema(self):
        request = build_request(self.tweets, self.prompt)
        records = json.loads(request['messages'][1]['content'].split('Source tweets (JSON data):\n')[1])
        self.assertEqual(records, [{"id": t.id, "text": t.text} for t in self.tweets])
        self.assertFalse(request['stream'])
        self.assertFalse(request['think'])
        self.assertEqual(request['format']['properties']['sentences']['maxItems'], 5)
        self.assertEqual(request['options']['num_ctx'], 16384)

    def test_actual_local_http_transport_and_metadata(self):
        with server(fixture()) as (client, received):
            result = generate_summary(self.tweets, self.prompt, client=client)
        self.assertEqual(received[0][0], '/api/chat')
        self.assertEqual(received[0][1], build_request(self.tweets, self.prompt))
        self.assertEqual(result['provider'], 'ollama')
        self.assertEqual(result['usage']['total_tokens'], 150)
        self.assertEqual(result['status'], 'draft_unchecked')
        self.assertFalse(result['reference_summaries_used'])

    def test_rejects_truncated_response_even_if_json_is_valid(self):
        data = fixture()
        data['done_reason'] = 'length'
        with server(data) as (client, _):
            with self.assertRaisesRegex(GenerationError, 'truncated'):
                generate_summary(self.tweets, self.prompt, client=client)

    def test_rejects_invalid_source_ids(self):
        data = fixture()
        output = json.loads(data['message']['content'])
        output['sentences'][0]['source_ids'] = ['unknown']
        data['message']['content'] = json.dumps(output)
        with server(data) as (client, _):
            with self.assertRaisesRegex(GenerationError, 'unknown source'):
                generate_summary(self.tweets, self.prompt, client=client)

    def test_missing_model_and_unavailable_server_errors(self):
        with server({'error': 'model missing'}, 404) as (client, _):
            with self.assertRaisesRegex(GenerationError, 'ollama pull'):
                generate_summary(self.tweets, self.prompt, client=client)
        opener = Mock()
        opener.open.side_effect = URLError('fixture: offline')
        with self.assertRaisesRegex(GenerationError, 'ollama serve'):
            generate_summary(self.tweets, self.prompt, client=OllamaClient(opener=opener))

    def test_default_client_requires_no_api_key(self):
        self.assertIsInstance(make_client(), OllamaClient)


if __name__ == '__main__':
    unittest.main()
