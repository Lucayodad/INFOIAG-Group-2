"""Local Ollama transport, using only the Python standard library."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OllamaClient:
    """Use loopback only, bypass proxies, and never fall back to a paid service."""

    endpoint = "http://127.0.0.1:11434/api/chat"

    def __init__(self, timeout=300, opener=None):
        self.timeout = timeout
        self.opener = opener or build_opener(ProxyHandler({}), NoRedirects())

    def chat(self, payload):
        from summarizer import GenerationError
        request = Request(self.endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                data = json.load(response)
        except HTTPError as exc:
            exc.close()
            hint = (f"Download the model first: ollama pull {payload['model']}" if exc.code == 404 else
                    "Check that Ollama and the selected model support these request settings.")
            raise GenerationError(f"Ollama returned HTTP {exc.code}. {hint}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise GenerationError("Could not complete the local Ollama request. Open Ollama or run "
                                  "'ollama serve'; if it is running, check for a timeout or insufficient memory.") from exc
        except (ValueError, UnicodeError) as exc:
            raise GenerationError("Ollama returned malformed JSON. No summary saved.") from exc
        if not isinstance(data, dict) or data.get("error"):
            raise GenerationError("Ollama reported an error. No summary saved.")
        if data.get("done") is not True or data.get("done_reason") != "stop":
            raise GenerationError("Ollama response was incomplete or truncated. No summary saved.")
        message = data.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise GenerationError("Ollama returned no assistant message.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise GenerationError("Ollama returned no summary text.")
        counts = [data.get("prompt_eval_count"), data.get("eval_count")]
        usage = {"input_tokens": counts[0], "output_tokens": counts[1],
                 "total_tokens": sum(counts) if all(type(x) is int for x in counts) else None}
        return {
            "status": "completed", "model": data.get("model"), "usage": usage,
            "output": [{"type": "message", "content": [{"type": "output_text", "text": content}]}],
            "backend_metadata": {key: data.get(key) for key in
                                 ("created_at", "total_duration", "load_duration", "prompt_eval_duration",
                                  "eval_duration", "done_reason")},
        }

    def close(self):
        pass  # urllib responses are closed immediately by their context manager.
