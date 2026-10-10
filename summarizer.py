"""Section 3: first-draft generation, without ontology revision or evaluation."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from data_loader import Tweet, validate_dataset
from prompts import SYSTEM_PROMPT


DEFAULT_MODEL = "qwen3.5:4b"
OPENAI_MODEL = "gpt-4.1-mini-2025-04-14"


class GenerationError(ValueError):
    """A generation request or its output could not be accepted."""


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def build_request(tweets: list[Tweet], summary_prompt: str,
                  model: str = DEFAULT_MODEL, provider: str = "ollama") -> dict:
    """All 90 tweet texts are supplied; no vector store or references are read."""
    validate_dataset(tweets)
    if provider not in {"ollama", "openai"}:
        raise GenerationError("Provider must be ollama or openai.")
    if not isinstance(summary_prompt, str) or not summary_prompt.strip():
        raise GenerationError("A nonempty summary prompt is required.")
    if not isinstance(model, str) or not model.strip():
        raise GenerationError("A nonempty model name is required.")
    tweet_ids = [tweet.id for tweet in tweets]
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "sentences": {
                "type": "array", "minItems": 5, "maxItems": 5,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "source_ids": {"type": "array", "minItems": 1,
                                       "items": {"type": "string", "enum": tweet_ids}},
                    },
                    "required": ["text", "source_ids"],
                },
            },
        },
        "required": ["sentences"],
    }
    # Dialect labels remain in the local dataset for later coverage checks.
    # The generator receives only original text and stable source IDs.
    source = [{"id": tweet.id, "text": tweet.text} for tweet in tweets]
    if provider == "ollama":
        return {
            "model": model, "stream": False, "think": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": summary_prompt +
                 "\n\nRequired JSON schema:\n" + json.dumps(schema) +
                 "\n\nSource tweets (JSON data):\n" + json.dumps(source, ensure_ascii=False)},
            ],
            "format": schema,
            "options": {"temperature": 0, "seed": 42, "num_ctx": 16384, "num_predict": 2000},
        }
    return {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": [{"role": "user", "content": summary_prompt +
                   "\n\nSource tweets (JSON data):\n" + json.dumps(source, ensure_ascii=False)}],
        "max_output_tokens": 2000,
        "store": False,
        "text": {"format": {"type": "json_schema", "name": "tweet_summary",
                              "strict": True, "schema": schema}},
    }


def validate_output(raw_text: str, tweets: list[Tweet]) -> dict:
    """Validate structure and referenced IDs, not semantic grounding or fairness."""
    try:
        data = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise GenerationError("The model did not return valid JSON.") from exc
    if not isinstance(data, dict) or set(data) != {"sentences"}:
        raise GenerationError("Expected exactly one top-level field: sentences.")
    sentences = data["sentences"]
    if not isinstance(sentences, list) or len(sentences) != 5:
        raise GenerationError("Expected exactly five sentence entries.")
    allowed_ids = {tweet.id for tweet in tweets}
    texts = set()
    for index, entry in enumerate(sentences, 1):
        if not isinstance(entry, dict) or set(entry) != {"text", "source_ids"}:
            raise GenerationError(f"Sentence {index}: expected text and source_ids.")
        text = entry["text"]
        if not isinstance(text, str) or not text.strip() or "\n" in text or "\r" in text:
            raise GenerationError(f"Sentence {index}: expected nonempty single-line text.")
        if text.strip() in texts:
            raise GenerationError(f"Sentence {index}: repeated sentence text.")
        texts.add(text.strip())
        ids = entry["source_ids"]
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
            raise GenerationError(f"Sentence {index}: source_ids must be a nonempty string list.")
        if len(set(ids)) != len(ids) or not set(ids).issubset(allowed_ids):
            raise GenerationError(f"Sentence {index}: duplicate or unknown source IDs.")
    return data


def parse_response(response: dict, tweets: list[Tweet]) -> dict:
    """Reject refusals, truncation, and malformed data rather than saving a draft."""
    if response.get("status") != "completed":
        reason = (response.get("incomplete_details") or {}).get("reason", "unknown")
        raise GenerationError(f"Response did not complete (reason: {reason}). No summary saved.")
    texts = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "refusal":
                raise GenerationError("The model refused this request. No summary saved.")
            if content.get("type") == "output_text":
                texts.append(content["text"])
    if not texts:
        raise GenerationError("The response contains no output text.")
    return validate_output("".join(texts), tweets)


def make_client(provider: str = "ollama"):
    if provider == "ollama":
        from ollama_backend import OllamaClient
        return OllamaClient()
    if provider != "openai":
        raise GenerationError("Provider must be ollama or openai.")
    return make_openai_client()


def make_openai_client():
    """Read the optional OpenAI key locally; never save it in requests."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise GenerationError("Set OPENAI_API_KEY in your terminal, or use --dry-run without a key.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise GenerationError("Install dependencies: python3 -m pip install -r requirements.txt") from exc
    return OpenAI(api_key=api_key, base_url="https://api.openai.com/v1", timeout=60.0, max_retries=0)


def generate_summary(tweets: list[Tweet], summary_prompt: str,
                     model: str = DEFAULT_MODEL, client=None, provider: str = "ollama") -> dict:
    """Take the dataset and prompt as inputs; return an unchecked draft.

    A client can be injected for offline transport tests. With no injected
    client, this performs one request to the selected backend and closes afterward.
    """
    request = build_request(tweets, summary_prompt, model, provider)
    owns_client = client is None
    client = make_client(provider) if owns_client else client
    start = perf_counter()
    try:
        if provider == "ollama":
            response_data = client.chat(request)
        else:
            response = client.responses.create(**request)
            response_data = response.model_dump(mode="json")
    except Exception as exc:
        # Do not echo an arbitrary exception body that could expose credentials.
        if isinstance(exc, GenerationError):
            raise
        if provider == "ollama":
            raise GenerationError(f"Local generation failed ({type(exc).__name__}); no summary saved.") from exc
        from openai import APIStatusError, APIConnectionError
        if isinstance(exc, APIStatusError):
            hint = {401: "Check your API key.", 403: "Check model/project access.",
                    429: "Check API quota, billing, or rate limits."}.get(exc.status_code, "Check API availability and request settings.")
            raise GenerationError(f"OpenAI API returned HTTP {exc.status_code}. {hint}") from exc
        if isinstance(exc, APIConnectionError):
            raise GenerationError("Could not reach OpenAI. Check your connection and TLS certificates.") from exc
        raise GenerationError(f"Generation failed ({type(exc).__name__}); no summary saved.") from exc
    finally:
        if owns_client:
            client.close()
    elapsed = perf_counter() - start
    output = parse_response(response_data, tweets)
    return {
        "status": "draft_unchecked", "provider": provider, "topic": tweets[0].topic,
        "summary": " ".join(entry["text"].strip() for entry in output["sentences"]),
        "sentences": output["sentences"],
        "summary_prompt": summary_prompt,
        "requested_model": model, "returned_model": response_data.get("model"),
        "response_id": response_data.get("id"),
        "generation_settings": request.get("options", {"temperature": "model default"}),
        "backend_metadata": response_data.get("backend_metadata"),
        "usage": response_data.get("usage"), "execution_seconds": round(elapsed, 6),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": digest([asdict(tweet) for tweet in tweets]),
        "request_sha256": digest(request),
        "ontology_used": False, "reference_summaries_used": False,
        "checks_completed": ["JSON structure", "five entries", "source ID membership"],
        "checks_pending": ["semantic grounding", "sentence boundaries", "bias and coverage", "ontology checks"],
    }


def save_result(result: dict, path: Path) -> None:
    """Write JSON metadata and a plain-text summary into a new run directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(".txt").write_text(result["summary"] + "\n", encoding="utf-8")
