"""Run one topic/prompt from raw tweets through the complete local LangGraph agent."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parent


class ProgressClient:
    """Log bounded call progress while preserving the existing client interface."""
    def __init__(self, client):
        self.client, self.calls = client, 0

    def chat(self, payload):
        self.calls += 1
        fields = set(payload.get("format", {}).get("properties", {}))
        phase = ("initial summary" if "sentences" in fields else "factual check" if "action" in fields else
                 "revision" if "replacements" in fields else "equivalence" if "decisions" in fields else
                 "representation" if "clusters" in fields else "claim extraction")
        print(f"  Qwen {self.calls}: {phase} ...", flush=True)
        start = perf_counter()
        result = self.client.chat(payload)
        print(f"  Qwen {self.calls}: finished in {perf_counter() - start:.1f}s", flush=True)
        return result

    def embed(self, texts, model="all-minilm"):
        print(f"  Embedding {len(texts)} texts with {model} ...", flush=True)
        return self.client.embed(texts, model=model)


def validate_reused_draft(draft, topic, prompt_id, tweets):
    """Require unambiguous topic/prompt provenance before reusing a draft."""
    from prompts import get_prompt
    from summarizer import validate_output
    expected_prompt = get_prompt(topic, prompt_id)
    if (not isinstance(draft, dict) or draft.get("topic") != topic
            or draft.get("prompt_id", prompt_id) != prompt_id
            or draft.get("summary_prompt", expected_prompt) != expected_prompt
            or not any(draft.get(key) for key in ("prompt_id", "summary_prompt"))):
        raise ValueError("Reused draft topic/prompt does not match the selected run or is missing prompt provenance.")
    validated = validate_output(json.dumps({"sentences": draft.get("sentences")}), tweets)
    return {**draft, "prompt_id": prompt_id, "summary_prompt": expected_prompt,
            "summary": " ".join(row["text"].strip() for row in validated["sentences"])}


def run_pipeline(args, client=None):
    """Reuse source checkpointing, RDF population, generation and the agent API."""
    from data_loader import load_tweets, validate_dataset
    from ollama_backend import OllamaClient
    from populate_ontology import populate_ontology
    from prompts import get_prompt
    from run_source_extraction import resume_source_extraction, save_json
    from summarizer import generate_summary, save_result
    from summary_agent import print_run_summary, run_summary_agent

    if not 0 <= args.disparity_threshold <= 1 or not 0 <= args.max_revisions <= 3:
        raise ValueError("Threshold must be in [0, 1]; max revisions must be in [0, 3].")
    if args.max_model_calls < 0 or args.max_tool_calls < 0:
        raise ValueError("Call budgets must be nonnegative.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = args.output or ROOT / "results" / "evidence" / "pipeline_runs" / f"{args.topic}_prompt{args.prompt_id}_{run_id}.json"
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists; choose another --output: {output}")
    artifacts = output.parent / (output.stem + "_artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    checkpoint = args.source_claims or ROOT / "results" / "source_claims" / f"{args.topic}.json"
    owns_client = client is None
    backend = OllamaClient() if owns_client else client
    progress = ProgressClient(backend)
    stage = "preflight"
    try:
        print("[1/6] Checking local dependencies, models and tweet data", flush=True)
        models = backend.check_models()
        tweets = load_tweets(args.topic)
        counts = validate_dataset(tweets)
        stage = "source_extraction"
        print(f"[2/6] Source claims: {len(tweets)} tweets; resume {checkpoint}", flush=True)
        records, unresolved = resume_source_extraction(tweets, checkpoint, client=progress)
        if unresolved:
            raise ValueError("Unresolved source extractions: " + ", ".join(unresolved) + ". Rerun the same command to resume.")
        stage = "ontology"
        print("[3/6] Populating isolated read-only checking evidence", flush=True)
        graph = populate_ontology(tweets, records, run_id=run_id)
        graph_file = artifacts / f"{args.topic}.ttl"
        graph.serialize(destination=graph_file, format="turtle")
        stage = "draft"
        print("[4/6] Preparing five-sentence initial summary", flush=True)
        if args.reuse_draft:
            draft = json.loads(args.reuse_draft.read_text(encoding="utf-8"))
            draft = validate_reused_draft(draft, args.topic, args.prompt_id, tweets)
            print(f"  Reusing explicitly selected draft: {args.reuse_draft}", flush=True)
        else:
            draft = generate_summary(tweets, get_prompt(args.topic, args.prompt_id), client=progress)
            draft["prompt_id"] = args.prompt_id
        draft_file = artifacts / f"{args.topic}_prompt{args.prompt_id}.json"
        save_result(draft, draft_file)
        manifest = {"topic": args.topic, "prompt_id": args.prompt_id, "run_id": run_id,
                    "source_claims": str(checkpoint.resolve()), "graph": str(graph_file),
                    "draft": str(draft_file), "output": str(output), "models": models,
                    "draft_reused": args.reuse_draft is not None, "group_counts": counts}
        save_json(artifacts / "manifest.json", manifest)
        stage = "agent"
        print("[5/6] LangGraph: extract -> facts -> representation -> route -> revision/recheck when justified", flush=True)
        result = run_summary_agent(graph, draft, topic=args.topic, prompt=draft.get("summary_prompt") or get_prompt(args.topic, args.prompt_id),
            client=progress, disparity_threshold=args.disparity_threshold, max_revisions=args.max_revisions,
            max_model_calls=args.max_model_calls, max_tool_calls=args.max_tool_calls)
        result["pipeline"] = manifest
        save_json(output, result)
        print("[6/6] Complete result saved", flush=True)
        print_run_summary(result, output)
        return result
    except Exception as exc:
        save_json(artifacts / "error.json", {"stage": stage, "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        if owns_client:
            backend.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", choices=("Obama", "McDonalds", "Superbowl"), required=True)
    parser.add_argument("--prompt-id", choices=("1", "2"), default="1")
    parser.add_argument("--disparity-threshold", type=float, default=0.25, help="Exploratory threshold, not paper-established")
    parser.add_argument("--max-revisions", type=int, default=3)
    parser.add_argument("--output", type=Path, help="New final JSON path; default: unique file under results/evidence/pipeline_runs")
    parser.add_argument("--source-claims", type=Path, help="Checkpoint path; default: results/source_claims/<topic>.json")
    parser.add_argument("--reuse-draft", type=Path, help="Explicitly reuse a saved draft instead of generating a new one")
    parser.add_argument("--max-model-calls", type=int, default=64, help="Representation chat budget per checking pass")
    parser.add_argument("--max-tool-calls", type=int, default=3, help="Factual retrieval budget per summary claim")
    args = parser.parse_args(argv)
    try:
        run_pipeline(args)
    except ImportError as exc:
        parser.exit(1, f"Missing Python dependency ({exc.name}). Install: python -m pip install -r requirements.txt\n")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Pipeline error: {exc}\n")


if __name__ == "__main__":
    main()
