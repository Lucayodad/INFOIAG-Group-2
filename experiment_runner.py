"""The 18 trials (3 topics x 2 prompts x 3 seeds) for two systems.

baseline: one Qwen summary per trial, no agent. ontology_agent: the same
generation as a draft, then the agent; each draft is saved under drafts/,
where rerun_agent_trials.py finds it.
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from data_loader import DEFAULT_DATA_DIR, TOPICS, DatasetError, load_tweets, validate_dataset
from prompts import get_prompt


DEFAULT_SEEDS = (101, 202, 303)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MODEL = "qwen3.5:4b"


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _generate_with_settings(tweets, prompt, model, seed, temperature, client) -> dict:
    """Generate an initial draft with the experiment-specific settings."""
    from summarizer import build_request, digest, parse_response

    request = build_request(tweets, prompt, model)
    request["options"]["temperature"] = temperature
    request["options"]["seed"] = seed
    started = perf_counter()
    response = client.chat(request)
    elapsed = perf_counter() - started
    output = parse_response(response, tweets)
    return {
        "status": "draft_unchecked",
        "provider": "ollama",
        "topic": tweets[0].topic,
        "seed": seed,
        "temperature": temperature,
        "summary": " ".join(entry["text"].strip() for entry in output["sentences"]),
        "sentences": output["sentences"],
        "summary_prompt": prompt,
        "requested_model": model,
        "returned_model": response.get("model"),
        "generation_settings": request["options"],
        "backend_metadata": response.get("backend_metadata"),
        "usage": response.get("usage"),
        "execution_seconds": round(elapsed, 6),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": digest([asdict(tweet) for tweet in tweets]),
        "request_sha256": digest(request),
        "ontology_used": False,
        "reference_summaries_used": False,
    }


def _find_graphs(graph_dir: Path, topics: tuple[str, ...]) -> dict[str, Path]:
    """Find the newest complete populated graph for each requested topic.

    Graphs may come from populate_ontology.py or run_pipeline.py. The latter
    stores them beside its pipeline result rather than under results/ontology.
    """
    graphs = {}
    for topic in topics:
        candidates = []
        for graph_path in graph_dir.glob(f"*/{topic}.ttl"):
            manifest_path = graph_path.parent / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = _read_json(manifest_path)
            if (manifest.get("topic") == topic and manifest.get("complete_topic") is True
                    and manifest.get("tweet_count") == 90):
                candidates.append((manifest_path.stat().st_mtime, graph_path))
        pipeline_root = Path(__file__).resolve().parent / "results" / "evidence" / "pipeline_runs"
        for graph_path in pipeline_root.glob(f"*_artifacts/{topic}.ttl"):
            manifest_path = graph_path.parent / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = _read_json(manifest_path)
            counts = manifest.get("group_counts", {})
            if (manifest.get("topic") == topic
                    and manifest.get("graph") == str(graph_path)
                    and sum(counts.values()) == 90
                    and all(counts.get(label) == 30 for label in ("AA", "Hisp", "White"))):
                candidates.append((manifest_path.stat().st_mtime, graph_path))
        if not candidates:
            raise ValueError(
                f"No complete 90-tweet ontology graph found for {topic} under {graph_dir}. "
                f"Run populate_ontology.py --topic {topic} first."
            )
        graphs[topic] = max(candidates, key=lambda item: item[0])[1]
    return graphs


def _load_graphs(graph_paths: dict[str, Path]):
    try:
        from rdflib import Graph
    except ImportError as exc:
        raise ValueError("Install requirements.txt before running ontology-agent experiments.") from exc
    return {topic: Graph().parse(path, format="turtle") for topic, path in graph_paths.items()}


def _agent_run(tweets, prompt, model, seed, temperature, client, graph, args) -> dict:
    """Run the existing LangGraph agent after producing its initial draft."""
    from summary_agent import run_summary_agent
    from model_usage import recording_client

    client = recording_client(client)
    start_call = len(client.calls)
    draft = _generate_with_settings(tweets, prompt, model, seed, temperature, client)
    started = perf_counter()
    agent = run_summary_agent(
        graph,
        draft,
        topic=tweets[0].topic,
        prompt=prompt,
        client=client,
        max_revisions=args.max_revisions,
        embedding_model=args.embedding_model,
    )
    elapsed = perf_counter() - started + draft["execution_seconds"]
    final = agent["current_summary"]
    call_counts = agent.get("call_counts", {})
    result = {
        "status": "completed",
        "system": "ontology_agent",
        "provider": "ollama",
        "topic": tweets[0].topic,
        "seed": seed,
        "temperature": temperature,
        "summary": final["summary"],
        "sentences": final["sentences"],
        "summary_prompt": prompt,
        "requested_model": model,
        "returned_model": draft.get("returned_model"),
        "initial_summary": draft,
        "agent_result": agent,
        "execution_seconds": round(elapsed, 6),
        "llm_calls": 1 + call_counts.get("model_calls", 0),
        "tool_calls": call_counts.get("tool_calls", 0),
        "embedding_calls": call_counts.get("embedding_calls", 0),
        **client.accounting(start_call),
        "usage_scope": "draft_and_all_agent_calls_excludes_source_preprocessing",
        "ontology_used": True,
        "reference_summaries_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return result


def run_experiment(args) -> int:
    topics = TOPICS if args.topic == "all" else (args.topic,)
    prompt_ids = ("1", "2") if args.prompt_id == "all" else (args.prompt_id,)
    systems = ("baseline", "ontology_agent") if args.system == "all" else (args.system,)
    seeds = tuple(args.seeds)
    model = args.model or DEFAULT_MODEL

    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than zero for the experiment.")
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("--seeds must contain exactly three different integer values.")

    batches = {}
    for topic in topics:
        tweets = load_tweets(topic, args.data_dir)
        validate_dataset(tweets)
        batches[topic] = tweets
    graph_paths = _find_graphs(args.graph_dir, topics) if "ontology_agent" in systems else {}
    graphs = _load_graphs(graph_paths)

    if args.resume_manifest:
        manifest_path = args.resume_manifest.resolve()
        manifest = _read_json(manifest_path)
        destination = manifest_path.parent
        expected = {"model": model, "temperature": args.temperature, "seeds": list(seeds),
                    "systems": list(systems), "topics": list(topics), "prompt_ids": list(prompt_ids)}
        mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
        if mismatches:
            raise ValueError(f"Resume manifest settings differ for: {', '.join(mismatches)}.")
        if not isinstance(manifest.get("runs"), list):
            raise ValueError("Resume manifest has no valid runs list.")
    else:
        run_id = _utc_run_id()
        destination = args.output_dir / "experiments" / run_id
        destination.mkdir(parents=True, exist_ok=False)
        manifest = {
            "status": "started",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "temperature": args.temperature,
            "seeds": list(seeds),
            "systems": list(systems),
            "topics": list(topics),
            "prompt_ids": list(prompt_ids),
            "expected_final_summaries": len(systems) * len(topics) * len(prompt_ids) * len(seeds),
            "graph_paths": {topic: str(path) for topic, path in graph_paths.items()},
            "agent_options": {
                "max_revisions": args.max_revisions,
                "embedding_model": args.embedding_model,
            },
            "runs": [],
        }
    _write_json(destination / "manifest.json", manifest)

    existing = {
        record.get("run_id"): record for record in manifest["runs"]
        if isinstance(record, dict) and record.get("run_id")
    }
    from ollama_backend import OllamaClient
    client = OllamaClient(timeout=args.timeout)
    try:
        for system in systems:
            for topic in topics:
                for prompt_id in prompt_ids:
                    prompt = get_prompt(topic, prompt_id)
                    for seed in seeds:
                        key = f"{system}_{topic}_prompt{prompt_id}_seed{seed}"
                        if existing.get(key, {}).get("status") == "completed":
                            print(f"skipped_completed: {key}")
                            continue
                        record = {
                            "run_id": key,
                            "system": system,
                            "topic": topic,
                            "prompt_id": prompt_id,
                            "seed": seed,
                            "temperature": args.temperature,
                            "model": model,
                        }
                        if system == "baseline":
                            result = _generate_with_settings(
                                batches[topic], prompt, model, seed, args.temperature, client
                            )
                            result["system"] = "baseline"
                        else:
                            result = _agent_run(
                                batches[topic], prompt, model, seed, args.temperature,
                                client, graphs[topic], args
                            )
                            _write_json(destination / "drafts" / f"{key}.json", result["initial_summary"])
                            record["draft_path"] = str(Path("drafts") / f"{key}.json")
                        result["prompt_id"] = prompt_id
                        _write_json(destination / "summaries" / f"{key}.json", result)
                        record["status"] = "completed"
                        record["output_path"] = str(Path("summaries") / f"{key}.json")
                        existing[key] = record
                        manifest["runs"] = list(existing.values())
                        _write_json(destination / "manifest.json", manifest)
                        print(f"{record['status']}: {key}")
    finally:
        client.close()

    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(destination / "manifest.json", manifest)
    print(f"\nExperiment manifest: {destination / 'manifest.json'}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("all", "baseline", "ontology_agent"), default="all")
    parser.add_argument("--topic", choices=("all", *TOPICS), default="all")
    parser.add_argument("--prompt-id", choices=("1", "2", "all"), default="all")
    parser.add_argument("--seeds", nargs=3, type=int, default=list(DEFAULT_SEEDS),
                        metavar=("SEED1", "SEED2", "SEED3"))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--graph-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "ontology",
                        help="Directory containing complete populated topic graphs.")
    parser.add_argument("--max-revisions", type=int, default=3)
    parser.add_argument("--embedding-model", default="all-minilm")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--resume-manifest", type=Path,
                        help="Continue an existing experiment manifest; completed runs are skipped.")
    args = parser.parse_args()
    try:
        raise SystemExit(run_experiment(args))
    except (DatasetError, OSError, ValueError) as exc:
        parser.exit(1, f"Experiment error: {exc}\n")


if __name__ == "__main__":
    main()
