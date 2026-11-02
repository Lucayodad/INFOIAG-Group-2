"""Execute and record the planned 2 x 3 x 2 x 3 summary experiment."""

import argparse
import json
import math
import os
import re
import statistics
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
DEFAULT_REFERENCES = (
    Path(__file__).resolve().parent / "data" / "divsumm" / "ready_to_use" / "references.json"
)


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

    request = build_request(tweets, prompt, model, "ollama")
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


def _usage_totals(value) -> dict[str, int]:
    """Sum Ollama usage dictionaries in an agent result without guessing costs."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if isinstance(value, dict):
        usage = value.get("usage")
        if isinstance(usage, dict):
            for key in totals:
                number = usage.get(key)
                if type(number) is int and number >= 0:
                    totals[key] += number
        for child in value.values():
            child_totals = _usage_totals(child)
            for key in totals:
                totals[key] += child_totals[key]
    elif isinstance(value, list):
        for child in value:
            child_totals = _usage_totals(child)
            for key in totals:
                totals[key] += child_totals[key]
    return totals


def _agent_run(tweets, prompt, model, seed, temperature, client, graph, args) -> dict:
    """Run the existing LangGraph agent after producing its initial draft."""
    from summary_agent import run_summary_agent

    draft = _generate_with_settings(tweets, prompt, model, seed, temperature, client)
    started = perf_counter()
    agent = run_summary_agent(
        graph,
        draft,
        topic=tweets[0].topic,
        prompt=prompt,
        client=client,
        disparity_threshold=args.disparity_threshold,
        max_revisions=args.max_revisions,
        embedding_model=args.embedding_model,
        representation_metric="mnli",
    )
    elapsed = perf_counter() - started + draft["execution_seconds"]
    final = agent["current_summary"]
    call_counts = agent.get("call_counts", {})
    draft_usage = _usage_totals(draft)
    agent_usage = _usage_totals({
        "factual_results": agent.get("factual_results"),
        "representation_results": agent.get("representation_results"),
        "revisions": agent.get("revisions"),
    })
    usage = {
        key: draft_usage[key] + agent_usage[key]
        for key in draft_usage
    }
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
        "usage": usage,
        "usage_breakdown": {"draft": draft_usage, "agent": agent_usage},
        "ontology_used": True,
        "reference_summaries_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return result


def _tokens(text: str) -> list[str]:
    """Use one deterministic, dependency-free tokenizer for all ROUGE scores."""
    return re.findall(r"\w+(?:['’-]\w+)?", text.casefold(), flags=re.UNICODE)


def _ngrams(tokens: list[str], size: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)]


def _rouge_f1(candidate: str, reference: str, size: int) -> float:
    candidate_ngrams = _ngrams(_tokens(candidate), size)
    reference_ngrams = _ngrams(_tokens(reference), size)
    if not candidate_ngrams or not reference_ngrams:
        return 0.0
    candidate_counts = {}
    reference_counts = {}
    for item in candidate_ngrams:
        candidate_counts[item] = candidate_counts.get(item, 0) + 1
    for item in reference_ngrams:
        reference_counts[item] = reference_counts.get(item, 0) + 1
    overlap = sum(min(count, reference_counts.get(item, 0))
                  for item, count in candidate_counts.items())
    precision = overlap / len(candidate_ngrams)
    recall = overlap / len(reference_ngrams)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, 1):
            current.append(previous[index - 1] + 1
                           if left_token == right_token
                           else max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def _rouge_l_f1(candidate: str, reference: str) -> float:
    candidate_tokens = _tokens(candidate)
    reference_tokens = _tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    overlap = _lcs_length(candidate_tokens, reference_tokens)
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _load_abstractive_references(path: Path) -> dict[str, list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read references file {path}: {exc}") from exc
    try:
        return {
            topic: [item["text"] for item in values["abstractive"]]
            for topic, values in data.items()
        }
    except (KeyError, TypeError):
        raise ValueError(f"References file {path} has no valid abstractive references.")


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def _sample_sd(values: list[float]) -> float | None:
    return round(statistics.stdev(values), 6) if len(values) >= 2 else None


def _evaluate_summary(path: Path, references: dict[str, list[str]]) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read summary {path}: {exc}") from exc
    topic = result.get("topic")
    summary = result.get("summary")
    if not isinstance(topic, str) or not isinstance(summary, str) or not summary.strip():
        raise ValueError(f"Summary {path} lacks a topic or nonempty summary.")
    topic_references = references.get(topic)
    if not topic_references:
        raise ValueError(f"No abstractive references found for topic {topic!r}.")

    scores = {
        "rouge1": [_rouge_f1(summary, reference, 1) for reference in topic_references],
        "rouge2": [_rouge_f1(summary, reference, 2) for reference in topic_references],
        "rougeL": [_rouge_l_f1(summary, reference) for reference in topic_references],
    }
    return {
        "run_id": path.stem,
        "system": result.get("system", "baseline"),
        "topic": topic,
        "prompt_id": result.get("prompt_id"),
        "seed": result.get("seed"),
        "summary_path": str(path),
        "rouge1_by_reference": scores["rouge1"],
        "rouge2_by_reference": scores["rouge2"],
        "rougeL_by_reference": scores["rougeL"],
        "rouge1_mean": _mean(scores["rouge1"]),
        "rouge2_mean": _mean(scores["rouge2"]),
        "rougeL_mean": _mean(scores["rougeL"]),
        "execution_seconds": result.get("execution_seconds"),
        "llm_calls": result.get("llm_calls", 1),
        "input_tokens": (result.get("usage") or {}).get("input_tokens"),
        "output_tokens": (result.get("usage") or {}).get("output_tokens"),
        "total_tokens": (result.get("usage") or {}).get("total_tokens"),
        "unavailable_metrics": [
            "equal coverage and dialect group coverage",
            "embedding-based group similarity gap",
            "DeepEval alignment and coverage",
        ],
    }


def _numeric_values(records: list[dict], field: str) -> list[float]:
    return [record[field] for record in records
            if isinstance(record.get(field), (int, float))
            and not isinstance(record.get(field), bool)
            and math.isfinite(record[field])]


def _aggregate_evaluations(records: list[dict]) -> list[dict]:
    groups = {}
    for record in records:
        for group_name, group_keys in (
            ("overall", ()),
            ("system", ("system",)),
            ("topic", ("topic",)),
            ("system_topic_prompt", ("system", "topic", "prompt_id")),
        ):
            key = (group_name, tuple(record.get(name) for name in group_keys))
            groups.setdefault(key, []).append(record)

    aggregate = []
    metric_names = ("rouge1_mean", "rouge2_mean", "rougeL_mean",
                    "execution_seconds", "llm_calls", "input_tokens",
                    "output_tokens", "total_tokens")
    for (group_name, group_values), group_records in groups.items():
        row = {"group": group_name, "count": len(group_records)}
        row.update(dict(zip(
            ("system", "topic", "prompt_id"), group_values
        )))
        for metric in metric_names:
            values = _numeric_values(group_records, metric)
            row[f"{metric}_mean"] = _mean(values)
            row[f"{metric}_sd"] = _sample_sd(values)
        aggregate.append(row)
    return aggregate


def evaluate_manifest(manifest_path: Path, references_path: Path) -> Path:
    """Evaluate completed summary files and write per-run and aggregate JSON."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read manifest {manifest_path}: {exc}") from exc
    references = _load_abstractive_references(references_path)
    summary_dir = manifest_path.parent / "summaries"
    summary_paths = sorted(summary_dir.glob("*.json")) if summary_dir.exists() else []
    records = [_evaluate_summary(path, references) for path in summary_paths]
    output = {
        "status": "partial" if records else "no_completed_summaries",
        "manifest": str(manifest_path),
        "references": str(references_path),
        "evaluated_summaries": len(records),
        "records": records,
        "aggregates": _aggregate_evaluations(records),
        "unavailable_metrics": [
            "Equal Coverage requires the planned MNLI entailment implementation.",
            "Group similarity gap requires the planned embedding implementation.",
            "DeepEval requires the planned local judge integration.",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path = manifest_path.parent / "evaluation.json"
    _write_json(output_path, output)
    return output_path


def run_experiment(args) -> int:
    topics = TOPICS if args.topic == "all" else (args.topic,)
    prompt_ids = ("1", "2") if args.prompt_id == "all" else (args.prompt_id,)
    systems = ("baseline", "ontology_agent") if args.system == "all" else (args.system,)
    seeds = tuple(args.seeds)
    model = args.model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)

    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than zero for the experiment.")
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("--seeds must contain exactly three different integer values.")
    if args.provider != "ollama":
        raise ValueError("The planned experiment uses the local Ollama model; use --provider ollama.")

    batches = {}
    for topic in topics:
        tweets = load_tweets(topic, args.data_dir)
        validate_dataset(tweets)
        batches[topic] = tweets

    graph_paths = {}
    graphs = {}
    if "ontology_agent" in systems and not args.dry_run:
        graph_paths = _find_graphs(args.graph_dir, topics)
        graphs = _load_graphs(graph_paths)

    if args.resume_manifest:
        manifest_path = args.resume_manifest.resolve()
        manifest = _read_json(manifest_path)
        destination = manifest_path.parent
        expected = {
            "model": model, "provider": args.provider, "temperature": args.temperature,
            "seeds": list(seeds), "systems": list(systems), "topics": list(topics),
            "prompt_ids": list(prompt_ids),
        }
        mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
        if mismatches:
            raise ValueError(f"Resume manifest settings differ for: {', '.join(mismatches)}.")
        if not isinstance(manifest.get("runs"), list):
            raise ValueError("Resume manifest has no valid runs list.")
        run_id = manifest.get("run_id")
    else:
        run_id = _utc_run_id()
        destination = args.output_dir / "experiments" / run_id
        destination.mkdir(parents=True, exist_ok=False)
        manifest = {
            "status": "started",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "provider": args.provider,
            "temperature": args.temperature,
            "seeds": list(seeds),
            "systems": list(systems),
            "topics": list(topics),
            "prompt_ids": list(prompt_ids),
            "expected_final_summaries": len(systems) * len(topics) * len(prompt_ids) * len(seeds),
            "graph_paths": {topic: str(path) for topic, path in graph_paths.items()},
            "agent_options": {
                "disparity_threshold": args.disparity_threshold,
                "max_revisions": args.max_revisions,
                "embedding_model": args.embedding_model,
            },
            "runs": [],
            "evaluation": {
                "status": "not_implemented",
                "pending": ["EC", "group coverage values", "group similarity gap",
                            "DeepEval alignment and coverage", "ROUGE", "aggregation"],
            },
        }
    _write_json(destination / "manifest.json", manifest)

    client = None
    existing = {
        record.get("run_id"): record for record in manifest["runs"]
        if isinstance(record, dict) and record.get("run_id")
    }
    try:
        if not args.dry_run:
            from ollama_backend import OllamaClient
            client = OllamaClient(timeout=args.timeout)

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
                        if args.dry_run:
                            if system == "baseline":
                                from summarizer import build_request
                                request = build_request(batches[topic], prompt, model, "ollama")
                                request["options"]["temperature"] = args.temperature
                                request["options"]["seed"] = seed
                                _write_json(destination / "requests" / f"{key}.json", request)
                                record["status"] = "request_saved"
                            else:
                                record.update({
                                    "status": "agent_request_pending",
                                    "note": "Dry run does not execute the LangGraph agent.",
                                })
                        elif system == "baseline":
                            result = _generate_with_settings(
                                batches[topic], prompt, model, seed, args.temperature, client
                            )
                            result["system"] = "baseline"
                            result["prompt_id"] = prompt_id
                            _write_json(destination / "summaries" / f"{key}.json", result)
                            record["status"] = "completed"
                            record["output_path"] = str(Path("summaries") / f"{key}.json")
                        else:
                            result = _agent_run(
                                batches[topic], prompt, model, seed, args.temperature,
                                client, graphs[topic], args
                            )
                            result["prompt_id"] = prompt_id
                            _write_json(destination / "drafts" / f"{key}.json",
                                        result["initial_summary"])
                            _write_json(destination / "summaries" / f"{key}.json", result)
                            record["status"] = "completed"
                            record["output_path"] = str(Path("summaries") / f"{key}.json")
                            record["draft_path"] = str(Path("drafts") / f"{key}.json")
                        existing[key] = record
                        manifest["runs"] = list(existing.values())
                        _write_json(destination / "manifest.json", manifest)
                        print(f"{record['status']}: {key}")
    finally:
        if client is not None:
            client.close()

    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(destination / "manifest.json", manifest)
    print(f"\nExperiment manifest: {destination / 'manifest.json'}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-manifest", type=Path,
                        help="Evaluate completed summaries from an experiment manifest.")
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES,
                        help="JSON file containing the two abstractive references per topic.")
    parser.add_argument("--system", choices=("all", "baseline", "ontology_agent"), default="all")
    parser.add_argument("--topic", choices=("all", *TOPICS), default="all")
    parser.add_argument("--prompt-id", choices=("1", "2", "all"), default="all")
    parser.add_argument("--seeds", nargs=3, type=int, default=list(DEFAULT_SEEDS),
                        metavar=("SEED1", "SEED2", "SEED3"))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--provider", choices=("ollama",), default="ollama")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--graph-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "ontology",
                        help="Directory containing complete populated topic graphs.")
    parser.add_argument("--disparity-threshold", type=float, default=0.25,
                        help="Agent revision threshold used by the internal checker.")
    parser.add_argument("--max-revisions", type=int, default=3)
    parser.add_argument("--embedding-model", default="all-minilm")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--resume-manifest", type=Path,
                        help="Continue an existing experiment manifest; completed runs are skipped.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Save all baseline request payloads without calling Ollama.")
    args = parser.parse_args()
    try:
        if args.evaluate_manifest is not None:
            output_path = evaluate_manifest(args.evaluate_manifest, args.references)
            print(f"Evaluation written: {output_path}")
            raise SystemExit(0)
        raise SystemExit(run_experiment(args))
    except (DatasetError, OSError, ValueError) as exc:
        parser.exit(1, f"Experiment error: {exc}\n")


if __name__ == "__main__":
    main()
