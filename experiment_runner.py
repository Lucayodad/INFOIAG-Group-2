"""Execute and record the planned 2 x 3 x 2 x 3 summary experiment.

The baseline system is executable with the current repository.  The
ontology-agent hook is deliberately isolated below because the current
repository does not yet expose the draft/check/revise/recheck pipeline.
Agent runs are recorded as pending placeholders; no synthetic summaries are
created.
"""

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


def _baseline_run(tweets, prompt, model, seed, temperature, client) -> dict:
    """Run one baseline request while varying only the experiment settings."""
    from summarizer import build_request, digest, parse_response

    request = build_request(tweets, prompt, model, "ollama")
    request["options"]["temperature"] = temperature
    request["options"]["seed"] = seed
    started = perf_counter()
    response = client.chat(request)
    elapsed = perf_counter() - started
    output = parse_response(response, tweets)
    return {
        "status": "completed",
        "system": "baseline",
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
        "checks_completed": ["dataset validation", "JSON structure", "five entries",
                             "source ID membership"],
        "checks_pending": ["EC", "group similarity gap", "DeepEval", "ROUGE"],
    }


def _ontology_agent_placeholder(topic, prompt_id, seed, temperature, model) -> dict:
    """Explicit hand-off point for the Section 5 pipeline.

    Replace this function with draft -> ontology check -> revision -> re-check.
    Keeping the record makes incomplete execution visible in the manifest.
    """
    return {
        "status": "pending_ontology_pipeline",
        "system": "ontology_agent",
        "topic": topic,
        "prompt_id": prompt_id,
        "seed": seed,
        "temperature": temperature,
        "requested_model": model,
        "ontology_used": False,
        "reference_summaries_used": False,
        "draft": None,
        "final": None,
        "checks_pending": ["draft generation", "ontology checks", "revision",
                           "re-check", "EC", "group similarity gap", "DeepEval", "ROUGE"],
        "note": "Placeholder only: the ontology-driven pipeline is not exposed by the current repository.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


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
        "runs": [],
        "evaluation": {
            "status": "not_implemented",
            "pending": ["EC", "group coverage values", "group similarity gap",
                        "DeepEval alignment and coverage", "ROUGE", "aggregation"],
        },
    }
    _write_json(destination / "manifest.json", manifest)

    client = None
    try:
        if "baseline" in systems and not args.dry_run:
            from ollama_backend import OllamaClient
            client = OllamaClient(timeout=args.timeout)

        for system in systems:
            for topic in topics:
                for prompt_id in prompt_ids:
                    prompt = get_prompt(topic, prompt_id)
                    for seed in seeds:
                        key = f"{system}_{topic}_prompt{prompt_id}_seed{seed}"
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
                                record.update(_ontology_agent_placeholder(
                                    topic, prompt_id, seed, args.temperature, model))
                        elif system == "baseline":
                            result = _baseline_run(
                                batches[topic], prompt, model, seed, args.temperature, client
                            )
                            result["prompt_id"] = prompt_id
                            _write_json(destination / "summaries" / f"{key}.json", result)
                            record["status"] = "completed"
                            record["output_path"] = str(Path("summaries") / f"{key}.json")
                        else:
                            record.update(_ontology_agent_placeholder(
                                topic, prompt_id, seed, args.temperature, model))
                        manifest["runs"].append(record)
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
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
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
