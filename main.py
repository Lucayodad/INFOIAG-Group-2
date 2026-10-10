"""Run data loading, ontology queries, or initial summary generation."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from data_loader import DEFAULT_DATA_DIR, TOPICS, DatasetError, load_tweets, validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("data", "ontology", "draft"), default="data")
    parser.add_argument("--topic", choices=("all", *TOPICS), default="all")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ontology-file", type=Path)
    parser.add_argument("--query", choices=("all", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6"), default="all")
    parser.add_argument("--asserted-only", action="store_true")
    parser.add_argument("--prompt-id", choices=("1", "2", "all"), default="1")
    parser.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--model", help="Override the provider's default model.")
    parser.add_argument("--dry-run", action="store_true", help="Save request previews; make no API calls.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    if args.section == "draft":
        from prompts import get_prompt
        from summarizer import build_request, generate_summary, save_result, GenerationError, make_client
        from summarizer import DEFAULT_MODEL, OPENAI_MODEL
        args.model = args.model or os.environ.get(
            "OLLAMA_MODEL" if args.provider == "ollama" else "OPENAI_MODEL",
            DEFAULT_MODEL if args.provider == "ollama" else OPENAI_MODEL)
        topics = TOPICS if args.topic == "all" else (args.topic,)
        prompt_ids = ("1", "2") if args.prompt_id == "all" else (args.prompt_id,)
        client = None
        try:
            # Validate all requested inputs before making any generation request.
            batches = {topic: load_tweets(topic, args.data_dir) for topic in topics}
            for tweets in batches.values():
                validate_dataset(tweets)
            if not args.dry_run:
                client = make_client(args.provider)
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
            destination = args.output_dir / ("previews" if args.dry_run else "drafts") / run_id
            destination.mkdir(parents=True, exist_ok=False)
            for topic, tweets in batches.items():
                for prompt_id in prompt_ids:
                    prompt = get_prompt(topic, prompt_id)
                    name = f"{topic}_prompt{prompt_id}"
                    if args.dry_run:
                        path = destination / f"{name}.request.json"
                        path.write_text(json.dumps(build_request(tweets, prompt, args.model, args.provider),
                                                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                        print(f"Preview saved: {path}")
                    else:
                        result = generate_summary(tweets, prompt, args.model, client=client, provider=args.provider)
                        result["prompt_id"] = prompt_id
                        path = destination / f"{name}.json"
                        save_result(result, path)
                        print(f"\n{topic}, prompt {prompt_id}:\n{result['summary']}\nSaved: {path}")
            print("\nNo API calls made; no summaries generated." if args.dry_run else
                  "\nDraft generation complete. Ontology/bias checking and revision are still pending.")
        except (DatasetError, GenerationError, OSError) as exc:
            parser.exit(1, f"Draft error: {exc}\n")
        finally:
            if client is not None:
                client.close()
        return

    if args.section == "ontology":
        try:
            from ontology_tools import OntologyTools, OntologyError, QUERY_DEFAULTS
        except ModuleNotFoundError as exc:
            parser.exit(1, f"Missing dependency {exc.name}. Run: python3 -m pip install -r requirements.txt\n")
        try:
            tools = OntologyTools(args.ontology_file) if args.ontology_file else OntologyTools()
            print(json.dumps(tools.statistics(), indent=2))
            queries = QUERY_DEFAULTS if args.query == "all" else (args.query,)
            for query in queries:
                result = tools.run_query(query, use_inference=not args.asserted_only)
                readable = [{k: v.get("label", v["value"]) for k, v in row.items()} for row in result["rows"]]
                print(f"\n{query}: {result['status']}")
                print(json.dumps(readable, indent=2, ensure_ascii=False))
            print("\nOntology assertions only; no match means unknown/absent, not false.")
            print("Q6 checks named disjoint classes only, not complete OWL consistency.")
        except OntologyError as exc:
            parser.exit(1, f"Ontology error: {exc}\n")
        return

    topics = TOPICS if args.topic == "all" else (args.topic,)
    total = 0
    try:
        for topic in topics:
            tweets = load_tweets(topic, args.data_dir)
            counts = validate_dataset(tweets)
            total += len(tweets)
            print(
                f"{topic}: {len(tweets)} tweets | "
                f"AA={counts['AA']} | Hisp={counts['Hisp']} | White={counts['White']} | OK"
            )
    except DatasetError as exc:
        parser.exit(1, f"Dataset error: {exc}\n")

    print(f"\nSection 1 complete: {total} tweets across {len(topics)} topic(s).")


if __name__ == "__main__":
    main()
