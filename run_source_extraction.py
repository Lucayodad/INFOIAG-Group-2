
import argparse
import json
from pathlib import Path

from data_loader import TOPICS, load_tweets, validate_dataset
from ollama_backend import OllamaClient
from source_claims import SourceExtractionError, extract_with_retries


ROOT = Path(__file__).resolve().parent


def save_json(path, data):
    """Keep the existing atomic checkpoint pattern; never edit prior records."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", choices=TOPICS, default="Obama")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    # Load and validate the complete source dataset.
    tweets = load_tweets(args.topic)
    validate_dataset(tweets)

    if not 1 <= args.limit <= len(tweets):
        parser.error("Limit must be between 1 and 90.")

    selected = tweets[:args.limit]

    # Each topic gets its own persisted extraction file.
    output = ROOT / "results" / "source_claims" / f"{args.topic}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Recover any previous successful extraction results.
    results = (
        json.loads(output.read_text(encoding="utf-8"))
        if output.exists() else []
    )

    if not isinstance(results, list):
        raise ValueError("Checkpoint must contain a JSON list.")

    completed = {record["tweet_id"] for record in results}

    if len(completed) != len(results):
        raise ValueError("Duplicate tweet IDs in checkpoint.")

    known_ids = {tweet.id for tweet in tweets}
    if not completed.issubset(known_ids):
        raise ValueError("Checkpoint contains unknown tweet IDs.")

    failure_file = output.with_name(f"{args.topic}.failures.json")
    failure_rows = json.loads(failure_file.read_text(encoding="utf-8")) if failure_file.exists() else []
    if not isinstance(failure_rows, list) or any(
        not isinstance(row, dict) or row.get("tweet_id") not in known_ids for row in failure_rows
    ):
        raise ValueError("Failure log must be a list of records with known tweet IDs.")
    failures = {row["tweet_id"]: row for row in failure_rows}
    if len(failures) != len(failure_rows):
        raise ValueError("Failure log contains duplicate tweet IDs.")
    # Recover a crash between saving a successful checkpoint and resolving its
    # sidecar entry. A checkpointed tweet must never be sent to Qwen again.
    reconciled = False
    for key, row in failures.items():
        resolved = key in completed
        if row.get("resolved") != resolved:
            row["resolved"] = resolved
            reconciled = True
    if reconciled:
        save_json(failure_file, list(failures.values()))

    client = OllamaClient()

    try:
        for tweet in selected:
            if tweet.id in completed:
                print(f"Skipping {tweet.id}: already saved")
                continue

            print(f"Extracting {tweet.id}...", flush=True)

            try:
                record = extract_with_retries(tweet, client, on_retry=lambda attempt, error: print(
                    f"Retrying {tweet.id}: attempt {attempt}/3; copying exact source quotations.", flush=True))
            except SourceExtractionError as exc:
                previous = failures.get(tweet.id, {})
                failures[tweet.id] = {
                    "tweet_id": tweet.id, "error": str(exc), "resolved": False,
                    "failed_attempt_count": exc.failed_attempt_count,
                    "total_failed_attempt_count": previous.get("total_failed_attempt_count", 0) + exc.failed_attempt_count,
                    "attempt_errors": exc.attempt_errors,
                }
                save_json(failure_file, list(failures.values()))
                print(f"Unresolved {tweet.id}: {exc.failed_attempt_count} attempts failed; continuing.", flush=True)
                continue

            results.append(record)
            save_json(output, results)
            completed.add(tweet.id)
            if tweet.id in failures:
                failures[tweet.id]["resolved"] = True
                save_json(failure_file, list(failures.values()))

            print(f"Saved {tweet.id}: {len(record['claims'])} claims")

    finally:
        client.close()

    print(f"\nCompleted: {len(completed)} / 90")
    print(f"Output: {output}")
    unresolved = [tweet.id for tweet in tweets if tweet.id in failures and tweet.id not in completed]
    if unresolved:
        print(f"Unresolved tweets ({len(unresolved)}): {', '.join(unresolved)}")
        print(f"Failure details: {failure_file}")
        print("Rerun the same command to retry unresolved tweets; validated tweets will be skipped.")
    else:
        print("Unresolved tweets: none.")
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
