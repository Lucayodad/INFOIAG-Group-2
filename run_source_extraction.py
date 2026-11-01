
import argparse
import json
from pathlib import Path

from data_loader import TOPICS, load_tweets, validate_dataset
from ollama_backend import OllamaClient
from source_claims import extract_one


ROOT = Path(__file__).resolve().parent


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

    client = OllamaClient()

    try:
        for tweet in selected:
            if tweet.id in completed:
                print(f"Skipping {tweet.id}: already saved")
                continue

            print(f"Extracting {tweet.id}...", flush=True)

            # Use the existing validated extractor.
            record = extract_one(tweet, client)

            results.append(record)
            completed.add(tweet.id)

            # Save after every successful tweet.
            # The temporary file prevents partially written JSON.
            temporary = output.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(results, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            temporary.replace(output)

            print(f"Saved {tweet.id}: {len(record['claims'])} claims")

    finally:
        client.close()

    print(f"\nCompleted: {len(completed)} / 90")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
