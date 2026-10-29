"""Populate an isolated RDF graph with saved source-claim extractions.

No model calls, truth assessments, or edits to the base OWL file are made.
The CLI requires complete topic evidence unless --sample is explicitly used.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import quote
from uuid import uuid4

from rdflib import Graph, Literal, Namespace, RDF, RDFS

from data_loader import DIALECT_LABELS, TOPICS, DatasetError, Tweet, load_tweets, validate_dataset
from ontology_tools import DEFAULT_ONTOLOGY, NS, OntologyTools
from source_claims import find_source_quote


ROOT = Path(__file__).resolve().parent
EVIDENCE_SCHEMA = ROOT / "ontology" / "evidence_schema.ttl"
STATUSES = {"extracted", "unclear", "no_claim"}
STANCES = {"positive", "negative", "neutral", "unclear"}


class EvidenceError(ValueError):
    """Evidence cannot be connected reliably to its original source tweets."""


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex


def _text(value, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{context}: expected a nonempty string.")
    return value


def _validate_records(tweets: list[Tweet], records: list[dict]) -> dict[str, dict]:
    """Require exactly one consistent extraction record per supplied tweet."""
    if not tweets:
        raise EvidenceError("At least one source tweet is required.")
    tweets_by_id = {}
    for tweet in tweets:
        _text(tweet.id, "Tweet ID")
        _text(tweet.text, f"{tweet.id}: tweet text")
        _text(tweet.source_file, f"{tweet.id}: source file")
        if type(tweet.source_line) is not int or tweet.source_line < 1:
            raise EvidenceError(f"{tweet.id}: invalid source line.")
        if tweet.topic not in TOPICS or tweet.dialect_label not in DIALECT_LABELS:
            raise EvidenceError(f"{tweet.id}: unknown topic or dialect label.")
        if tweet.id in tweets_by_id:
            raise EvidenceError(f"Duplicate source tweet: {tweet.id}")
        tweets_by_id[tweet.id] = tweet
    if len({tweet.topic for tweet in tweets}) != 1:
        raise EvidenceError("Expected source tweets from exactly one topic.")
    if not isinstance(records, list):
        raise EvidenceError("Extraction records must be a JSON list.")

    records_by_id = {}
    claim_ids = set()
    for record in records:
        if not isinstance(record, dict):
            raise EvidenceError("Each extraction record must be an object.")
        tweet_id = _text(record.get("tweet_id"), "Extraction tweet ID")
        if tweet_id not in tweets_by_id:
            raise EvidenceError(f"Unknown tweet in extraction: {tweet_id}")
        if tweet_id in records_by_id:
            raise EvidenceError(f"Duplicate extraction record: {tweet_id}")
        tweet = tweets_by_id[tweet_id]
        if record.get("dialect_label") != tweet.dialect_label:
            raise EvidenceError(f"{tweet_id}: extraction dialect mismatch.")
        status = _text(record.get("status"), f"{tweet_id}: extraction status")
        entries = record.get("claims")
        if status not in STATUSES or not isinstance(entries, list):
            raise EvidenceError(f"{tweet_id}: invalid extraction status or claims list.")
        if (status == "extracted") != bool(entries):
            raise EvidenceError(f"{tweet_id}: inconsistent extraction status and claims.")

        for claim in entries:
            if not isinstance(claim, dict):
                raise EvidenceError(f"{tweet_id}: each claim must be an object.")
            for field in ("claim_id", "tweet_id", "dialect_label", "source_quote",
                          "claim", "target", "stance"):
                _text(claim.get(field), f"{tweet_id}: claim {field}")
            if claim["tweet_id"] != tweet_id:
                raise EvidenceError(f"{tweet_id}: claim references the wrong tweet.")
            if claim["dialect_label"] != tweet.dialect_label:
                raise EvidenceError(f"{tweet_id}: claim dialect mismatch.")
            if claim["claim_id"] in claim_ids:
                raise EvidenceError(f"Duplicate claim ID: {claim['claim_id']}")
            claim_ids.add(claim["claim_id"])
            if claim["stance"] not in STANCES:
                raise EvidenceError(f"{tweet_id}: invalid claim stance.")
            if find_source_quote(claim["source_quote"], tweet.text) is None:
                raise EvidenceError(f"{tweet_id}: source quotation not found in tweet.")
        records_by_id[tweet_id] = record

    missing = set(tweets_by_id) - set(records_by_id)
    if missing:
        raise EvidenceError(f"Missing extraction records: {', '.join(sorted(missing))}")
    return records_by_id


def populate_ontology(tweets: list[Tweet], extraction_records: list[dict], *,
                      ontology_file: Path = DEFAULT_ONTOLOGY,
                      run_id: str | None = None) -> Graph:
    """Return a fresh base + schema + evidence graph for exactly these tweets.

    The caller chooses the input batch; the CLI separately enforces 90 tweets.
    Run-specific IRIs isolate tweets/claims even if graphs are later combined.
    Original local IDs remain available as literals. No inputs are mutated.
    """
    records = _validate_records(tweets, extraction_records)
    run_id = new_run_id() if run_id is None else run_id
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise EvidenceError("Run ID must contain 1-128 letters, digits, dots, hyphens or underscores, starting with a letter or digit.")
    run = Namespace(f"http://www.semanticweb.org/group2/runs/{run_id}/")
    graph = Graph(identifier=run[""])
    graph.bind("", NS)
    graph.bind("run", run)
    # OntologyTools validates the namespace; only asserted base triples are copied.
    graph += OntologyTools(ontology_file).asserted
    graph.parse(EVIDENCE_SCHEMA, format="turtle")

    for tweet in tweets:
        record = records[tweet.id]
        tweet_iri = run["tweet/" + quote(tweet.id, safe="")]
        topic_iri = NS["Topic_" + tweet.topic]
        for triple in (
            (topic_iri, RDF.type, NS.Topic),
            (topic_iri, RDFS.label, Literal(tweet.topic)),
            (tweet_iri, RDF.type, NS.Tweet),
            (tweet_iri, NS.tweetId, Literal(tweet.id)),
            (tweet_iri, NS.tweetText, Literal(tweet.text)),
            (tweet_iri, NS.sourceFile, Literal(tweet.source_file)),
            (tweet_iri, NS.sourceLine, Literal(tweet.source_line)),
            (tweet_iri, NS.extractionStatus, Literal(record["status"])),
            (tweet_iri, NS.aboutTopic, topic_iri),
            (tweet_iri, NS.dialectLabel, Literal(tweet.dialect_label)),
        ):
            graph.add(triple)
        for claim in record["claims"]:
            claim_iri = run["claim/" + quote(claim["claim_id"], safe="")]
            graph.add((tweet_iri, NS.containsClaim, claim_iri))
            graph.add((claim_iri, RDF.type, NS.Claim))
            for prop, value in (
                (NS.claimId, claim["claim_id"]),
                (NS.hasText, claim["claim"]),
                (NS.sourceQuote, find_source_quote(claim["source_quote"], tweet.text)),
                (NS.targetText, claim["target"]),
                (NS.stance, claim["stance"]),
            ):
                graph.add((claim_iri, prop, Literal(value)))
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", choices=TOPICS, required=True)
    parser.add_argument("--claims", type=Path, help="Default: results/source_claims/<topic>.json")
    parser.add_argument("--ontology-file", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "ontology")
    parser.add_argument("--sample", action="store_true",
                        help="Explicitly allow only the tweets present in the extraction file.")
    args = parser.parse_args()
    claims_file = args.claims or ROOT / "results" / "source_claims" / f"{args.topic}.json"
    try:
        tweets = load_tweets(args.topic)
        validate_dataset(tweets)
        records = json.loads(claims_file.read_text(encoding="utf-8"))
        if args.sample:
            if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
                raise EvidenceError("Extraction records must be a list of objects.")
            ids = {_text(r.get("tweet_id"), "Extraction tweet ID") for r in records}
            selected = [tweet for tweet in tweets if tweet.id in ids]
        else:
            selected = tweets
        run_id = new_run_id()
        graph = populate_ontology(selected, records, ontology_file=args.ontology_file, run_id=run_id)
        destination = args.output_dir / run_id
        destination.mkdir(parents=True, exist_ok=False)
        graph_file = destination / f"{args.topic}.ttl"
        graph.serialize(destination=graph_file, format="turtle")
        manifest = {
            "run_id": run_id, "graph_iri": str(graph.identifier), "topic": args.topic,
            "sample": args.sample, "complete_topic": len(selected) == len(tweets),
            "tweet_count": len(selected), "expected_tweet_count": len(tweets),
            "claim_count": sum(len(r["claims"]) for r in records),
            "extraction_status_counts": {s: sum(r["status"] == s for r in records) for s in sorted(STATUSES)},
            "claims_file": str(claims_file.resolve()),
            "ontology_file": str(args.ontology_file.resolve()),
            "schema_file": str(EVIDENCE_SCHEMA),
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (EvidenceError, DatasetError, ValueError, OSError) as exc:
        parser.exit(1, f"Population error: {exc}\n")
    label = "Sample" if args.sample else "Complete topic"
    print(f"{label}: {len(selected)}/{len(tweets)} tweets, {manifest['claim_count']} claims.")
    print(f"Saved graph: {graph_file}\nSaved manifest: {destination / 'manifest.json'}")
    print("Extracted statements only; no truth assessments were created.")


if __name__ == "__main__":
    main()
