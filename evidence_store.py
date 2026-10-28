
"""
Connect extracted tweet claims to existing ontology entities.

This module does NOT verify whether claims are true.
It creates an auditable evidence representation.
"""

import json
import re
from pathlib import Path

from rdflib import RDF, OWL

from data_loader import load_tweets
from ontology_tools import OntologyTools, NS


ROOT = Path(__file__).resolve().parent

INPUT_FILE = ROOT / "results/source_claims/Obama_sample.json"
OUTPUT_FILE = ROOT / "results/evidence/Obama_sample_linked.json"


def identify_entity(target):
    """
    Identify an existing ontology entity mentioned in a target.

    Only use explicit, conservative matching rules.
    """

    text = target.lower()

    # Michelle Obama is not the same entity as Barack Obama.
    if re.search(r"\bmichelle\s+obama\b", text):
        return None

    if re.search(r"\bobama\b", text):
        return "Obama"

    if re.search(r"\bmcdonald'?s\b", text):
        return "McDonalds"

    if re.search(r"\bsuper[\s-]?bowl\b", text):
        return "SuperBowl1"

    return None


def link_claim_to_ontology(claim, ontology):
    """
    Attach an ontology reference when a known entity is mentioned.
    """

    entity = identify_entity(claim["target"])

    if entity is None:
        link = {
            "status": "unmapped",
            "entity": None
        }

    elif (NS[entity], RDF.type, OWL.NamedIndividual) not in ontology.asserted:
        link = {
            "status": "unmapped",
            "entity": None
        }

    else:
        facts = ontology.get_entity_facts(entity)

        link = {
            "status": "entity_mentioned",
            "entity": entity,
            "uri": str(NS[entity]),
            "available_triples": len(facts["triples"])
        }

    return {
        **claim,
        "ontology_link": link,
        "semantic_status": "unverified"
    }


def build_evidence_store(records, tweets, ontology):
    """
    Validate source identities and enrich each extracted claim.
    """

    if not isinstance(records, list):
        raise ValueError("Expected a list of extraction records.")

    tweets_by_id = {tweet.id: tweet for tweet in tweets}
    processed_ids = set()
    evidence = []

    for record in records:
        tweet_id = record["tweet_id"]

        if tweet_id not in tweets_by_id:
            raise ValueError(f"Unknown tweet: {tweet_id}")

        if tweet_id in processed_ids:
            raise ValueError(f"Duplicate tweet result: {tweet_id}")

        processed_ids.add(tweet_id)

        original = tweets_by_id[tweet_id]

        if record["dialect_label"] != original.dialect_label:
            raise ValueError(f"Dialect mismatch: {tweet_id}")

        claims = record["claims"]

        if not isinstance(claims, list):
            raise ValueError(f"Invalid claims for {tweet_id}")

        linked_claims = []

        for claim in claims:
            if claim["tweet_id"] != tweet_id:
                raise ValueError("Claim references wrong tweet.")

            if claim["dialect_label"] != original.dialect_label:
                raise ValueError("Claim has incorrect dialect label.")

            linked_claims.append(
                link_claim_to_ontology(claim, ontology)
            )

        evidence.append({
            "tweet_id": tweet_id,
            "dialect_label": original.dialect_label,
            "original_text": original.text,
            "extraction_status": record["status"],
            "claims": linked_claims
        })

    missing = set(tweets_by_id) - processed_ids

    if missing:
        raise ValueError(f"Missing tweet results: {sorted(missing)}")

    return evidence


def main():
    # Load our previously saved extraction results.
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing extraction file: {INPUT_FILE}"
        )

    records = json.loads(
        INPUT_FILE.read_text(encoding="utf-8")
    )

    # Reuse the three original tweets.
    tweets = load_tweets("Obama")[:3]

    # Load the existing ontology without modifying it.
    ontology = OntologyTools()

    # Build the linked evidence representation.
    evidence = build_evidence_store(
        records, tweets, ontology
    )

    # Save the structured evidence.
    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    OUTPUT_FILE.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Display the result.
    for tweet in evidence:
        print(f"\n{tweet['tweet_id']}")

        for claim in tweet["claims"]:
            link = claim["ontology_link"]

            print(f"  Claim: {claim['claim']}")
            print(f"  Ontology: {link['entity']}")
            print(f"  Status: {link['status']}")

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
