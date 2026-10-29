
"""Extract claims from an AI-generated five-sentence summary."""

import argparse
import json
from pathlib import Path

from ollama_backend import OllamaClient
from source_claims import CLAIM_SCHEMA, find_source_quote


INSTRUCTIONS = """
Extract individual claims from one sentence of an AI-generated summary.

Split distinct factual assertions and viewpoints into separate claims.
Preserve the meaning, negation, uncertainty and attribution.

For every claim return:
- quote: a quotation from the summary sentence
- claim: one atomic statement supported by that quotation
- target: the subject of the claim
- stance: positive, negative, neutral or unclear

Do not introduce new information.
Do not determine whether the summary is factually correct.
Do not use outside knowledge.

Return status "extracted" with claims when possible.
Return status "unclear" or "no_claim" with an empty list otherwise.
"""


def extract_sentence(sentence, number, client):
    """Extract claims from one summary sentence."""

    text = sentence["text"]
    source_ids = sentence["source_ids"]

    request = {
        "model": "qwen3.5:4b",
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": text}
        ],
        "format": CLAIM_SCHEMA,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": 900
        }
    }

    response = client.chat(request)
    raw = response["output"][0]["content"][0]["text"]
    data = json.loads(raw)

    if not isinstance(data, dict) or set(data) != {"status", "claims"}:
        raise ValueError(f"Sentence {number}: invalid response structure")

    status = data["status"]
    entries = data["claims"]

    if status not in {"extracted", "unclear", "no_claim"}:
        raise ValueError(f"Sentence {number}: invalid status")

    if not isinstance(entries, list):
        raise ValueError(f"Sentence {number}: claims must be a list")

    if (status == "extracted") != bool(entries):
        raise ValueError(f"Sentence {number}: inconsistent status")

    claims = []

    for index, entry in enumerate(entries, 1):
        required = {"quote", "claim", "target", "stance"}

        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError(f"Sentence {number}: invalid claim fields")

        if any(
            not isinstance(entry[key], str) or not entry[key].strip()
            for key in required
        ):
            raise ValueError(f"Sentence {number}: empty claim field")

        if entry["stance"] not in {
            "positive", "negative", "neutral", "unclear"
        }:
            raise ValueError(f"Sentence {number}: invalid stance")

        quote = find_source_quote(entry["quote"], text)

        if quote is None:
            raise ValueError(
                f"Sentence {number}: quotation not found in summary"
            )

        claims.append({
            "claim_id": f"S{number:02d}_C{index:02d}",
            "summary_quote": quote,
            "claim": entry["claim"],
            "target": entry["target"],
            "stance": entry["stance"],
            "cited_source_ids": source_ids,
            "verification_status": "unverified"
        })

    return {
        "sentence_id": f"S{number:02d}",
        "sentence": text,
        "status": status,
        "claims": claims
    }


def flatten_summary_claims(records):
    """Validate the existing five-sentence extraction format and attach sentence IDs.

    Returns copies; preserves the input. Unclear sentences remain visible in the
    original records for the checking components to flag as unresolved.
    """
    if not isinstance(records, list) or len(records) != 5:
        raise ValueError("Expected extraction records for exactly five summary sentences.")
    sentences, claim_ids, claims = set(), set(), []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each sentence extraction must be an object.")
        for field in ("sentence_id", "sentence", "status"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Missing or invalid sentence {field}.")
        if record["sentence_id"] in sentences:
            raise ValueError("Duplicate summary sentence ID.")
        sentences.add(record["sentence_id"])
        entries = record.get("claims")
        if record["status"] not in {"extracted", "unclear", "no_claim"} or not isinstance(entries, list):
            raise ValueError("Invalid summary extraction status or claims list.")
        if (record["status"] == "extracted") != bool(entries):
            raise ValueError("Inconsistent summary extraction status and claims.")
        for claim in entries:
            if not isinstance(claim, dict):
                raise ValueError("Each summary claim must be an object.")
            for field in ("claim_id", "claim", "summary_quote"):
                if not isinstance(claim.get(field), str) or not claim[field].strip():
                    raise ValueError(f"Missing or invalid summary claim {field}.")
            if claim["claim_id"] in claim_ids:
                raise ValueError("Duplicate summary claim ID.")
            claim_ids.add(claim["claim_id"])
            citations = claim.get("cited_source_ids")
            if not isinstance(citations, list) or any(not isinstance(x, str) or not x.strip() for x in citations):
                raise ValueError("Invalid summary claim citations.")
            if find_source_quote(claim["summary_quote"], record["sentence"]) is None:
                raise ValueError("Summary quotation is absent from its sentence.")
            claims.append({**claim, "sentence_id": record["sentence_id"]})
    return claims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("draft", type=Path)
    args = parser.parse_args()

    # Load the summary generated earlier.
    draft = json.loads(
        args.draft.read_text(encoding="utf-8")
    )

    sentences = draft["sentences"]

    if not isinstance(sentences, list) or len(sentences) != 5:
        raise ValueError("Expected a five-sentence summary")

    client = OllamaClient()
    results = []

    try:
        for number, sentence in enumerate(sentences, 1):
            print(f"Extracting claims from sentence {number}...", flush=True)

            result = extract_sentence(sentence, number, client)
            results.append(result)

            print(f"  Found {len(result['claims'])} claims")
    finally:
        client.close()

    # Save the extracted summary claims.
    output = (
        Path("results/summary_claims")
        / f"{args.draft.stem}_claims.json"
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
