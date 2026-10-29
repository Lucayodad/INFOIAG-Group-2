"""Read source-claim evidence through the existing RDF vocabulary and query."""

from pathlib import Path

from rdflib import Graph, Literal, URIRef

from data_loader import DIALECT_LABELS, TOPICS
from ontology_tools import NS


QUERY_FILE = Path(__file__).resolve().parent / "queries" / "source_evidence.rq"


def get_group_claims(graph, topic, dialect_label):
    """Return this topic/group's tweets and per-tweet claims, including claimless tweets.

    Labels are existing literal dialectLabel values. No ontology changes or
    demographic inference are performed. Integrity problems are returned for
    the representation checker to mark UNKNOWN.
    """
    if not isinstance(graph, Graph):
        raise TypeError("graph must be an rdflib.Graph.")
    if topic not in TOPICS or dialect_label not in DIALECT_LABELS:
        raise ValueError("Use an existing dataset topic and dialect label.")
    rows = graph.query(QUERY_FILE.read_text(encoding="utf-8"), initBindings={
        "topic": NS["Topic_" + topic], "dialect_label": Literal(dialect_label),
    })
    tweets, claims, problems = {}, {}, []
    for row in rows:
        data = {name: value.toPython() for name, value in row.asdict().items()}
        uri = data["tweet"]
        if uri not in tweets:
            source_file = graph.value(row.tweet, NS.sourceFile)
            source_line = graph.value(row.tweet, NS.sourceLine)
            tweets[uri] = {
                "tweet_id": data["tweet_id"], "tweet_uri": uri, "text": data["text"],
                "dialect_label": data["dialect_label"], "extraction_status": data["status"],
                "source_file": source_file.toPython() if source_file is not None else None,
                "source_line": source_line.toPython() if source_line is not None else None,
                "source_claim_ids": [],
            }
        if data.get("claim") is not None:
            claim = {
                "claim_id": data["claim_id"], "claim_uri": data["claim"],
                "claim": data["claim_text"], "source_quote": data["quote"],
                "target": data["target"], "stance": data["stance"],
                **{key: tweets[uri][key] for key in ("tweet_id", "tweet_uri", "dialect_label", "source_file", "source_line")},
            }
            if claim["source_quote"] not in tweets[uri]["text"]:
                problems.append(f"Quotation absent from source tweet {claim['tweet_id']}.")
            support_key = (data["claim"], uri)
            if support_key in claims and claims[support_key] != claim:
                problems.append(f"Ambiguous source claim {data['claim_id']}.")
            claims[support_key] = claim
            if data["claim_id"] not in tweets[uri]["source_claim_ids"]:
                tweets[uri]["source_claim_ids"].append(data["claim_id"])
    typed = set(graph.subjects(NS.aboutTopic, NS["Topic_" + topic]))
    expected = {uri for uri in typed if (uri, NS.dialectLabel, Literal(dialect_label)) in graph}
    if expected != {URIRef(uri) for uri in tweets}:
        problems.append("Some topic/group tweets have incomplete RDF fields.")
    for uri, tweet in tweets.items():
        linked = set(graph.objects(URIRef(uri), NS.containsClaim))
        if len(linked) != len(tweet["source_claim_ids"]):
            problems.append(f"Incomplete claim fields for {tweet['tweet_id']}.")
        if (tweet["extraction_status"] == "extracted") != bool(tweet["source_claim_ids"]):
            problems.append(f"Inconsistent extraction status for {tweet['tweet_id']}.")
        if tweet["extraction_status"] not in {"extracted", "unclear", "no_claim"}:
            problems.append(f"Invalid extraction status for {tweet['tweet_id']}.")
    if len({tweet["tweet_id"] for tweet in tweets.values()}) != len(tweets):
        problems.append("Duplicate tweet IDs across RDF individuals; use one run's graph.")
    identities = {}
    for claim in claims.values():
        identities.setdefault(claim["claim_id"], set()).add(claim["claim_uri"])
    if any(len(uris) != 1 for uris in identities.values()):
        problems.append("Duplicate source claim IDs across RDF individuals.")
    return {
        "topic": topic, "dialect_label": dialect_label, "tweet_count": len(tweets),
        "tweets": list(tweets.values()), "claims": list(claims.values()),
        "integrity_issues": sorted(set(problems)),
    }
