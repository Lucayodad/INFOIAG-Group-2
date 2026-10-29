"""Retrieve cited RDF evidence without judging a summary claim's truth."""

from rdflib import Graph, Literal
from rdflib.plugins.sparql import prepareQuery

from ontology_tools import EVIDENCE_NOTE, NS, OntologyTools, encode_term


def lookup_fact(graph: Graph, summary_claim: dict) -> dict:
    """Return cited tweets, source claims, quotations and available provenance.

    Accepts one claim dictionary emitted by summary_claims.extract_sentence.
    Status describes citation availability: complete, partial, not_found, or
    no_citations. It says nothing about semantic support or factual correctness.
    Duplicate citation IDs are retrieved once, in their first-occurrence order.
    Missing optional RDF values are returned as None. Neither input is mutated.
    """
    if not isinstance(graph, Graph):
        raise TypeError("graph must be an rdflib.Graph.")
    if not isinstance(summary_claim, dict):
        raise TypeError("summary_claim must be a dictionary.")
    for field in ("claim_id", "claim"):
        if not isinstance(summary_claim.get(field), str) or not summary_claim[field].strip():
            raise ValueError(f"summary_claim.{field} must be a nonempty string.")
    citations = summary_claim.get("cited_source_ids")
    if not isinstance(citations, list) or any(
        not isinstance(tweet_id, str) or not tweet_id.strip() for tweet_id in citations
    ):
        raise ValueError("summary_claim.cited_source_ids must be a list of nonempty strings.")
    cited_ids = list(dict.fromkeys(citations))
    result = {
        "summary_claim_id": summary_claim["claim_id"],
        "summary_claim_text": summary_claim["claim"],
        "cited_source_ids": cited_ids,
        "retrieval_status": "no_citations",
        "tweets": [],
        "missing_tweet_ids": [],
    }
    if not cited_ids:
        return result

    query = prepareQuery("""
        SELECT ?tweet ?tweet_id ?tweet_text ?source_file ?source_line
               ?topic ?dialect_label ?extraction_status ?source_claim
               ?source_claim_id ?source_claim_text ?source_quote ?target ?stance
        WHERE {
            ?tweet a :Tweet ; :tweetId ?tweet_id .
            OPTIONAL { ?tweet :tweetText ?tweet_text . }
            OPTIONAL { ?tweet :sourceFile ?source_file . }
            OPTIONAL { ?tweet :sourceLine ?source_line . }
            OPTIONAL { ?tweet :aboutTopic ?topic . }
            OPTIONAL { ?tweet :dialectLabel ?dialect_label . }
            OPTIONAL { ?tweet :extractionStatus ?extraction_status . }
            OPTIONAL {
                ?tweet :containsClaim ?source_claim .
                OPTIONAL { ?source_claim :claimId ?source_claim_id . }
                OPTIONAL { ?source_claim :hasText ?source_claim_text . }
                OPTIONAL { ?source_claim :sourceQuote ?source_quote . }
                OPTIONAL { ?source_claim :targetText ?target . }
                OPTIONAL { ?source_claim :stance ?stance . }
            }
        }
        ORDER BY ?tweet ?source_claim
    """, initNs={"": NS})
    tweets = {}
    for tweet_id in cited_ids:
        rows = list(graph.query(query, initBindings={"tweet_id": Literal(tweet_id)}))
        if not rows:
            result["missing_tweet_ids"].append(tweet_id)
            continue
        for row in rows:
            data = {name: value.toPython() for name, value in row.asdict().items()}
            tweet_uri = data["tweet"]
            if tweet_uri not in tweets:
                tweets[tweet_uri] = {
                    "tweet_id": data["tweet_id"],
                    "tweet_uri": tweet_uri,
                    "text": data.get("tweet_text"),
                    "source_file": data.get("source_file"),
                    "source_line": data.get("source_line"),
                    "topic_uri": data.get("topic"),
                    "dialect_label": data.get("dialect_label"),
                    "extraction_status": data.get("extraction_status"),
                    "source_claims": [],
                }
            if data.get("source_claim") is not None:
                source_claim = {
                    "claim_id": data.get("source_claim_id"),
                    "claim_uri": data["source_claim"],
                    "text": data.get("source_claim_text"),
                    "source_quote": data.get("source_quote"),
                    "target": data.get("target"),
                    "stance": data.get("stance"),
                }
                if source_claim not in tweets[tweet_uri]["source_claims"]:
                    tweets[tweet_uri]["source_claims"].append(source_claim)
    result["tweets"] = list(tweets.values())
    missing = result["missing_tweet_ids"]
    result["retrieval_status"] = (
        "complete" if not missing else
        "not_found" if len(missing) == len(cited_ids) else "partial"
    )
    return result


def lookup_entity_facts(graph: Graph, entity: str) -> dict:
    """Retrieve subject/object triples for an explicit ontology name or full IRI.

    Pass an asserted graph, such as OntologyTools.asserted or populate_ontology's
    output, not OntologyTools.inferred. A plain Graph does not preserve which
    triples were inferred if a caller has already materialized them into it.
    No file is loaded, no inference is performed, and the graph is not changed.
    Source filenames are unavailable in the plain graph and are not guessed.
    """
    if not isinstance(graph, Graph):
        raise TypeError("graph must be an rdflib.Graph of asserted triples.")
    iri = OntologyTools.entity_iri(entity)
    triples = set(graph.triples((iri, None, None)))
    triples.update(graph.triples((None, None, iri)))
    records = []
    for subject, predicate, obj in sorted(triples, key=lambda triple: tuple(map(str, triple))):
        directions = []
        if subject == iri:
            directions.append("outgoing")
        if obj == iri:
            directions.append("incoming")
        records.append({
            "subject": encode_term(subject),
            "predicate": encode_term(predicate),
            "object": encode_term(obj),
            "directions": directions,
            "provenance": {
                "origin": "asserted",
                "graph_identifier": encode_term(graph.identifier),
                "source_file": None,
            },
        })
    return {
        "entity": str(iri),
        "status": "matches" if records else "unknown_entity",
        "scope": "asserted_only",
        "triples": records,
        "evidence_note": EVIDENCE_NOTE,
        "provenance_note": (
            "Origin describes membership in the supplied asserted graph. "
            "A plain Graph does not retain per-triple source files or inference history."
        ),
    }
