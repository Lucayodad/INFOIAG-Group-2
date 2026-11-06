"""Run the ontology queries of the report (Section 3.9, Table 8) with rdflib.

    python run_queries.py
        Q1-Q9, each on the asserted graph and on the limited OWL closure (inverse,
        symmetric, transitive and named subClassOf rules). Q7 and Q8 run on a copy
        of the ontology with the test assertions below; the saved file is never
        changed. Q1, Q4, Q7 and Q8 return nothing without the closure.

    python run_queries.py --graph results/agent_rerun/Obama_prompt1_seed101_artifacts/Obama.ttl --topic Obama
        source_evidence on a populated run graph: every tweet of the topic with its
        extracted claims. It reads the dialect labels, so it belongs to the
        evaluation; the agent never runs it.

The agent's own query, lookup_fact, lives in fact_tools.py. The queries are read from queries/.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from rdflib import RDF, Graph

from data_loader import TOPICS
from ontology_tools import NS, QUERY_DEFAULTS, TOPIC_IRIS, OntologyTools, limited_closure

ROOT = Path(__file__).resolve().parent
# Q7: a branch in a city in a country. Q8: the two advertisements of scenario S4,
# typed only as TVCommercial (Superbowl_060 and McDonalds_019).
TEST_ASSERTIONS = [
    (NS.branch1, NS.locatedIn, NS.Utrecht),
    (NS.Utrecht, NS.locatedIn, NS.Netherlands),
    (NS.Netherlands, RDF.type, NS.Country),
    (NS.TacoBellAd, RDF.type, NS.TVCommercial),
    (NS.TacoBellAd, NS.promotes, NS.TacoBellProduct),
    (NS.TacoBellAd, NS.createdBy, NS.TacoBell),
    (NS.TacoBellAd, NS.shownAt, NS.SuperBowl1),
    (NS.FishBurgerAd, RDF.type, NS.TVCommercial),
    (NS.FishBurgerAd, NS.promotes, NS.FishBurger),
    (NS.FishBurgerAd, NS.createdBy, NS.McDonalds),
]


def label(term):
    return term.get("label", term["value"])


def with_test_assertions():
    """A fresh copy of the ontology with TEST_ASSERTIONS, for Q7 and Q8."""
    tools = OntologyTools()
    for triple in TEST_ASSERTIONS:
        tools.asserted.add(triple)
    tools.inferred = limited_closure(tools.asserted)
    return tools


def ontology_queries():
    plain, examples = OntologyTools(), with_test_assertions()
    print(json.dumps(plain.statistics(), indent=2))
    for query_id in QUERY_DEFAULTS:
        tools = examples if query_id in ("Q7", "Q8") else plain
        for inference in (False, True):
            result = tools.run_query(query_id, use_inference=inference)
            rows = ["; ".join(f"{key}={label(term)}" for key, term in row.items()) for row in result["rows"]]
            print(f"{query_id} {result['reasoning']:13} {' | '.join(rows) or 'none'}")


def source_evidence(graph_file, topic):
    graph = Graph().parse(graph_file, format="turtle")
    query = (ROOT / "queries" / "source_evidence.rq").read_text(encoding="utf-8")
    rows = graph.query(query, initBindings={"topic": TOPIC_IRIS[topic]})
    tweets, claims = {}, set()
    for row in rows:
        tweets[str(row.tweet_id)] = str(row.dialect_label)
        if row.claim_id is not None:
            claims.add(str(row.claim_id))
    print(f"{topic}: {len(tweets)} tweets, {len(claims)} extracted claims; tweets per dialect label: "
          f"{dict(sorted(Counter(tweets.values()).items()))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", type=Path, help="A populated run graph (Turtle) for source_evidence")
    parser.add_argument("--topic", choices=TOPICS, help="The graph's topic, with --graph")
    args = parser.parse_args()
    if args.graph:
        if not args.topic:
            parser.error("--graph needs --topic")
        source_evidence(args.graph, args.topic)
    else:
        ontology_queries()


if __name__ == "__main__":
    main()
