"""Label-free viewpoint check: does the summary report the viewpoints the tweets hold?

The agent's reflection on its own bias (Lecture 5) without dialect labels: the
stances come from the claims extracted automatically from each tweet's text
(Tweet containsClaim Claim, stance). A viewpoint (positive or negative) held
by at least min_share of the topic's tweets must be reported by a summary
claim that is not CONTRADICTED. Under the open world assumption a sentence
whose claims could not be extracted may report it, so the viewpoint is then
UNKNOWN, never missing: uncertainty alone never triggers a revision.
"""

from ontology_tools import NS, TOPIC_IRIS


VIEWPOINTS = ("negative", "positive")

VIEWPOINT_NOT_RUN = {
    "status": "NOT_RUN", "min_share": None, "topic_tweet_count": 0, "tweets_by_stance": {},
    "share_by_stance": {}, "reported": {}, "supported": {}, "missing": [], "uncertain": [], "issues": [],
    "explanation": "No viewpoint check: the agent revises only contradicted claims.",
}

# Each tweet of the topic with the stances of its extracted claims (no label is read).
TOPIC_STANCES = """PREFIX : <http://www.semanticweb.org/group2/ontology#>
SELECT ?tweet_id ?stance WHERE {
    ?tweet a :Tweet ; :tweetId ?tweet_id ; :aboutTopic ?topic .
    OPTIONAL { ?tweet :containsClaim ?claim . ?claim :stance ?stance . }
}"""


def topic_stances(graph, topic):
    """{tweet_id: stances of its claims} for every tweet of the topic."""
    stances = {}
    for iri in sorted({TOPIC_IRIS[topic], NS["Topic_" + topic]}, key=str):
        for row in graph.query(TOPIC_STANCES, initBindings={"topic": iri}):
            found = stances.setdefault(row.tweet_id.toPython(), set())
            if row.stance is not None:
                found.add(str(row.stance))
    return stances


def check_viewpoints(graph, topic, summary_claims, factual_results, *, min_share=0.25):
    """FAIL when a viewpoint of at least min_share of the tweets is absent from the summary.

    A summary claim reports its stance unless it is CONTRADICTED; it supports
    the viewpoint only when SUPPORTED (the revision gate adds a viewpoint only
    then). Deterministic: no model is called.
    """
    if type(min_share) not in (int, float) or not 0 < min_share <= 1:
        raise ValueError("min_share must be a number in (0, 1].")
    stances = topic_stances(graph, topic)
    total = len(stances)
    by_stance = {s: sorted(t for t, found in stances.items() if s in found) for s in VIEWPOINTS}
    verdicts = {a["summary_claim_id"]: a["verdict"] for a in factual_results.get("assessments", [])}
    reported = {s: [] for s in VIEWPOINTS}
    supported = {s: [] for s in VIEWPOINTS}
    for record in summary_claims:
        for claim in record["claims"]:
            verdict = verdicts.get(claim["claim_id"])
            if claim["stance"] in VIEWPOINTS and verdict != "CONTRADICTED":
                reported[claim["stance"]].append(claim["claim_id"])
                if verdict == "SUPPORTED":
                    supported[claim["stance"]].append(claim["claim_id"])
    held = [s for s in VIEWPOINTS if total and len(by_stance[s]) / total >= min_share]
    absent = [s for s in held if not reported[s]]
    unclear = any(record["status"] == "unclear" for record in summary_claims)
    missing, uncertain = ([], absent) if unclear else (absent, [])
    issues = [{"type": "missing_viewpoint", "severity": "serious", "stance": s,
               "explanation": f"No summary claim reports the {s} viewpoint, although "
                              f"{len(by_stance[s])} of the topic's {total} tweets express it."} for s in missing]
    status = "FAIL" if missing else "UNKNOWN" if uncertain or not total else "PASS"
    return {"status": status, "min_share": min_share, "topic_tweet_count": total, "tweets_by_stance": by_stance,
            "share_by_stance": {s: round(len(by_stance[s]) / total, 4) if total else None for s in VIEWPOINTS},
            "reported": reported, "supported": supported, "missing": missing, "uncertain": uncertain,
            "issues": issues,
            "explanation": "Stances of the claims extracted from the tweets' text; no group annotation is read."}
