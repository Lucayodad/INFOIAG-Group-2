"""Bounded local embedding retrieval of original tweets as revision evidence, label-free."""

import json
import math

from fact_tools import lookup_fact
from ontology_tools import NS, TOPIC_IRIS
from summary_claims import flatten_summary_claims


# The topic's original tweets, without dialect labels: the agent never reads
# them; they are joined back only by the final evaluation.
TOPIC_TWEETS = """PREFIX : <http://www.semanticweb.org/group2/ontology#>
SELECT ?tweet ?tweet_id ?text WHERE {
    ?tweet a :Tweet ; :tweetId ?tweet_id ; :aboutTopic ?topic .
    ?tweet (:tweetHasText|:tweetText) ?text .
}"""


def topic_tweets(graph, topic):
    """Return (tweets, integrity problems) for one topic, read label-free."""
    tweets, problems = {}, []
    for iri in sorted({TOPIC_IRIS[topic], NS["Topic_" + topic]}, key=str):
        for row in graph.query(TOPIC_TWEETS, initBindings={"topic": iri}):
            source_file, source_line = graph.value(row.tweet, NS.sourceFile), graph.value(row.tweet, NS.sourceLine)
            tweet = {"tweet_id": row.tweet_id.toPython(), "text": row.text.toPython(),
                     "source_file": source_file.toPython() if source_file is not None else None,
                     "source_line": source_line.toPython() if source_line is not None else None}
            if tweets.get(str(row.tweet), tweet) != tweet:
                problems.append(f"Conflicting tweet text values for {tweet['tweet_id']}.")
            tweets[str(row.tweet)] = tweet
    return list(tweets.values()), sorted(set(problems))


def cosine(left, right):
    if not left or len(left) != len(right) or any(
        type(x) not in (int, float) or not math.isfinite(x) for x in [*left, *right]
    ):
        raise ValueError("Embeddings need matching, finite, nonempty dimensions.")
    a, b = math.hypot(*left), math.hypot(*right)
    if not a or not b or not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Zero embeddings cannot be ranked.")
    return sum((x / a) * (y / b) for x, y in zip(left, right))


def _store_embeddings(cache, model, texts, vectors):
    """Validate vectors before filling the cache, including dimensions across calls."""
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise ValueError("Embedding provider returned the wrong number of vectors.")
    existing = next(iter(cache.values()), None)
    reference = existing if existing is not None else vectors[0]
    for vector in vectors:
        cosine(reference, vector)
    cache.update({(model, text): vector for text, vector in zip(texts, vectors)})


def closest_topic_tweets(graph, topic, query, embedder, *, top_k=3, embedding_cache=None,
                         embedding_model="all-minilm"):
    """The topic's top_k tweets most similar to query, [(tweet_id, similarity)], and the embedding calls made.

    Wherever a tweet is in the list; stable cosine/ID order, the same ranking
    as a proportion repair's. ValueError if the tweets are missing or ambiguous.
    """
    corpus, problems = topic_tweets(graph, topic)
    if problems or not corpus or len({t["tweet_id"] for t in corpus}) != len(corpus):
        raise ValueError("The topic's original tweets are missing or ambiguous.")
    cache = {} if embedding_cache is None else embedding_cache
    missing = [text for text in sorted({t["text"] for t in corpus} | {query}) if (embedding_model, text) not in cache]
    if missing:
        _store_embeddings(cache, embedding_model, missing, embedder(missing))
    ranked = sorted((-cosine(cache[(embedding_model, query)], cache[(embedding_model, t["text"])]), t["tweet_id"])
                    for t in corpus)
    return [(key, -score) for score, key in ranked[:top_k]], int(bool(missing))


def proportion_task(summary_claims, factual_results, proportion_results):
    """A task adding the under-represented viewpoint named by the proportion feedback, or None.

    The rewritten sentence is the one the feedback chose when it was extracted
    or has no claim; otherwise a no-claim sentence, else the one with the
    fewest SUPPORTED claims and most issues (the last on ties).
    The sentence's own cited tweets stay available for its grounded content.
    The tweets expressing the viewpoint are found at retrieval (preferred_ids).
    """
    found = proportion_results["under_represented"]
    supported = {r["sentence_id"]: 0 for r in summary_claims}
    issues = dict(supported)
    for assessment in factual_results["assessments"]:
        if assessment["verdict"] == "SUPPORTED" and assessment["sentence_id"] in supported:
            supported[assessment["sentence_id"]] += 1
    for issue in factual_results["issues"]:
        if issue["sentence_id"] in issues:
            issues[issue["sentence_id"]] += 1
    candidates = [r for r in summary_claims if r["status"] in ("extracted", "no_claim")]
    if not candidates:
        return None
    chosen = [r for r in candidates if r["sentence_id"] == proportion_results["sentence_id"]]
    target = chosen[0] if chosen else min(candidates, key=lambda r: (
        r["status"] != "no_claim", supported[r["sentence_id"]], -issues[r["sentence_id"]], -int(r["sentence_id"][1:])))
    return {"task_id": "proportion", "kind": "proportion", "sentence_id": target["sentence_id"],
            "query": found["viewpoint"], "viewpoint": found["viewpoint"], "preferred_ids": [],
            "keep_ids": sorted({i for c in target["claims"] for i in c["cited_source_ids"]}),
            "explanation": f'Proportion feedback: the summary under-represents this opinion of the tweets: '
                           f'"{found["viewpoint"]}". Rewrite this sentence so that it also reports what the '
                           "retrieved tweets say about it, in your own words and only as far as they express it; "
                           "do not copy this description. Cite at least one of those tweets, and keep the "
                           "sentence's own grounded content where it fits."}


def retrieve_revision_evidence(graph, topic, summary_claims, factual_results, embedder, *, top_k=3,
                               max_tweets=9, max_evidence_bytes=7000, embedding_cache=None,
                               embedding_model="all-minilm", proportion_results=None):
    """Retrieve for confirmed issues only; similarity is relevance, not entailment.

    Exact cited evidence is prioritized for factual repairs. An
    under-represented viewpoint named by the proportion feedback is repaired
    only once no contradiction is left: the topic's top_k tweets closest to
    the viewpoint come first, then the sentence's own. Stable cosine/ID
    ordering and whole-tweet limits make this reproducible for fixed
    embeddings. No graph or source text is modified.
    """
    for name, n in (("top_k", top_k), ("max_tweets", max_tweets), ("max_evidence_bytes", max_evidence_bytes)):
        if type(n) is not int or n <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    claims = {c["claim_id"]: c for c in flatten_summary_claims(summary_claims)}
    tasks, direct, tools = [], {}, 0
    for issue in factual_results["issues"]:
        if issue["severity"] != "serious":
            continue
        claim = claims.get(issue["claim_id"])
        if claim is None:
            continue
        tools += 1
        cited = lookup_fact(graph, claim)
        # Prefer the original sources already inspected by the factual checker.
        direct[issue["claim_id"]] = {row["tweet_id"] for row in cited["tweets"] if row["text"]}
        tasks.append({"task_id": "fact:" + issue["claim_id"], "kind": "factual",
                      "sentence_id": issue["sentence_id"], "query": claim["claim"],
                      "explanation": issue["explanation"],
                      "preferred_ids": sorted(direct[issue["claim_id"]])})
    if not tasks and proportion_results is not None and proportion_results["status"] == "FAIL":
        task = proportion_task(summary_claims, factual_results, proportion_results)
        if task is not None:
            tasks.append(task)
    result = {"status": "insufficient", "reason": "no_actionable_source_evidence", "tasks": tasks,
              "tweets": [], "embedding_model": embedding_model, "embedding_call_count": 0,
              "tool_call_count": tools, "evidence_bytes": 0}
    if not tasks:
        return result
    corpus, problems = topic_tweets(graph, topic)
    result["tool_call_count"] += 1
    if problems:
        result["reason"] = "ambiguous_source_evidence"
        return result
    if len({t["tweet_id"] for t in corpus}) != len(corpus):
        result["reason"] = "ambiguous_source_identity"
        return result
    corpus = sorted(corpus, key=lambda t: t["tweet_id"])
    if not corpus:
        result["reason"] = "original_tweets_missing"
        return result
    cache = {} if embedding_cache is None else embedding_cache
    texts = sorted({t["text"] for t in corpus} | {t["query"] for t in tasks})
    missing = [text for text in texts if (embedding_model, text) not in cache]
    try:
        if missing:
            result["embedding_call_count"] += 1
            _store_embeddings(cache, embedding_model, missing, embedder(missing))
        chosen = {}
        for task in tasks:
            first = set(task.get("keep_ids", task["preferred_ids"]))
            score = {tweet["tweet_id"]: cosine(cache[(embedding_model, task["query"])],
                                               cache[(embedding_model, tweet["text"])]) for tweet in corpus}
            kept = set()
            if task["kind"] == "proportion":
                # The topic's tweets closest to the viewpoint come first (wherever they are
                # in the list; as in closest_topic_tweets), then the sentence's own.
                closest = sorted((-score[key], key) for key in score)
                task["preferred_ids"] = [key for _, key in closest[:top_k]]
                kept, first = first - set(task["preferred_ids"]), set(task["preferred_ids"])
            ranking = []
            for tweet in corpus:
                key = tweet["tweet_id"]
                rank = 0 if key in first else 1 if key in kept else 2
                if rank == 0 or task["kind"] == "proportion" and rank == 1 \
                        or task["kind"] != "proportion" and score[key] > 0:
                    ranking.append((rank, -score[key], key, tweet))
            task["evidence_ids"] = []
            task["similarities"] = {}
            # A factual repair keeps all of the sentence's own cited tweets (within the
            # tweet and byte limits): the claims it does not contest still need their support.
            limit = (max(top_k, len(task["preferred_ids"])) if task["kind"] == "factual" else
                     len(first) + len(kept))
            for _, negative_score, key, tweet in sorted(ranking, key=lambda item: item[:3]):
                if len(task["evidence_ids"]) >= limit:
                    break
                row = {"tweet_id": key, "quote": tweet["text"],
                       "source_file": tweet["source_file"], "source_line": tweet["source_line"]}
                candidate = {**chosen, key: row}
                size = len(json.dumps(list(candidate.values()), ensure_ascii=False).encode("utf-8"))
                if len(candidate) > max_tweets or size > max_evidence_bytes:
                    continue
                chosen = candidate
                task["evidence_ids"].append(key)
                task["similarities"][key] = round(-negative_score, 12)
            if task["kind"] == "proportion":
                # The tweets expressing the missing viewpoint, which the replacement must cite.
                task["proportion_ids"] = [key for key in task["evidence_ids"] if key in set(task["preferred_ids"])]
            if not task["evidence_ids"] or task["kind"] == "proportion" and not task["proportion_ids"]:
                result["reason"] = "evidence_missing_or_budget_exhausted"
                return result
        result.update(status="ready", reason="retrieved", tweets=list(chosen.values()),
                      evidence_bytes=len(json.dumps(list(chosen.values()), ensure_ascii=False).encode("utf-8")))
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        result.update(reason="embedding_unavailable_or_invalid", error=str(exc))
    return result
