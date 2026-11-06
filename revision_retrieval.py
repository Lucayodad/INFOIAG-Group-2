"""Bounded local embedding retrieval of original, group-specific revision evidence."""

import json
import math

from data_loader import DIALECT_LABELS
from fact_tools import lookup_fact
from ontology_tools import NS, TOPIC_IRIS
from representation_tools import get_group_claims
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


def viewpoint_task(summary_claims, factual_results, viewpoint_results):
    """A task adding the most widely held missing viewpoint to one sentence, or None.

    The rewritten sentence is a no-claim sentence if any, else the one with the
    fewest SUPPORTED claims and most issues (the last on ties), never the only
    sentence reporting another viewpoint. Its own cited tweets are kept so its
    grounded content can stay; tweets expressing the viewpoint are added.
    """
    stance = max(viewpoint_results["missing"], key=lambda s: (len(viewpoint_results["tweets_by_stance"][s]), s))
    sentence_of = {c["claim_id"]: r["sentence_id"] for r in summary_claims for c in r["claims"]}
    carriers = [{sentence_of[i] for i in ids} for ids in viewpoint_results["reported"].values()]
    sole = {next(iter(sentences)) for sentences in carriers if len(sentences) == 1}
    supported = {r["sentence_id"]: 0 for r in summary_claims}
    issues = dict(supported)
    for assessment in factual_results["assessments"]:
        if assessment["verdict"] == "SUPPORTED" and assessment["sentence_id"] in supported:
            supported[assessment["sentence_id"]] += 1
    for issue in factual_results["issues"]:
        if issue["sentence_id"] in issues:
            issues[issue["sentence_id"]] += 1
    candidates = [r for r in summary_claims if r["status"] in ("extracted", "no_claim") and r["sentence_id"] not in sole]
    if not candidates:
        return None
    target = min(candidates, key=lambda r: (r["status"] != "no_claim", supported[r["sentence_id"]],
                                            -issues[r["sentence_id"]], -int(r["sentence_id"][1:])))
    count, total = len(viewpoint_results["tweets_by_stance"][stance]), viewpoint_results["topic_tweet_count"]
    return {"task_id": f"viewpoint:{stance}", "kind": "viewpoint", "sentence_id": target["sentence_id"],
            "query": target["sentence"], "stance": stance,
            "preferred_ids": list(viewpoint_results["tweets_by_stance"][stance]),
            "keep_ids": sorted({i for c in target["claims"] for i in c["cited_source_ids"]}),
            "explanation": f"No sentence reports the {stance} viewpoint that {count} of the topic's {total} "
                           f"tweets express. Rewrite this sentence so that it also reports that {stance} "
                           "viewpoint, citing a tweet that expresses it, and keep the sentence's own grounded content."}


def proportion_task(summary_claims, factual_results, proportion_results):
    """A task adding the under-represented viewpoint named by the proportion feedback, or None.

    The rewritten sentence is the one the feedback chose when it was extracted
    or has no claim; otherwise the viewpoint task's rule picks it (a no-claim
    sentence, else the fewest SUPPORTED claims and most issues, the last on ties).
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


def retrieve_revision_evidence(graph, topic, summary_claims, factual_results, representation_results,
                               embedder, *, top_k=3, max_tweets=9, max_evidence_bytes=7000,
                               embedding_cache=None, embedding_model="all-minilm", viewpoint_results=None,
                               proportion_results=None):
    """Retrieve for confirmed issues only; similarity is relevance, not entailment.

    Exact cited evidence is prioritized for factual repairs. Representation
    candidates are restricted to supporters of confirmed omitted claims in the
    affected group. A missing viewpoint is repaired only once no contradiction
    is left: the rewritten sentence keeps its own cited tweets and gains the
    top_k most similar tweets expressing the viewpoint. An under-represented
    viewpoint named by the proportion feedback is repaired likewise, also only
    once no contradiction is left: the topic's top_k tweets closest to the
    viewpoint come first, then the sentence's own. Stable cosine/ID
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
    factual_targets = sorted({t["sentence_id"] for t in tasks})
    representation = representation_results
    if representation["status"] == "FAIL":
        if representation.get("metric") == "mnli_equal_coverage":
            labels = sorted({issue["dialect_label"] for issue in representation["representation_issues"]
                             if issue["severity"] == "serious" and "dialect_label" in issue})
            groups = {label: get_group_claims(graph, topic, label) for label in DIALECT_LABELS}
            for label in labels:
                preferred_ids = [tweet["tweet_id"] for tweet in groups[label]["tweets"]]
                target = next((row for row in summary_claims if row["status"] == "no_claim"), summary_claims[-1])
                tasks.append({
                    "task_id": f"representation:{label}:mnli",
                    "kind": "representation",
                    "sentence_id": target["sentence_id"],
                    # Rank the group's tweets by the sentence being rewritten:
                    # a group-generic query returned tweets on unrelated topics,
                    # which the reviser then cited for the old sentence's content.
                    "query": target["sentence"],
                    "dialect_label": label,
                    "preferred_ids": preferred_ids,
                    "explanation": "Improve the group's MNLI Equal Coverage.",
                })
            # The generic ontology-claim branch below cannot select evidence
            # for a metric that intentionally has no claim clusters.
            if tasks:
                representation = {**representation, "status": "FAIL", "omitted_claims": []}
        labels = sorted({issue["dialect_label"] for issue in representation["representation_issues"]
                         if issue["severity"] == "serious" and "dialect_label" in issue})
        if not labels:
            coverage = representation["coverage_by_group"]
            low = min((row["lower"] for row in coverage.values() if row["lower"] is not None), default=None)
            labels = [g for g in DIALECT_LABELS if low is not None and coverage[g]["lower"] == low]
        for label in labels:
            omitted = [c for c in representation["omitted_claims"]
                       if c["status"] == "NOT_MATCHED" and c["supporting_tweet_ids"][label]]
            if not omitted:
                continue
            def gain(cluster):
                weights = {g: len(cluster["supporting_tweet_ids"][g]) /
                           max(1, representation["coverage_by_group"][g]["total_weight"])
                           for g in DIALECT_LABELS}
                return weights[label] - max(weights[g] for g in DIALECT_LABELS if g != label)
            # Prefer the omission that helps this group's normalized coverage
            # most relative to other groups, rather than an equally shared claim
            # whose inclusion leaves the measured disparity unchanged.
            cluster = min(omitted, key=lambda c: (-gain(c), -len(c["supporting_tweet_ids"][label]), c["canonical_claim_id"]))
            member = min((m for m in cluster["members"] if m["dialect_label"] == label), key=lambda m: m["claim_id"])
            # Reuse a sentence already needing correction; otherwise use the last
            # no-claim sentence, then the last sentence. Full rechecking guards
            # against losing an already represented viewpoint.
            vacant = [r["sentence_id"] for r in summary_claims if r["status"] == "no_claim"]
            target = factual_targets[0] if factual_targets else (vacant or [r["sentence_id"] for r in summary_claims])[-1]
            tasks.append({"task_id": f"representation:{label}:{cluster['canonical_claim_id']}",
                          "kind": "representation", "sentence_id": target, "query": member["claim"],
                          "dialect_label": label, "explanation": "Represent this omitted source viewpoint.",
                          "preferred_ids": sorted(cluster["supporting_tweet_ids"][label])})
    if not tasks and viewpoint_results is not None and viewpoint_results["status"] == "FAIL":
        task = viewpoint_task(summary_claims, factual_results, viewpoint_results)
        if task is not None:
            tasks.append(task)
    if not tasks and proportion_results is not None and proportion_results["status"] == "FAIL":
        task = proportion_task(summary_claims, factual_results, proportion_results)
        if task is not None:
            tasks.append(task)
    result = {"status": "insufficient", "reason": "no_actionable_source_evidence", "tasks": tasks,
              "tweets": [], "embedding_model": embedding_model, "embedding_call_count": 0,
              "tool_call_count": tools, "evidence_bytes": 0}
    if not tasks or representation["status"] == "FAIL" and not any(t["kind"] == "representation" for t in tasks):
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
            # A (legacy) group task's preferred IDs already are that group's tweets;
            # a viewpoint task ranks its sentence's own tweets first.
            first = set(task.get("keep_ids", task["preferred_ids"]))
            allowed = [tweet for tweet in corpus if task["kind"] in ("factual", "proportion")
                       or tweet["tweet_id"] in task["preferred_ids"] or tweet["tweet_id"] in first]
            score = {tweet["tweet_id"]: cosine(cache[(embedding_model, task["query"])],
                                               cache[(embedding_model, tweet["text"])]) for tweet in allowed}
            kept = set()
            if task["kind"] == "proportion":
                # The topic's tweets closest to the viewpoint come first (wherever they are
                # in the list; as in closest_topic_tweets), then the sentence's own.
                closest = sorted((-score[key], key) for key in score)
                task["preferred_ids"] = [key for _, key in closest[:top_k]]
                kept, first = first - set(task["preferred_ids"]), set(task["preferred_ids"])
            ranking = []
            for tweet in allowed:
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
                     len(first) + top_k if task["kind"] == "viewpoint" else
                     len(first) + len(kept) if task["kind"] == "proportion" else top_k)
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
            if task["kind"] in ("viewpoint", "proportion"):
                # The tweets expressing the missing viewpoint, which the replacement must cite.
                task[task["kind"] + "_ids"] = [key for key in task["evidence_ids"] if key in set(task["preferred_ids"])]
            if not task["evidence_ids"] or task["kind"] in ("viewpoint", "proportion") \
                    and not task[task["kind"] + "_ids"]:
                result["reason"] = "evidence_missing_or_budget_exhausted"
                return result
        result.update(status="ready", reason="retrieved", tweets=list(chosen.values()),
                      evidence_bytes=len(json.dumps(list(chosen.values()), ensure_ascii=False).encode("utf-8")))
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        result.update(reason="embedding_unavailable_or_invalid", error=str(exc))
    return result
