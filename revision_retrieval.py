"""Bounded local embedding retrieval of original, group-specific revision evidence."""

import json
import math

from data_loader import DIALECT_LABELS
from fact_tools import lookup_fact
from representation_tools import get_group_claims
from summary_claims import flatten_summary_claims


def cosine(left, right):
    if not left or len(left) != len(right) or any(
        type(x) not in (int, float) or not math.isfinite(x) for x in [*left, *right]
    ):
        raise ValueError("Embeddings need matching, finite, nonempty dimensions.")
    a, b = math.hypot(*left), math.hypot(*right)
    if not a or not b or not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Zero embeddings cannot be ranked.")
    return sum((x / a) * (y / b) for x, y in zip(left, right))


def retrieve_revision_evidence(graph, topic, summary_claims, factual_results, representation_results,
                               embedder, *, top_k=3, max_tweets=9, max_evidence_bytes=7000,
                               embedding_cache=None, embedding_model="all-minilm"):
    """Retrieve for confirmed issues only; similarity is relevance, not entailment.

    Exact cited evidence is prioritized for factual repairs. Representation
    candidates are restricted to supporters of confirmed omitted claims in the
    affected group. Stable cosine/ID ordering and whole-tweet limits make this
    reproducible for fixed embeddings. No graph or source text is modified.
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
                      "failed_claim": claim["claim"],
                      "failure_explanation": issue["explanation"],
                      "explanation": issue["explanation"], "dialect_label": None,
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
                tasks.append({
                    "task_id": f"representation:{label}:mnli",
                    "kind": "representation",
                    "min_source_ids": 2,
                    "sentence_id": next(
                        (row["sentence_id"] for row in summary_claims if row["status"] == "no_claim"),
                        summary_claims[-1]["sentence_id"],
                    ),
                    "query": f"Represent viewpoints from the {label} group.",
                    "failed_claim": None,
                    "failure_explanation": "The group's coverage is below the configured representation threshold.",
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
                          "kind": "representation",
                          "min_source_ids": 2, "sentence_id": target, "query": member["claim"],
                          "failed_claim": member["claim"],
                          "failure_explanation": "This viewpoint is missing from the current summary.",
                          "dialect_label": label, "explanation": "Represent this omitted source viewpoint.",
                          "preferred_ids": sorted(cluster["supporting_tweet_ids"][label])})
    result = {"status": "insufficient", "reason": "no_actionable_source_evidence", "tasks": tasks,
              "tweets": [], "embedding_model": embedding_model, "embedding_call_count": 0,
              "tool_call_count": tools, "evidence_bytes": 0}
    if not tasks or representation["status"] == "FAIL" and not any(t["kind"] == "representation" for t in tasks):
        return result
    groups = {label: get_group_claims(graph, topic, label) for label in DIALECT_LABELS}
    result["tool_call_count"] += len(groups)
    if any(group["integrity_issues"] for group in groups.values()):
        result["reason"] = "ambiguous_source_evidence"
        return result
    corpus = [tweet for group in groups.values() for tweet in group["tweets"]]
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
            vectors = embedder(missing)
            if not isinstance(vectors, list) or len(vectors) != len(missing):
                raise ValueError("Embedding provider returned the wrong number of vectors.")
            # Validate before filling the cache, including dimensions across calls.
            existing = next(iter(cache.values()), None)
            reference = existing if existing is not None else vectors[0]
            for vector in vectors:
                cosine(reference, vector)
            cache.update({(embedding_model, text): vector for text, vector in zip(missing, vectors)})
        chosen = {}
        for task in tasks:
            allowed = [tweet for tweet in corpus if task["dialect_label"] is None or
                       tweet["dialect_label"] == task["dialect_label"]]
            ranking = []
            for tweet in allowed:
                score = cosine(cache[(embedding_model, task["query"])], cache[(embedding_model, tweet["text"])])
                preferred = tweet["tweet_id"] in task["preferred_ids"]
                if preferred or score > 0:
                    ranking.append((not preferred, -score, tweet["tweet_id"], tweet))
            task["evidence_ids"] = []
            task["similarities"] = {}
            for _, negative_score, key, tweet in sorted(ranking, key=lambda item: item[:3]):
                required = max(top_k, task.get("min_source_ids", 1))
                if len(task["evidence_ids"]) >= required:
                    break
                row = {"tweet_id": key, "dialect_label": tweet["dialect_label"], "quote": tweet["text"],
                       "source_file": tweet["source_file"], "source_line": tweet["source_line"]}
                candidate = {**chosen, key: row}
                size = len(json.dumps(list(candidate.values()), ensure_ascii=False).encode("utf-8"))
                if len(candidate) > max_tweets or size > max_evidence_bytes:
                    continue
                chosen = candidate
                task["evidence_ids"].append(key)
                task["similarities"][key] = round(-negative_score, 12)
            if not task["evidence_ids"]:
                result["reason"] = "evidence_missing_or_budget_exhausted"
                return result
        result.update(status="ready", reason="retrieved", tweets=list(chosen.values()),
                      evidence_bytes=len(json.dumps(list(chosen.values()), ensure_ascii=False).encode("utf-8")))
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        result.update(reason="embedding_unavailable_or_invalid", error=str(exc))
    return result
