"""Bounded evidence-guided revision with the existing factual/coverage checkers."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from live_feed import observe_node
from rdflib import Graph
from rdflib.graph import ReadOnlyGraphAggregate
from typing_extensions import TypedDict

from data_loader import DIALECT_LABELS, TOPICS
from equal_coverage import CachedScorer
from fact_check import CHECKER_VERSION, claim_cache_key, fact_check_summary
from ollama_backend import OllamaClient
from model_usage import recording_client
from ontology_tools import NS
from proportion_check import PROPORTION_NOT_RUN, check_proportion, quoting_claims, verify_feedback
from representation_check import representation_check
from revision_retrieval import closest_topic_tweets, retrieve_revision_evidence
from summary_revision import PROFILED_REVISION_INSTRUCTIONS, REVISION_INSTRUCTIONS, revise_summary
from summary_claims import extract_sentence, extract_sentence_with_retry, flatten_summary_claims
from summarizer import GenerationError, digest
from viewpoint_check import VIEWPOINT_NOT_RUN, check_viewpoints


DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "results" / "evidence" / "agent_runs"


class SummaryAgentState(TypedDict):
    topic: str
    prompt: str
    initial_summary: dict
    current_summary: dict
    summary_claims: list[dict]
    extraction_errors: list[dict]
    factual_results: dict
    factual_issues: list[dict]
    representation_results: dict
    representation_issues: list[dict]
    viewpoint_results: dict
    viewpoint_issues: list[dict]
    proportion_results: dict
    proportion_issues: list[dict]
    revision_count: int
    execution_history: list[dict]
    uncertainty_flags: list[str]
    route: str | None
    revision_evidence: dict
    retrieval_history: list[dict]
    revisions: list[dict]
    summary_versions: list[dict]
    iterations: list[dict]
    best_iteration: int
    routing_decisions: list[dict]
    call_counts: dict[str, int]
    status: str
    stop_reason: str | None


def _snapshot(state):
    """Compact transitions; full evidence/LLM traces stay in checking results."""
    return {
        "status": state["status"], "revision_count": state["revision_count"],
        "extracted_claim_count": sum(len(row["claims"]) for row in state["summary_claims"]),
        "factual_status": state["factual_results"].get("status"),
        "representation_status": state["representation_results"].get("status"),
        "route": state["route"], "stop_reason": state["stop_reason"],
        "retained_iteration": state["best_iteration"],
        "uncertainty_flags": state["uncertainty_flags"],
        "call_counts": dict(state["call_counts"]),
    }


def _update(state, node, **changes):
    event = {"node": node, "updated_fields": list(changes),
             "before": _snapshot(state), "after": _snapshot({**state, **changes})}
    return {**changes, "execution_history": state["execution_history"] + [event]}


CHECKED_FIELDS = ("current_summary", "summary_claims", "extraction_errors", "factual_results",
                  "factual_issues", "representation_results", "representation_issues",
                  "viewpoint_results", "viewpoint_issues", "proportion_results", "proportion_issues")

# Dialect labels are evaluation data: by default the agent has no group check,
# never reads a label and revises only on contradicted claims.
REPRESENTATION_NOT_RUN = {
    "status": "NOT_RUN", "metric": None, "data_complete": False, "disparity": None,
    "coverage_by_group": {}, "matching": {"clusters": []}, "representation_issues": [],
    "reasons": [], "uncertain_claims": [], "model_call_count": 0,
    "explanation": "Dialect labels are used only by the final evaluation, never by the agent.",
}


def _metrics(state):
    facts, rep = state["factual_results"], state["representation_results"]
    measured = (rep["status"] in ("PASS", "FAIL") and rep["data_complete"]
                and not rep["uncertain_claims"] and rep["disparity"] is not None
                and all(row["coverage"] is not None for row in rep["coverage_by_group"].values()))
    partition = sorted((sorted(c["source_claim_ids"]), sorted(
        (group, sorted(ids)) for group, ids in c["supporting_tweet_ids"].items()))
        for c in rep["matching"]["clusters"])
    # Recorded checks from before the viewpoint check have no viewpoint results.
    views = state.get("viewpoint_results") or VIEWPOINT_NOT_RUN
    proportion = state.get("proportion_results") or PROPORTION_NOT_RUN
    return {"serious_factual_count": sum(i["severity"] == "serious" for i in facts["issues"]),
            "unknown_factual_count": sum(i["severity"] == "unresolved" for i in facts["issues"]),
            "disparity": rep["disparity"] if measured else None,
            "partition": digest(partition), "representation_status": rep["status"],
            "coverage": {g: row["coverage"] for g, row in rep["coverage_by_group"].items()},
            "missing_viewpoints": list(views["missing"]),
            "reported_viewpoints": sorted(s for s, ids in views["reported"].items() if ids),
            "supported_viewpoints": sorted(s for s, ids in views["supported"].items() if ids),
            "proportion_status": proportion["status"]}


def _proportion_ids(attempt):
    """The tweets retrieved for the under-represented viewpoint in this revision attempt."""
    return sorted({key for task in (attempt.get("evidence") or {}).get("tasks", [])
                   if task["kind"] == "proportion" for key in task.get("proportion_ids", [])})


def _sentence_findings(state, sentence_ids):
    """SUPPORTED claims per sentence, and the factual issues raised in those sentences."""
    supported = dict.fromkeys(sentence_ids, 0)
    for assessment in state["factual_results"]["assessments"]:
        if assessment["sentence_id"] in supported and assessment["verdict"] == "SUPPORTED" \
                and assessment["evidence_complete"]:
            supported[assessment["sentence_id"]] += 1
    issues = [issue for issue in state["factual_results"]["issues"] if issue["sentence_id"] in sentence_ids]
    return supported, issues


def _worst_coverage(metrics):
    return None if metrics["disparity"] is None else min(metrics["coverage"].values())


def compare_revision(previous, current, changes, proportion_ids=()):
    """Accept a revision only if it is measurably better and nowhere worse.

    Changed sentences must be extracted, contain no CONTRADICTED claim, keep at
    least one SUPPORTED claim and raise no more issues than the sentences they
    replace: a leftover UNKNOWN phrase no longer voids a repair. The whole
    summary may gain no contradiction, and no issue overall: a contradiction
    that becomes UNKNOWN is progress, a SUPPORTED claim that does is not.
    Only the optional (legacy) group check measures coverage: there the
    worst-covered group's coverage may not fall either. With the viewpoint
    check no reported viewpoint may be lost, and a missing one counts as added
    only when a SUPPORTED claim reports it. With the proportion feedback, the
    under-represented viewpoint counts as added only when a SUPPORTED claim of
    a sentence whose text changed quotes one of proportion_ids, the tweets
    retrieved for it. Progress
    means fewer contradictions, an added viewpoint or a better-covered worst group.
    """
    targets = {row["sentence_id"] for row in changes}
    if any(record["sentence_id"] in targets and record["status"] != "extracted"
           for record in current["summary_claims"]):
        return False, "unsupported_revision"
    supported, issues_after = _sentence_findings(current, targets)
    _, issues_before = _sentence_findings(previous, targets)
    if any(issue["severity"] == "serious" for issue in issues_after) or not all(supported.values()) \
            or len(issues_after) > len(issues_before):
        return False, "unsupported_revision"
    before, after = _metrics(previous), _metrics(current)
    if after["serious_factual_count"] > before["serious_factual_count"] or \
            after["serious_factual_count"] + after["unknown_factual_count"] > \
            before["serious_factual_count"] + before["unknown_factual_count"]:
        return False, "factual_regression"
    if set(before["reported_viewpoints"]) - set(after["reported_viewpoints"]):
        return False, "viewpoint_regression"
    worst_before, worst_after = _worst_coverage(before), _worst_coverage(after)
    if worst_before is not None and (worst_after is None or before["partition"] != after["partition"]
                                     or worst_after < worst_before - 1e-12):
        return False, "representation_regression_or_incomparable"
    if before["representation_status"] != "FAIL" and after["representation_status"] == "FAIL":
        return False, "representation_regression_or_incomparable"
    added = set(before["missing_viewpoints"]) & set(after["supported_viewpoints"])
    named = set(proportion_ids) if before["proportion_status"] == "FAIL" else set()
    # Only a sentence whose text changed can add a viewpoint: citations alone are not read.
    rewritten = {row["sentence_id"] for row in changes
                 if "before" not in row or row["before"]["text"].strip() != row["after"]["text"].strip()}
    grounded = bool(named) and bool(quoting_claims(current["factual_results"], named, rewritten))
    improved = (after["serious_factual_count"] < before["serious_factual_count"] or bool(added) or grounded or
                worst_before is not None and worst_after > worst_before + 1e-12)
    if not improved and before["missing_viewpoints"]:
        # A failed viewpoint repair, retried with this finding.
        still = set(before["missing_viewpoints"]) - set(after["reported_viewpoints"])
        return False, "viewpoint_still_missing" if still else "viewpoint_not_supported"
    if not improved and named and not before["serious_factual_count"]:
        # A failed proportion repair (contradictions are repaired first), retried with this finding.
        return False, "proportion_not_supported"
    return (True, "measured_improvement") if improved else (False, "no_improvement")


def _counts(state, **increments):
    return {key: value + increments.get(key, 0) for key, value in state["call_counts"].items()}


def _extraction_key(number, sentence):
    return json.dumps([number, sentence["text"], sentence["source_ids"]], ensure_ascii=False)


def seed_check_cache(summary, checked, *, any_checker=False):
    """Caches holding a recorded first check of this same draft, to reuse.

    Extraction and fact checks run at temperature 0 and repeated identically in
    every recorded recheck. Sentences whose extraction failed are not seeded,
    so they get the current extraction retry; their claims are checked anew.
    A fact verdict is reused only if the current checker made it (its
    checker_version); older verdicts are checked anew. any_checker=True
    reuses them anyway, for replaying a historical run exactly.
    """
    if [record["sentence"] for record in checked["summary_claims"]] != [s["text"] for s in summary["sentences"]]:
        raise ValueError("The recorded check belongs to a different draft.")
    cache = {"extraction": {}, "facts": {}}
    for number, (sentence, record) in enumerate(zip(summary["sentences"], checked["summary_claims"]), 1):
        if record["status"] != "unclear":
            cache["extraction"][_extraction_key(number, sentence)] = ("ok", deepcopy(record))
    by_id = {a["summary_claim_id"]: a for a in checked["factual_results"]["assessments"]}
    for claim in flatten_summary_claims(checked["summary_claims"]):
        if claim["claim_id"] in by_id and (any_checker or
                                           by_id[claim["claim_id"]].get("checker_version") == CHECKER_VERSION):
            cache["facts"][claim_cache_key(claim)] = deepcopy(by_id[claim["claim_id"]])
    return cache


def build_summary_agent(graph, client, *, disparity_threshold=None,
                        allowed_entity_ids=None, max_tool_calls=3,
                        expected_group_counts=None, max_model_calls=64,
                        max_input_bytes=12000, batch_size=8, max_revisions=3,
                        embedder=None, embedding_model="all-minilm", revision_top_k=3,
                        max_revision_tweets=9, revision_evidence_bytes=7000,
                        revision_max_input_bytes=14000, mnli_scorer=None,
                        representation_metric=None, check_cache=None,
                        viewpoint_check=False, min_viewpoint_share=0.25,
                        proportion_check=False):
    """Compile the sequential graph with read-only evidence outside its state.

    Serious issues mean conclusive under the existing evidence guards, not
    human-verified truth. UNKNOWN alone never triggers revision. With
    representation_metric None (the default) there is no group check: dialect
    labels are evaluation-only. "mnli"/"internal" restore the earlier
    label-based group check and are kept for the recorded runs.
    viewpoint_check adds the label-free reflection on bias: a viewpoint held by
    at least min_viewpoint_share of the topic's tweets (from the stances of
    their extracted claims) that no summary claim reports is repaired once no
    contradiction is left, and the reviser gets a role profile.
    proportion_check adds one label-free feedback layer: Qwen judges whether
    the summary represents all the topic's tweets in proportion, and an opinion
    it finds under-represented is repaired the same way, from the topic's
    tweets closest to it. Unchanged
    sentences are not re-extracted or re-checked: their earlier results in this
    run (or a recorded first check from seed_check_cache) are reused.
    """
    if not isinstance(graph, Graph):
        raise TypeError("graph must be an rdflib.Graph.")
    if type(max_revisions) is not int or not 0 <= max_revisions <= 3:
        raise ValueError("max_revisions must be an integer from 0 to 3.")
    for value in (revision_top_k, max_revision_tweets, revision_evidence_bytes, revision_max_input_bytes):
        if type(value) is not int or value <= 0:
            raise ValueError("Revision retrieval limits must be positive integers.")
    for name, value, minimum in (("max_tool_calls", max_tool_calls, 0),
                                 ("max_model_calls", max_model_calls, 0),
                                 ("max_input_bytes", max_input_bytes, 1),
                                 ("batch_size", batch_size, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if representation_metric not in (None, "mnli", "internal"):
        raise ValueError("representation_metric must be None, 'mnli' or 'internal'.")
    if type(viewpoint_check) is not bool:
        raise ValueError("viewpoint_check must be True or False.")
    if type(min_viewpoint_share) not in (int, float) or not 0 < min_viewpoint_share <= 1:
        raise ValueError("min_viewpoint_share must be a number in (0, 1].")
    if type(proportion_check) is not bool:
        raise ValueError("proportion_check must be True or False.")
    if disparity_threshold is not None and (
        type(disparity_threshold) not in (int, float)
        or not math.isfinite(disparity_threshold) or not 0 <= disparity_threshold <= 1
    ):
        raise ValueError("disparity_threshold must be None or a finite number in [0, 1].")
    if expected_group_counts is not None and (
        set(expected_group_counts) != set(DIALECT_LABELS)
        or any(type(n) is not int or n <= 0 for n in expected_group_counts.values())
    ):
        raise ValueError("Expected counts must specify positive integers for AA, Hisp and White.")
    if representation_metric is None:
        # Without a group check the agent never holds a dialect label: its
        # read-only copy of the graph omits them (the caller's graph is unchanged).
        label_free = Graph()
        label_free += (triple for triple in graph if triple[1] != NS.dialectLabel)
        graph = label_free
    evidence = ReadOnlyGraphAggregate([graph])
    cache = {"extraction": {}, "facts": {}} if check_cache is None else check_cache
    cache.setdefault("extraction", {})
    cache.setdefault("facts", {})
    scorer = CachedScorer(mnli_scorer)
    embedding_cache, proportion_cache = {}, {}
    embedding_function = embedder if embedder is not None else lambda texts: client.embed(texts, model=embedding_model)

    def extract_claims(state: SummaryAgentState):
        records, errors, calls = [], [], 0
        for number, sentence in enumerate(state["current_summary"]["sentences"], 1):
            key = _extraction_key(number, sentence)
            if key not in cache["extraction"]:
                try:
                    record, used = extract_sentence_with_retry(sentence, number, client, extract=extract_sentence)
                    cache["extraction"][key] = ("ok", record)
                    calls += used
                except (GenerationError, ValueError, KeyError, IndexError, TypeError) as exc:
                    cache["extraction"][key] = ("error", type(exc).__name__, str(exc))
                    calls += 2
            outcome = cache["extraction"][key]
            if outcome[0] == "ok":
                records.append(deepcopy(outcome[1]))
            else:
                # A failed extraction is missing knowledge, never a no-claim result.
                records.append({"sentence_id": f"S{number:02d}", "sentence": sentence["text"],
                                "status": "unclear", "claims": []})
                errors.append({"sentence_id": f"S{number:02d}", "error_type": outcome[1], "message": outcome[2]})
        flatten_summary_claims(records)
        return _update(state, "extract_summary_claims", summary_claims=records,
                       extraction_errors=errors, call_counts=_counts(state, model_calls=calls))

    def check_facts(state: SummaryAgentState):
        result = fact_check_summary(evidence, state["summary_claims"], client,
                                    allowed_entity_ids=allowed_entity_ids,
                                    max_tool_calls=max_tool_calls, cache=cache["facts"])
        return _update(state, "factual_check", factual_results=result,
                       factual_issues=result["issues"], call_counts=_counts(
                           state, model_calls=result["model_call_count"], tool_calls=result["tool_call_count"]))

    def check_representation(state: SummaryAgentState):
        result = representation_check(
            evidence, state["summary_claims"], client, state["topic"],
            disparity_threshold=disparity_threshold, expected_group_counts=expected_group_counts,
            max_model_calls=max_model_calls, max_input_bytes=max_input_bytes, batch_size=batch_size,
            embedder=embedding_function, embedding_model=embedding_model, embedding_cache=embedding_cache,
            metric=representation_metric, mnli_scorer=scorer)
        return _update(state, "representation_check", representation_results=result,
                       representation_issues=result["representation_issues"],
                       call_counts=_counts(state, model_calls=result["model_call_count"],
                                           embedding_calls=result.get("embedding_call_count", 0)))

    def check_views(state: SummaryAgentState):
        result = check_viewpoints(evidence, state["topic"], state["summary_claims"], state["factual_results"],
                                  min_share=min_viewpoint_share)
        return _update(state, "viewpoint_check", viewpoint_results=result, viewpoint_issues=result["issues"])

    def check_proportions(state: SummaryAgentState):
        result = check_proportion(evidence, state["topic"], state["current_summary"], client, cache=proportion_cache)
        calls = 0
        if result["status"] == "FAIL":
            # Qwen proposes, the verified evidence decides: an opinion whose closest
            # tweets a SUPPORTED claim already quotes is already reported.
            try:
                closest, calls = closest_topic_tweets(evidence, state["topic"], result["under_represented"]["viewpoint"],
                                                      embedding_function, top_k=revision_top_k,
                                                      embedding_cache=embedding_cache, embedding_model=embedding_model)
                result = verify_feedback(result, state["factual_results"], closest)
            except (ValueError, TypeError, RuntimeError, OSError) as exc:
                result = {**result, "status": "UNKNOWN", "reason": "embedding_unavailable_or_invalid", "issues": [],
                          "error": str(exc)}
        return _update(state, "proportion_check", proportion_results=result, proportion_issues=result["issues"],
                       call_counts=_counts(state, model_calls=result["model_call_count"], embedding_calls=calls))

    def route_checks(state: SummaryAgentState):
        checked = {key: deepcopy(state[key]) for key in CHECKED_FIELDS}
        iteration = {**checked, "metrics": _metrics(state), "revision_count": state["revision_count"],
                     "call_counts": dict(state["call_counts"]), "retained": True}
        iterations, revisions = deepcopy(state["iterations"]), deepcopy(state["revisions"])
        iterations.append(iteration)
        best = len(iterations) - 1
        if revisions and revisions[-1]["status"] == "candidate":
            accepted, reason = compare_revision(iterations[state["best_iteration"]], checked, revisions[-1]["changes"],
                                                _proportion_ids(revisions[-1]))
            revisions[-1].update(status="accepted" if accepted else "rolled_back", reason=reason,
                                  metrics_after=iteration["metrics"], checked_iteration=best)
            if not accepted:
                iterations[-1]["retained"] = False
                previous = iterations[state["best_iteration"]]
                restored = {key: deepcopy(previous[key]) for key in CHECKED_FIELDS}
                # Retry a bad fix (unsupported or regressing) with the recheck's
                # findings as feedback; a valid fix with no measured gain stops.
                retry = reason != "no_improvement" and state["revision_count"] < max_revisions
                route = "retrieve_revision_evidence" if retry else "finalize"
                decision = {"node": "route_checks", "iteration": best, "route": route, "reason": reason, "rolled_back": True}
                return _update(state, "route_checks", **restored, revisions=revisions, iterations=iterations,
                               route=route, stop_reason=reason,
                               routing_decisions=state["routing_decisions"] + [decision])
        serious = [issue for issue in state["factual_issues"] + state["representation_issues"]
                   + state["viewpoint_issues"] + state["proportion_issues"] if issue["severity"] == "serious"]
        flags = []
        if state["extraction_errors"]:
            flags.append("summary_extraction_failed")
        if state["factual_results"]["status"] == "UNKNOWN" or any(
            issue["severity"] == "unresolved" for issue in state["factual_issues"]
        ):
            flags.append("factual_uncertainty")
        representation = state["representation_results"]
        # A conclusive disparity bound can fail without identifying a single
        # group whose marginal interval is strictly below another's. Honor the
        # checker's FAIL even when its per-group issue list is empty.
        confirmed = bool(serious) or any(result["status"] == "FAIL" for result in
                                         (state["factual_results"], representation, state["viewpoint_results"],
                                          state["proportion_results"]))
        if representation["status"] == "UNKNOWN":
            flags.append("representation_uncertainty")
        if state["viewpoint_results"]["status"] == "UNKNOWN":
            flags.append("viewpoint_uncertainty")
        if state["proportion_results"]["status"] == "UNKNOWN":
            flags.append("proportion_uncertainty")
        flags.extend(representation["reasons"])
        if representation["uncertain_claims"]:
            flags.append("uncertain_representation_matches")
        reason = ("revision_limit" if confirmed and state["revision_count"] >= max_revisions else
                  "confirmed_serious_issues" if confirmed else
                  "uncertainty_only" if flags else "no_confirmed_serious_issues")
        route = "retrieve_revision_evidence" if confirmed and state["revision_count"] < max_revisions else "finalize"
        decision = {"node": "route_checks", "iteration": best, "route": route, "reason": reason, "rolled_back": False}
        return _update(state, "route_checks", route=route, uncertainty_flags=sorted(set(flags)),
                       stop_reason=reason, iterations=iterations, revisions=revisions, best_iteration=best,
                       routing_decisions=state["routing_decisions"] + [decision])

    def retrieve_evidence(state: SummaryAgentState):
        retrieved = retrieve_revision_evidence(
            evidence, state["topic"], state["summary_claims"], state["factual_results"],
            state["representation_results"], embedding_function, top_k=revision_top_k,
            max_tweets=max_revision_tweets, max_evidence_bytes=revision_evidence_bytes,
            embedding_cache=embedding_cache, embedding_model=embedding_model,
            viewpoint_results=state["viewpoint_results"], proportion_results=state["proportion_results"])
        ready = retrieved["status"] == "ready"
        decision = {"node": "retrieve_revision_evidence", "route": "revise_summary" if ready else "finalize",
                    "reason": retrieved["reason"]}
        return _update(state, "retrieve_revision_evidence", revision_evidence=retrieved,
                       retrieval_history=state["retrieval_history"] + [retrieved],
                       route="revise_summary" if ready else "finalize",
                       stop_reason="revision_evidence_ready" if ready else "insufficient_revision_evidence",
                       routing_decisions=state["routing_decisions"] + [decision],
                       call_counts=_counts(state, embedding_calls=retrieved["embedding_call_count"],
                                           tool_calls=retrieved["tool_call_count"]))

    def revise(state: SummaryAgentState):
        issues = [{key: issue[key] for key in ("type", "severity", "explanation")}
                  for issue in state["factual_issues"] + state["representation_issues"] + state["viewpoint_issues"]
                  + state["proportion_issues"] if issue["severity"] == "serious"]
        last = state["revisions"][-1] if state["revisions"] else None
        if last and last["status"] == "rolled_back":
            checked = state["iterations"][last["checked_iteration"]]["factual_results"]["assessments"]
            changed = {row["sentence_id"]: row["after"]["text"] for row in last["changes"]}
            findings = [f'{a["summary_claim_text"]}: {a["verdict"]}; {a["explanation"][:240]}'
                        for a in checked if a["sentence_id"] in changed and a["verdict"] != "SUPPORTED"]
            if last["reason"].startswith("representation") and representation_metric is not None:
                # Without numbers the reviser cannot tell what went wrong.
                before, after = last["metrics_before"]["coverage"], last["metrics_after"]["coverage"]
                findings.append("Group coverage before " + json.dumps({g: round(v, 4) for g, v in before.items()
                                                                        if v is not None}) +
                                ", after " + json.dumps({g: round(v, 4) for g, v in after.items() if v is not None}) +
                                ": the worst-covered group must gain coverage and no group may fall below it.")
            if last["reason"].startswith("viewpoint"):
                before, after = last["metrics_before"], last["metrics_after"]
                findings.append(f"Viewpoints reported before {before['reported_viewpoints']}, after "
                                f"{after['reported_viewpoints']} (SUPPORTED after: {after['supported_viewpoints']}): "
                                f"a missing viewpoint counts only when a SUPPORTED claim reports it, "
                                "and no reported viewpoint may be lost.")
            if last["reason"].startswith("proportion"):
                findings.append("The under-represented viewpoint counts only when a SUPPORTED claim of the "
                                "replacement quotes one of the tweets " + ", ".join(_proportion_ids(last)) + ".")
            issues.append({"type": "rejected_previous_revision", "severity": "serious", "explanation":
                           f"Rechecking rolled back your previous replacement ({last['reason']}): "
                           f"{json.dumps(changed, ensure_ascii=False)}. Findings: {' | '.join(findings)}"})
        elif last and last["status"] == "rejected" and last.get("error"):
            issues.append({"type": "rejected_previous_revision", "severity": "serious", "explanation":
                           f"Your previous proposal was invalid and was not checked: {last['error']}"})
        # At temperature 0 the seed is ignored, so a retry would repeat the
        # failed sentence; only retries sample, reproducibly per attempt.
        retry = bool(last and last["status"] in ("rolled_back", "rejected"))
        proposal = revise_summary(state["current_summary"], issues, state["revision_evidence"],
                                  client, max_input_bytes=revision_max_input_bytes,
                                  temperature=0.7 if retry else 0, seed=42 + state["revision_count"],
                                  instructions=PROFILED_REVISION_INSTRUCTIONS if viewpoint_check or proportion_check
                                  else REVISION_INSTRUCTIONS)
        attempt = {**proposal, "attempt": state["revision_count"] + 1,
                   "summary_before": deepcopy(state["current_summary"]), "evidence": state["revision_evidence"],
                   "metrics_before": _metrics(state)}
        candidate = proposal["status"] == "candidate"
        count = state["revision_count"] + bool(proposal["model_call_count"])
        # An invalid proposal (e.g. a pasted tweet after its one format repair)
        # is retried with its error, like a rolled-back one, while attempts remain.
        route = ("extract_summary_claims" if candidate else
                 "revise_summary" if proposal["reason"] == "invalid_revision" and proposal["model_call_count"]
                 and count < max_revisions else "finalize")
        decision = {"node": "revise_summary", "route": route, "reason": proposal["reason"]}
        changes = {"current_summary": proposal["candidate"],
                   "summary_versions": state["summary_versions"] + [deepcopy(proposal["candidate"])]} if candidate else {}
        return _update(state, "revise_summary", **changes,
                       revisions=state["revisions"] + [attempt],
                       # A format repair belongs to the same revision attempt.
                       revision_count=count,
                       route=route,
                       stop_reason="revision_requires_rechecking" if candidate else proposal["reason"],
                       routing_decisions=state["routing_decisions"] + [decision],
                       call_counts=_counts(state, model_calls=proposal["model_call_count"]))

    def finalize(state: SummaryAgentState):
        remaining = any(result["status"] == "FAIL" for result in
                        (state["factual_results"], state["representation_results"], state["viewpoint_results"],
                         state["proportion_results"]))
        status = ("stopped_with_issues" if remaining else
                  "completed_with_uncertainty" if state["uncertainty_flags"] else "completed")
        return _update(state, "finalize", status=status)

    builder = StateGraph(SummaryAgentState)
    nodes = (("extract_summary_claims", extract_claims), ("factual_check", check_facts),
             ("route_checks", route_checks), ("retrieve_revision_evidence", retrieve_evidence),
             ("revise_summary", revise), ("finalize", finalize))
    if representation_metric is not None:
        nodes += (("representation_check", check_representation),)
    if viewpoint_check:
        nodes += (("viewpoint_check", check_views),)
    if proportion_check:
        nodes += (("proportion_check", check_proportions),)
    for name, function in nodes:
        builder.add_node(name, observe_node(name, function, client))
    builder.add_edge(START, "extract_summary_claims")
    builder.add_edge("extract_summary_claims", "factual_check")
    checks = ["factual_check"] + ["representation_check"] * (representation_metric is not None) \
        + ["viewpoint_check"] * viewpoint_check + ["proportion_check"] * proportion_check + ["route_checks"]
    for left, right in zip(checks, checks[1:]):
        builder.add_edge(left, right)
    builder.add_conditional_edges("route_checks", lambda state: state["route"],
                                  {"finalize": "finalize", "retrieve_revision_evidence": "retrieve_revision_evidence"})
    builder.add_conditional_edges("retrieve_revision_evidence", lambda state: state["route"],
                                  {"finalize": "finalize", "revise_summary": "revise_summary"})
    builder.add_conditional_edges("revise_summary", lambda state: state["route"],
                                  {"finalize": "finalize", "extract_summary_claims": "extract_summary_claims",
                                   "revise_summary": "revise_summary"})
    builder.add_edge("finalize", END)
    return builder.compile()


def run_summary_agent(graph, initial_summary, *, topic, prompt, client=None, **check_options):
    """Check one existing five-sentence draft (the dict from generate_summary).

    The draft must contain five {text, source_ids} entries in sentences. Source
    IDs are proposed citations; missing RDF evidence remains UNKNOWN. Results
    are JSON-serializable and retain every proposal, recheck and rollback.
    """
    if topic not in TOPICS or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Provide a dataset topic and a nonempty prompt string.")
    if not isinstance(initial_summary, dict) or not isinstance(initial_summary.get("sentences"), list) \
            or len(initial_summary["sentences"]) != 5:
        raise ValueError("initial_summary must contain exactly five sentence entries.")
    if initial_summary.get("topic", topic) != topic:
        raise ValueError("Draft topic does not match the requested topic.")
    for sentence in initial_summary["sentences"]:
        if not isinstance(sentence, dict) or set(sentence) != {"text", "source_ids"}:
            raise ValueError("Each sentence needs exactly text and source_ids.")
        if not isinstance(sentence["text"], str) or not sentence["text"].strip():
            raise ValueError("Sentence text must be a nonempty string.")
        if not isinstance(sentence["source_ids"], list) or any(
            not isinstance(key, str) or not key.strip() for key in sentence["source_ids"]
        ):
            raise ValueError("Sentence source_ids must be a list of nonempty strings.")
    state: SummaryAgentState = {
        "topic": topic, "prompt": prompt,
        "initial_summary": deepcopy(initial_summary), "current_summary": deepcopy(initial_summary),
        "summary_claims": [], "extraction_errors": [], "factual_results": {}, "factual_issues": [],
        "representation_results": deepcopy(REPRESENTATION_NOT_RUN)
        if check_options.get("representation_metric") is None else {},
        "representation_issues": [], "viewpoint_results": deepcopy(VIEWPOINT_NOT_RUN), "viewpoint_issues": [],
        "proportion_results": deepcopy(PROPORTION_NOT_RUN), "proportion_issues": [],
        "revision_count": 0,
        "execution_history": [], "uncertainty_flags": [], "route": None,
        "revision_evidence": {}, "retrieval_history": [], "revisions": [],
        "summary_versions": [deepcopy(initial_summary)], "iterations": [], "best_iteration": 0,
        "routing_decisions": [], "call_counts": {"model_calls": 0, "tool_calls": 0, "embedding_calls": 0},
        "status": "running", "stop_reason": None,
    }
    owns_client = client is None
    client = OllamaClient() if owns_client else client
    try:
        client = recording_client(client)
        start_call = len(client.calls)
        workflow = build_summary_agent(graph, client, **check_options)
        result = workflow.invoke(state, config={"recursion_limit": 40})
        result.update(client.accounting(start_call), usage_scope="agent_only_excludes_initial_draft_and_source_extraction")
        return result
    finally:
        if owns_client:
            client.close()


def print_run_summary(result, output):
    """Concise terminal report; the complete state is saved separately as JSON."""
    factual, representation = result["factual_results"], result["representation_results"]
    counts = factual["verdict_counts"]
    print(f"Factual: {factual['status']} | " + " | ".join(f"{key}={value}" for key, value in counts.items()))
    if representation["status"] == "NOT_RUN":
        print("Representation: no group check in the agent (dialect labels are evaluation-only)")
    for label, row in representation["coverage_by_group"].items():
        coverage = f"{row['coverage']:.1%}" if row["coverage"] is not None else "UNKNOWN"
        if row["coverage"] is None and row["lower"] is not None and row["upper"] is not None:
            coverage += f" [{row['lower']:.1%}, {row['upper']:.1%}]"
        print(f"Coverage {label}: {coverage} | tweets "
              f"{representation['source_tweet_counts'][label]}/{representation['expected_group_counts'][label]}")
    if representation["status"] != "NOT_RUN":
        disparity = representation["disparity"]
        print(f"Representation: {representation['status']} | disparity="
              f"{disparity if disparity is not None else 'UNKNOWN'} | threshold={representation['disparity_threshold']}")
        if not representation["data_complete"]:
            print("Evidence is incomplete; full-topic representation cannot be determined.")
    views = result.get("viewpoint_results", VIEWPOINT_NOT_RUN)
    if views["status"] != "NOT_RUN":
        print(f"Viewpoints: {views['status']} | missing={views['missing'] or 'none'} | "
              f"tweet share {views['share_by_stance']}")
    proportion = result.get("proportion_results", PROPORTION_NOT_RUN)
    if proportion["status"] != "NOT_RUN":
        found = proportion["under_represented"]
        print(f"Proportion feedback: {proportion['status']} ({proportion['reason']})" +
              (f' | under-represented: "{found["viewpoint"]}"' if found else ""))
    revision_counts = {status: sum(r["status"] == status for r in result["revisions"])
                       for status in ("accepted", "rejected", "rolled_back")}
    print(f"Revisions: {result['revision_count']} attempted | " +
          " | ".join(f"{key}={value}" for key, value in revision_counts.items()))
    print(f"Final status: {result['status']} | stop reason: {result['stop_reason']}")
    if result["uncertainty_flags"]:
        print("Uncertainty: " + "; ".join(result["uncertainty_flags"]))
    print(f"Complete results: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("draft", type=Path, help="Saved summarizer JSON with five sentences")
    parser.add_argument("--graph", type=Path, required=True, help="Populated run graph in Turtle")
    parser.add_argument("--topic", choices=TOPICS, required=True)
    parser.add_argument("--prompt", help="Actual summary prompt; defaults to draft.summary_prompt")
    parser.add_argument("--disparity-threshold", type=float, default=0.2,
                        help="Max 1 - min/max group coverage (default: 0.2, the four-fifths rule)")
    parser.add_argument("--max-revisions", type=int, default=3)
    parser.add_argument("--representation-metric", choices=("none", "mnli", "internal"), default="none",
                        help="none (default): no group check, dialect labels are evaluation-only; "
                             "mnli/internal: the earlier label-based group check")
    parser.add_argument("--viewpoint-check", action="store_true",
                        help="Also repair a viewpoint of the tweets that the summary omits (label-free)")
    parser.add_argument("--proportion-check", action="store_true",
                        help="Add the feedback layer: Qwen checks the summary is in proportion to all tweets (label-free)")
    parser.add_argument("--embedding-model", default="all-minilm")
    parser.add_argument("--output", type=Path, help="Full JSON result file (default: unique file in results/evidence/agent_runs)")
    args = parser.parse_args()
    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    result = run_summary_agent(Graph().parse(args.graph, format="turtle"), draft,
                               topic=args.topic, prompt=args.prompt or draft.get("summary_prompt"),
                               disparity_threshold=args.disparity_threshold, max_revisions=args.max_revisions,
                               embedding_model=args.embedding_model, viewpoint_check=args.viewpoint_check,
                               proportion_check=args.proportion_check,
                               representation_metric=None if args.representation_metric == "none" else args.representation_metric)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    output = args.output or DEFAULT_RUN_DIR / (
        f"{args.topic}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print_run_summary(result, output)


if __name__ == "__main__":
    main()
