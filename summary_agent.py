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
from fact_check import fact_check_summary
from ollama_backend import OllamaClient
from model_usage import recording_client
from representation_check import representation_check
from revision_retrieval import retrieve_revision_evidence
from summary_revision import revise_summary, rewrite_unsupported_sentences
from summary_claims import extract_sentence, flatten_summary_claims
from summarizer import GenerationError, digest


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
                  "factual_issues", "representation_results", "representation_issues")


def _metrics(state):
    facts, rep = state["factual_results"], state["representation_results"]
    measured = (rep["status"] in ("PASS", "FAIL") and rep["data_complete"]
                and not rep["uncertain_claims"] and rep["disparity"] is not None
                and all(row["coverage"] is not None for row in rep["coverage_by_group"].values()))
    partition = sorted((sorted(c["source_claim_ids"]), sorted(
        (group, sorted(ids)) for group, ids in c["supporting_tweet_ids"].items()))
        for c in rep["matching"]["clusters"])
    return {"serious_factual_count": sum(i["severity"] == "serious" for i in facts["issues"]),
            "unknown_factual_count": sum(i["severity"] == "unresolved" for i in facts["issues"]),
            "disparity": rep["disparity"] if measured else None,
            "partition": digest(partition), "representation_status": rep["status"],
            "coverage": {g: row["coverage"] for g, row in rep["coverage_by_group"].items()}}


def compare_revision(previous, current, changes):
    """Conservative Pareto acceptance on the same measured quantities.

    Changed sentences must contain supported claims, not disappear into UNKNOWN
    or no_claim. A changed source partition cannot establish disparity progress.
    """
    targets = {row["sentence_id"] for row in changes}
    for record in current["summary_claims"]:
        if record["sentence_id"] not in targets:
            continue
        assessments = [a for a in current["factual_results"]["assessments"] if a["sentence_id"] == record["sentence_id"]]
        if record["status"] != "extracted" or not assessments or any(
            a["verdict"] != "SUPPORTED" or not a["evidence_complete"] for a in assessments
        ):
            return False, "unsupported_revision"
    before, after = _metrics(previous), _metrics(current)
    if after["serious_factual_count"] > before["serious_factual_count"] or \
            after["unknown_factual_count"] > before["unknown_factual_count"]:
        return False, "factual_regression"
    same_partition = before["partition"] == after["partition"]
    if before["disparity"] is not None and (
        after["disparity"] is None or not same_partition or after["disparity"] > before["disparity"] + 1e-12
    ):
        return False, "representation_regression_or_incomparable"
    if before["representation_status"] != "FAIL" and after["representation_status"] == "FAIL":
        return False, "representation_regression_or_incomparable"
    improved = (after["serious_factual_count"] < before["serious_factual_count"] or
                same_partition and before["disparity"] is not None and after["disparity"] is not None
                and after["disparity"] < before["disparity"] - 1e-12)
    return (True, "measured_improvement") if improved else (False, "no_improvement")


def _counts(state, **increments):
    return {key: value + increments.get(key, 0) for key, value in state["call_counts"].items()}


def build_summary_agent(graph, client, *, disparity_threshold=None,
                        allowed_entity_ids=None, max_tool_calls=3,
                        expected_group_counts=None, max_model_calls=64,
                        max_input_bytes=12000, batch_size=8, max_revisions=3,
                        embedder=None, embedding_model="all-minilm", revision_top_k=3,
                        max_revision_tweets=9, revision_evidence_bytes=7000,
                        revision_max_input_bytes=14000, mnli_scorer=None,
                        representation_metric="mnli"):
    """Compile the sequential graph with read-only evidence outside its state.

    Serious issues mean conclusive under the existing evidence guards, not
    human-verified truth. UNKNOWN alone never triggers revision.
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
    evidence = ReadOnlyGraphAggregate([graph])
    embedding_cache = {}
    embedding_function = embedder if embedder is not None else lambda texts: client.embed(texts, model=embedding_model)

    def extract_claims(state: SummaryAgentState):
        records, errors = [], []
        for number, sentence in enumerate(state["current_summary"]["sentences"], 1):
            try:
                records.append(extract_sentence(sentence, number, client))
            except (GenerationError, ValueError, KeyError, IndexError, TypeError) as exc:
                # A failed extraction is missing knowledge, never a no-claim result.
                records.append({"sentence_id": f"S{number:02d}", "sentence": sentence["text"],
                                "status": "unclear", "claims": []})
                errors.append({"sentence_id": f"S{number:02d}",
                               "error_type": type(exc).__name__, "message": str(exc)})
        flatten_summary_claims(records)
        return _update(state, "extract_summary_claims", summary_claims=records,
                       extraction_errors=errors, call_counts=_counts(state, model_calls=5))

    def check_facts(state: SummaryAgentState):
        result = fact_check_summary(evidence, state["summary_claims"], client,
                                    allowed_entity_ids=allowed_entity_ids,
                                    max_tool_calls=max_tool_calls)
        return _update(state, "factual_check", factual_results=result,
                       factual_issues=result["issues"], call_counts=_counts(
                           state, model_calls=result["model_call_count"], tool_calls=result["tool_call_count"]))

    def check_representation(state: SummaryAgentState):
        result = representation_check(
            evidence, state["summary_claims"], client, state["topic"],
            disparity_threshold=disparity_threshold, expected_group_counts=expected_group_counts,
            max_model_calls=max_model_calls, max_input_bytes=max_input_bytes, batch_size=batch_size,
            embedder=embedding_function, embedding_model=embedding_model, embedding_cache=embedding_cache,
            metric=representation_metric, mnli_scorer=mnli_scorer)
        return _update(state, "representation_check", representation_results=result,
                       representation_issues=result["representation_issues"],
                       call_counts=_counts(state, model_calls=result["model_call_count"],
                                           embedding_calls=result.get("embedding_call_count", 0)))

    def route_checks(state: SummaryAgentState):
        if (state["revisions"] and state["revisions"][-1]["status"] == "candidate"
                and "partial_repair" not in state["revisions"][-1]):
            last = state["revisions"][-1]
            evidence = {row["tweet_id"]: row for row in state["revision_evidence"].get("tweets", [])}
            partial = rewrite_unsupported_sentences(
                state["current_summary"], last["changes"],
                state["factual_results"].get("assessments", []), evidence, client)
            if partial["status"] == "candidate":
                candidate = partial["candidate"]
                revised_changes = deepcopy(last["changes"])
                for change in revised_changes:
                    index = int(change["sentence_id"][1:]) - 1
                    change["after"] = deepcopy(candidate["sentences"][index])
                revisions = deepcopy(state["revisions"])
                revisions[-1].update(candidate=candidate, changes=revised_changes,
                                     partial_repair=partial)
                return _update(
                    state, "route_checks", current_summary=candidate,
                    summary_versions=state["summary_versions"] + [deepcopy(candidate)],
                    revisions=revisions, route="extract_summary_claims",
                    stop_reason="partial_repair_requires_rechecking",
                    call_counts=_counts(state, model_calls=partial["model_call_count"]))
        checked = {key: deepcopy(state[key]) for key in CHECKED_FIELDS}
        iteration = {**checked, "metrics": _metrics(state), "revision_count": state["revision_count"],
                     "call_counts": dict(state["call_counts"]), "retained": True}
        iterations, revisions = deepcopy(state["iterations"]), deepcopy(state["revisions"])
        iterations.append(iteration)
        best = len(iterations) - 1
        if revisions and revisions[-1]["status"] == "candidate":
            accepted, reason = compare_revision(iterations[state["best_iteration"]], checked, revisions[-1]["changes"])
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
                   if issue["severity"] == "serious"]
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
                                         (state["factual_results"], representation))
        if representation["status"] == "UNKNOWN":
            flags.append("representation_uncertainty")
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
            embedding_cache=embedding_cache, embedding_model=embedding_model)
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
                  for issue in state["factual_issues"] + state["representation_issues"] if issue["severity"] == "serious"]
        last = state["revisions"][-1] if state["revisions"] else None
        if last and last["status"] == "rolled_back":
            checked = state["iterations"][last["checked_iteration"]]["factual_results"]["assessments"]
            changed = {row["sentence_id"]: row["after"]["text"] for row in last["changes"]}
            findings = [f'{a["summary_claim_text"]}: {a["verdict"]}; {a["explanation"][:240]}'
                        for a in checked if a["sentence_id"] in changed and a["verdict"] != "SUPPORTED"]
            issues.append({"type": "rejected_previous_revision", "severity": "serious", "explanation":
                           f"Rechecking rolled back your previous replacement ({last['reason']}): "
                           f"{json.dumps(changed, ensure_ascii=False)}. Findings: {' | '.join(findings)}"})
        proposal = revise_summary(state["current_summary"], issues, state["revision_evidence"],
                                  client, max_input_bytes=revision_max_input_bytes)
        attempt = {**proposal, "attempt": state["revision_count"] + 1,
                   "summary_before": deepcopy(state["current_summary"]), "evidence": state["revision_evidence"],
                   "metrics_before": _metrics(state)}
        candidate = proposal["status"] == "candidate"
        decision = {"node": "revise_summary", "route": "extract_summary_claims" if candidate else "finalize",
                    "reason": proposal["reason"]}
        changes = {"current_summary": proposal["candidate"],
                   "summary_versions": state["summary_versions"] + [deepcopy(proposal["candidate"])]} if candidate else {}
        return _update(state, "revise_summary", **changes,
                       revisions=state["revisions"] + [attempt],
                       # A format repair belongs to the same revision attempt.
                       revision_count=state["revision_count"] + bool(proposal["model_call_count"]),
                       route="extract_summary_claims" if candidate else "finalize",
                       stop_reason="revision_requires_rechecking" if candidate else proposal["reason"],
                       routing_decisions=state["routing_decisions"] + [decision],
                       call_counts=_counts(state, model_calls=proposal["model_call_count"]))

    def finalize(state: SummaryAgentState):
        remaining = any(result["status"] == "FAIL" for result in
                        (state["factual_results"], state["representation_results"]))
        status = ("stopped_with_issues" if remaining else
                  "completed_with_uncertainty" if state["uncertainty_flags"] else "completed")
        return _update(state, "finalize", status=status)

    builder = StateGraph(SummaryAgentState)
    for name, function in (("extract_summary_claims", extract_claims), ("factual_check", check_facts),
                           ("representation_check", check_representation), ("route_checks", route_checks),
                           ("retrieve_revision_evidence", retrieve_evidence),
                           ("revise_summary", revise), ("finalize", finalize)):
        builder.add_node(name, observe_node(name, function, client))
    builder.add_edge(START, "extract_summary_claims")
    builder.add_edge("extract_summary_claims", "factual_check")
    builder.add_edge("factual_check", "representation_check")
    builder.add_edge("representation_check", "route_checks")
    builder.add_conditional_edges("route_checks", lambda state: state["route"],
                                  {"finalize": "finalize", "retrieve_revision_evidence": "retrieve_revision_evidence",
                                   "extract_summary_claims": "extract_summary_claims"})
    builder.add_conditional_edges("retrieve_revision_evidence", lambda state: state["route"],
                                  {"finalize": "finalize", "revise_summary": "revise_summary"})
    builder.add_conditional_edges("revise_summary", lambda state: state["route"],
                                  {"finalize": "finalize", "extract_summary_claims": "extract_summary_claims"})
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
        "representation_results": {}, "representation_issues": [], "revision_count": 0,
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
    for label, row in representation["coverage_by_group"].items():
        coverage = f"{row['coverage']:.1%}" if row["coverage"] is not None else "UNKNOWN"
        if row["coverage"] is None and row["lower"] is not None and row["upper"] is not None:
            coverage += f" [{row['lower']:.1%}, {row['upper']:.1%}]"
        print(f"Coverage {label}: {coverage} | tweets "
              f"{representation['source_tweet_counts'][label]}/{representation['expected_group_counts'][label]}")
    disparity = representation["disparity"]
    print(f"Representation: {representation['status']} | disparity="
          f"{disparity if disparity is not None else 'UNKNOWN'} | threshold={representation['disparity_threshold']}")
    if not representation["data_complete"]:
        print("Evidence is incomplete; full-topic representation cannot be determined.")
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
    parser.add_argument("--representation-metric", choices=("mnli", "internal"), default="mnli")
    parser.add_argument("--embedding-model", default="all-minilm")
    parser.add_argument("--output", type=Path, help="Full JSON result file (default: unique file in results/evidence/agent_runs)")
    args = parser.parse_args()
    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    result = run_summary_agent(Graph().parse(args.graph, format="turtle"), draft,
                               topic=args.topic, prompt=args.prompt or draft.get("summary_prompt"),
                               disparity_threshold=args.disparity_threshold, max_revisions=args.max_revisions,
                               embedding_model=args.embedding_model, representation_metric=args.representation_metric)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    output = args.output or DEFAULT_RUN_DIR / (
        f"{args.topic}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print_run_summary(result, output)


if __name__ == "__main__":
    main()
