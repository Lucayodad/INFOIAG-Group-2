"""Check-only LangGraph demo; use demo_revision_loop.py for revision and rechecking."""

import argparse
import json
from pathlib import Path
from unittest.mock import Mock

from data_loader import Tweet
from populate_ontology import populate_ontology
from summary_agent import run_summary_agent


EXPECTED_COUNTS = {label: 3 for label in ("AA", "Hisp", "White")}


def response(data):
    return {"output": [{"content": [{"text": json.dumps(data)}]}]}


def demo_fixture(contradiction=True):
    """Nine fully specified source tweets, three per group; no real evaluation."""
    categories = {"AA": ("speech", "speech", "policy"),
                  "Hisp": ("speech", "speech", "policy"),
                  "White": ("speech", "policy", "detail")}
    meanings = {
        "speech": ("I like Obama's speech.", "The author likes Obama's speech.", "positive"),
        "policy": ("I dislike the policy.", "The author dislikes the policy.", "negative"),
        "detail": ("I want more detail.", "The author wants more detail.", "neutral"),
    }
    tweets, records = [], []
    for group, entries in categories.items():
        for index, category in enumerate(entries, 1):
            key = f"{group}_{index:02d}"
            text, claim, stance = meanings[category]
            tweets.append(Tweet(key, "Obama", group, text, "synthetic_langgraph.txt", len(tweets) + 1))
            records.append({"tweet_id": key, "dialect_label": group, "status": "extracted", "claims": [{
                "claim_id": key + "_C01", "tweet_id": key, "dialect_label": group,
                "source_quote": text, "claim": claim, "target": category, "stance": stance,
            }]})
    texts = ["The author likes Obama's speech.",
             "The author dislikes Obama's speech." if contradiction else "An author approves of the speech.",
             "Millions watched the speech." if contradiction else "A tweet expresses praise for the speech.",
             "End of briefing.", "Further context follows."]
    draft = {"topic": "Obama", "summary_prompt": "Synthetic five-sentence summary for orchestration testing.",
             "summary": " ".join(texts),
             "sentences": [{"text": text, "source_ids": ["AA_01"]} for text in texts]}
    return populate_ontology(tweets, records, run_id="langgraph_demo"), draft


def mock_qwen(draft, verdicts=("SUPPORTED", "CONTRADICTED", "UNKNOWN"), match_status="MATCHED",
              policy_status="NOT_MATCHED"):
    """Mock model replies only; extraction, tools, checks and LangGraph are real."""
    replies = []
    for index, sentence in enumerate(draft["sentences"]):
        entries = [{"quote": sentence["text"], "claim": sentence["text"], "target": "speech",
                    "stance": "positive" if index == 0 else "negative" if index == 1 else "neutral"}] if index < 3 else []
        replies.append(response({"status": "extracted" if entries else "no_claim", "claims": entries}))
    for verdict in verdicts:
        replies.append(response({"action": "lookup_fact", "entity_id": None,
                                 "reason": "Read the cited source tweet.",
                                 "verdict": None, "explanation": None, "references": []}))
        references = [{"evidence_id": "http://www.semanticweb.org/group2/runs/langgraph_demo/tweet/AA_01",
                       "quote": "I like Obama's speech."}] if verdict != "UNKNOWN" else []
        replies.append(response({"action": "finalize", "entity_id": None, "reason": "Evidence reviewed.",
                                 "verdict": verdict, "references": references,
                                 "explanation": "Synthetic decision based on the cited tweet; no independent truth claim."}))
    specs = [("K001", "AA_01_C01", "I like Obama's speech.", match_status),
             ("K002", "AA_03_C01", "I dislike the policy.", policy_status),
             ("K003", "White_03_C01", "I want more detail.", "NOT_MATCHED")]
    clusters = []
    for key, claim_id, quote, status in specs:
        matched = status == "MATCHED"
        clusters.append({"candidate_ids": [key], "status": status,
                         "summary_claim_ids": ["S01_C01"] if matched else [],
                         "source_evidence": [{"claim_id": claim_id, "quote": quote}],
                         "summary_evidence": [{"claim_id": "S01_C01", "quote": draft["sentences"][0]["text"]}]
                                             if matched else [],
                         "explanation": "Gold-labelled synthetic match decision."})
    replies.append(response({"equivalence_complete": True, "clusters": clusters}))
    replies.insert(len(replies) - 1, response({"equivalence_complete": True, "clusters": [
        {key: entry[key] for key in ("candidate_ids", "source_evidence", "explanation")}
        for entry in clusters]}))
    client = Mock()
    client.chat.side_effect = replies
    return client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional full execution state JSON")
    args = parser.parse_args()
    graph, draft = demo_fixture()
    result = run_summary_agent(graph, draft, topic="Obama", prompt=draft["summary_prompt"],
                               client=mock_qwen(draft), expected_group_counts=EXPECTED_COUNTS,
                               disparity_threshold=0.25, max_revisions=0, representation_metric="internal")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    representation = result["representation_results"]
    print(json.dumps({
        "fixture": "Synthetic complete input: 3 tweets per group; all model replies mocked.",
        "nodes_visited": [event["node"] for event in result["execution_history"]],
        "state_transitions": result["execution_history"],
        "factual_results": {key: result["factual_results"][key] for key in
                            ("status", "claims_examined", "verdict_counts", "model_call_count", "tool_call_count")},
        "representation_results": {"status": representation["status"],
                                   "coverage": {key: value["coverage"] for key, value in representation["coverage_by_group"].items()},
                                   "disparity": representation["disparity"],
                                   "disparity_threshold": representation["disparity_threshold"]},
        "route": result["route"], "status": result["status"], "stop_reason": result["stop_reason"],
        "revision_count": result["revision_count"], "uncertainty_flags": result["uncertainty_flags"],
    }, indent=2))


if __name__ == "__main__":
    main()
