"""Controlled revision/rechecking demo. All decisions/embeddings are mocked unless --live."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
from unittest.mock import Mock

from data_loader import Tweet
from demo_summary_agent import demo_fixture, response
from ollama_backend import OllamaClient
from populate_ontology import populate_ontology
from revision_retrieval import retrieve_revision_evidence
from summary_agent import run_summary_agent
from summary_revision import revise_summary


def mock_embeddings(texts):
    """Gold-topic vectors for deterministic cosine tests, not real embeddings."""
    return [[1.0 if any(w in text.lower() for w in ("speech", "praise", "approve")) else 0.01,
             1.0 if "policy" in text.lower() else 0.01,
             1.0 if "detail" in text.lower() else 0.01] for text in texts]


def fixture(kind="both", many=False, per_group=8):
    if not many:
        graph, draft = demo_fixture(contradiction=kind in ("both", "factual"))
        draft["sentences"][2]["text"] = "A tweet expresses praise for the speech."
        draft["summary"] = " ".join(s["text"] for s in draft["sentences"])
        return graph, draft, {g: 3 for g in ("AA", "Hisp", "White")}
    tweets, records = [], []
    for group in ("AA", "Hisp", "White"):
        for i in range(per_group):
            key = f"{group}_{i:02d}"
            text = f"I want detail item {(i - 1) % 7 + 1}." if group == "White" and i else "I like Obama's speech."
            tweets.append(Tweet(key, "Obama", group, text, "synthetic_revision.txt", len(tweets) + 1))
            records.append({"tweet_id": key, "dialect_label": group, "status": "extracted", "claims": [{
                "claim_id": key + "_C01", "tweet_id": key, "dialect_label": group,
                "source_quote": text, "claim": text, "target": "speech" if "speech" in text else text,
                "stance": "positive" if "speech" in text else "neutral"}]})
    sentences = [{"text": "One author likes Obama's speech.", "source_ids": ["AA_00"]}]
    sentences += [{"text": f"End of briefing {i}.", "source_ids": ["AA_00"]} for i in range(2, 6)]
    return populate_ontology(tweets, records, run_id="revision_many"), {
        "topic": "Obama", "summary_prompt": "Controlled revision example.",
        "sentences": sentences, "summary": " ".join(s["text"] for s in sentences)}, {g: per_group for g in ("AA", "Hisp", "White")}


def meaning(text):
    text = text.lower()
    if "detail" in text:
        number = re.search(r"item (\d+)", text)
        return "detail" + (number.group(1) if number else "")
    if "policy" in text:
        return "policy"
    if any(word in text for word in ("speech", "praise", "approve")):
        return "speech"
    return "other"


def mock_client(transform_revision=None, no_improvement=False, unknown_matching=False):
    """Generate schema-valid model replies; real Python functions do all checks."""
    client = Mock()
    client.embed.side_effect = lambda texts, model="all-minilm": mock_embeddings(texts)
    def chat(payload):
        fields = set(payload["format"]["properties"])
        raw = payload["messages"][-1]["content"]
        if "replacements" in fields:
            body = json.loads(payload["messages"][1]["content"])  # original request, also on repair
            by_id = {row["tweet_id"]: row for row in body["original_tweets"]}
            replacements = []
            for sentence_id in sorted({t["sentence_id"] for t in body["tasks"]}):
                tasks = [t for t in body["tasks"] if t["sentence_id"] == sentence_id]
                task = next((t for t in tasks if t["kind"] == "representation"), tasks[0])
                key = task["evidence_ids"][0]
                quote = by_id[key]["quote"]
                text = ("One author requests detail item " + re.search(r"item (\d+)", quote).group(1) + "."
                        if "item " in quote else "One author requests more detail." if "detail" in quote else
                        "One author dislikes the policy." if "policy" in quote else "One author likes Obama's speech.")
                ids = [key]
                if task["kind"] == "representation":
                    ids = task["evidence_ids"][:2]
                replacements.append({"sentence_id": sentence_id, "text": text, "source_ids": ids,
                                     "evidence": [{"tweet_id": i, "quote": by_id[i]["quote"]} for i in ids]})
            data = {"replacements": replacements}
            return response(transform_revision(data) if transform_revision else data)
        if fields == {"status", "claims"}:
            text = raw
            entries = [] if text.startswith(("End of briefing", "Further context")) else [{
                "quote": text, "claim": text, "target": meaning(text),
                "stance": "negative" if "dislikes" in text else "positive" if meaning(text) == "speech" else "neutral"}]
            return response({"status": "extracted" if entries else "no_claim", "claims": entries})
        if "action" in fields:
            original = json.loads(payload["messages"][1]["content"])["summary_claim"]
            if len(payload["messages"]) == 2:
                return response({"action": "lookup_fact", "entity_id": None, "reason": "Read original citations.",
                                 "verdict": None, "explanation": None, "references": []})
            observation = json.loads(raw)["observation"]
            tweets = [{"tweet_uri": key, "text": item["quote_text"]}
                      for key, item in observation["evidence_catalogue"].items() if item["kind"] == "tweet"]
            text = original["claim"]
            supporting = [t for t in tweets if meaning(t["text"] or "") == meaning(text)]
            contradicted = "dislikes" in text and "speech" in text and any("I like" in (t["text"] or "") for t in tweets)
            verdict = "CONTRADICTED" if contradicted else "SUPPORTED" if supporting and "Millions" not in text and "invented" not in text else "UNKNOWN"
            sources = tweets if contradicted else supporting
            refs = [{"evidence_id": sources[0]["tweet_uri"], "quote": sources[0]["text"]}] if verdict != "UNKNOWN" else []
            return response({"action": "finalize", "entity_id": None, "reason": "Controlled evidence decision.",
                             "verdict": verdict, "explanation": "Gold-labelled synthetic decision.", "references": refs})
        body = json.loads(raw)
        if "summary_claims" not in body:
            return response({"equivalence_complete": True, "clusters": [
                {"candidate_ids": [key], "source_evidence": [{"claim_id": c["source_evidence"][0]["claim_id"],
                 "quote": c["source_evidence"][0]["source_quote"]}], "explanation": "Distinct fixture propositions."}
                for key, c in body["candidates"].items()]})
        clusters = []
        for key, candidate in body["candidates"].items():
            category = meaning(candidate["claim"])
            matches = [c for c in body["summary_claims"] if meaning(c["claim"]) == category]
            if no_improvement and category != "speech":
                matches = []
            status = "UNKNOWN" if unknown_matching and category != "speech" else "MATCHED" if matches else "NOT_MATCHED"
            matched = matches[:1] if status == "MATCHED" else []
            source = candidate["source_evidence"][0]
            clusters.append({"candidate_ids": [key], "status": status,
                             "summary_claim_ids": [c["claim_id"] for c in matched],
                             "source_evidence": [{"claim_id": source["claim_id"], "quote": source["source_quote"]}],
                             "summary_evidence": [{"claim_id": c["claim_id"], "quote": c["summary_quote"]} for c in matched],
                             "explanation": "Controlled semantic matching decision."})
        return response({"equivalence_complete": True, "clusters": clusters})
    client.chat.side_effect = chat
    return client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/evidence/revision_demo.json"))
    parser.add_argument("--live", action="store_true", help="One separate live Qwen proposal on controlled evidence")
    args = parser.parse_args()
    graph, draft, counts = fixture()
    result = run_summary_agent(graph, draft, topic="Obama", prompt=draft["summary_prompt"],
                               client=mock_client(), expected_group_counts=counts, disparity_threshold=0.25,
                               representation_metric="internal")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"mode": "MOCKED Qwen and gold-topic embeddings; real graph, retrieval and checkers",
              "original_summary": draft["summary"], "final_summary": result["current_summary"]["summary"],
              "nodes": [e["node"] for e in result["execution_history"]],
              "iterations": [{"revision_count": i["revision_count"], "metrics": i["metrics"],
                              "factual_issues": i["factual_issues"]} for i in result["iterations"]],
              "revisions": [{"changes": r["changes"], "status": r["status"], "reason": r["reason"],
                             "evidence": r["evidence"]} for r in result["revisions"]],
              "stop_reason": result["stop_reason"], "status": result["status"], "call_counts": result["call_counts"]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.live:
        client = OllamaClient(timeout=60)
        try:
            initial = result["iterations"][0]
            retrieved = retrieve_revision_evidence(graph, "Obama", initial["summary_claims"], initial["factual_results"],
                                                   initial["representation_results"], client.embed)
            live = revise_summary(draft, [{"type": "factual_contradiction", "severity": "serious",
                                          "explanation": "Correct the contradiction and represent the omitted viewpoint."}], retrieved, client)
            live["mode"] = "REAL local all-minilm retrieval and one Qwen revision proposal; not an accuracy evaluation"
            live["retrieval"] = retrieved
            path = args.output.with_name("revision_live_probe.json")
            path.write_text(json.dumps(live, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"live_status": live["status"], "reason": live["reason"],
                              "changes": live["changes"], "saved": str(path)}, indent=2))
        finally:
            client.close()


if __name__ == "__main__":
    main()
