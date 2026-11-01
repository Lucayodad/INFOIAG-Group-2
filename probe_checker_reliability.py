"""Two focused LIVE Qwen probes against the saved Obama pilot, never a full run."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from rdflib import Graph

from fact_agent import run_fact_agent
from ollama_backend import OllamaClient
from representation_check import match_summary_claims
from summary_claims import flatten_summary_claims


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("results/evidence/Obama_prompt1_pilot.json"))
    parser.add_argument("--graph", type=Path, default=Path(
        "results/ontology/20261101T183009Z_d38cb5571fa94c4cb684226f813c2646/Obama.ttl"))
    parser.add_argument("--output", type=Path, default=Path("results/evidence/checker_reliability_probes.json"))
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    # Exclusive creation prevents overwriting a previous probe or the pilot.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("Output already exists; choose a new --output path.")
    pilot_bytes, graph_bytes = args.pilot.read_bytes(), args.graph.read_bytes()
    pilot = json.loads(pilot_bytes)
    initial = pilot["iterations"][0]
    graph = Graph().parse(data=graph_bytes.decode("utf-8"), format="turtle")
    before = set(graph)
    client = OllamaClient(timeout=args.timeout)
    claim = next(c for c in flatten_summary_claims(initial["summary_claims"]) if c["claim_id"] == "S01_C01")
    print("LIVE factual probe: original S01_C01, up to 3 model calls.", flush=True)
    factual = run_fact_agent(graph, claim, client, max_tool_calls=1)
    print(f"Factual: {factual['verdict']} / {factual['stop_reason']}", flush=True)
    previous_batch = initial["representation_results"]["execution_trace"][1]["input"]["candidates"]
    source_ids = {item["claim_id"] for candidate in previous_batch.values() for item in candidate["source_evidence"]}
    groups = deepcopy(initial["representation_results"]["supporting_source_evidence"])
    for group in groups.values():
        group["claims"] = [claim for claim in group["claims"] if claim["claim_id"] in source_ids]
    print("LIVE representation probe: the 8 propositions in original K009-K016, up to 4 calls.", flush=True)
    representation = match_summary_claims(groups, initial["summary_claims"], client,
                                           max_model_calls=4, batch_size=8)
    report = {
        "mode": "LIVE local Qwen qwen3.5:4b; saved evidence; two focused probes, no LangGraph experiment",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pilot_sha256": hashlib.sha256(pilot_bytes).hexdigest(),
        "graph_sha256": hashlib.sha256(graph_bytes).hexdigest(),
        "original_representation_candidate_ids": list(previous_batch),
        "note": "The filtered representation probe assigns K001-K008 to the original K009-K016 propositions. Original claim IDs and quotes are preserved.",
        "factual": factual, "representation": representation,
    }
    assert set(graph) == before, "Probe must leave the RDF graph unchanged."
    assert args.pilot.read_bytes() == pilot_bytes and args.graph.read_bytes() == graph_bytes
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"factual": {key: factual.get(key) for key in
          ("verdict", "explanation", "stop_reason", "model_call_count", "evidence_selection")},
        "representation": {"aggregation_status": representation["aggregation_status"],
            "model_call_count": representation["model_call_count"], "reason": representation["reason"],
            "clusters": [{key: c[key] for key in ("source_claim_ids", "status", "explanation")}
                         for c in representation["clusters"]]}, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
