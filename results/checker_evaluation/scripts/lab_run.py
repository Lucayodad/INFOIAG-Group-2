"""Run a checker configuration on recorded claims; resumable JSONL output.

    python lab_run.py CONFIG {dev|test|all|KEY,...} OUT.jsonl [--live-first-turn]

The agent's first turn (always lookup_fact in the 210 recorded checks) is
replayed from the record unless --live-first-turn is given; every verdict
turn is a live Qwen call with the production request settings.
"""
import json, sys, time, os
from pathlib import Path

REPO = Path(r"C:\Users\saif-\Desktop\IA\INFOIAG-Group-2")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(REPO)]
os.chdir(REPO)

from ollama_backend import OllamaClient
from run_final_agent import LabelFreeClient, label_free_graph, first_check
from summary_claims import flatten_summary_claims
import fact_check_v, fact_agent_v
import configs


def first_turn(text):
    """The recorded opening decision, in the structured envelope when that is under test."""
    if not getattr(fact_agent_v, "STRUCTURED", False):
        return text
    decision = json.loads(text)
    decision.pop("verdict", None)
    return json.dumps({**decision, "missing_details": [], "contradicting": []}, ensure_ascii=False, indent=0)


class ReplayFirstTurn:
    """Returns the recorded first decision for the agent's opening turn."""

    def __init__(self, client):
        self._client, self.first = client, None

    def chat(self, payload):
        if self.first is not None and len(payload["messages"]) == 2:
            text, self.first = self.first, None
            return {"status": "completed", "usage": None,
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
        return self._client.chat(payload)

    def __getattr__(self, name):
        return getattr(self._client, name)


def main():
    name, which, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    live_first = "--live-first-turn" in sys.argv
    configs.apply(name, fact_check_v, fact_agent_v)
    claims = json.loads((HERE / "claims.json").read_text(encoding="utf-8"))
    if which == "dev":
        claims = [c for c in claims if c["topic"] == "McDonalds"]
    elif which == "test":
        claims = [c for c in claims if c["topic"] != "McDonalds"]
    elif which != "all":
        wanted = set(which.split(","))
        claims = [c for c in claims if c["key"] in wanted]
    done = set()
    if out.exists():
        done = {json.loads(line)["key"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    client = ReplayFirstTurn(LabelFreeClient(OllamaClient(timeout=600)))
    graphs, recorded = {}, {}
    for number, claim in enumerate(claims, 1):
        if claim["key"] in done:
            continue
        trial = claim["trial"]
        if trial not in graphs:
            rerun = json.loads((REPO / "results" / "agent_rerun" / f"{trial}.json").read_text(encoding="utf-8"))
            rerun = rerun.get("agent_result", rerun)
            graphs[trial] = label_free_graph(REPO / Path(rerun["pipeline"]["graph"].replace("\\", "/")))
            first = json.loads((REPO / "results" / "agent_final" / f"{trial}.json").read_text(encoding="utf-8"))
            checked = first_check(first["iterations"][0])
            recorded[trial] = (flatten_summary_claims(checked["summary_claims"]),
                               {a["summary_claim_id"]: a for a in checked["factual_results"]["assessments"]})
        flat, assessments = recorded[trial]
        summary_claim = next(c for c in flat if c["claim_id"] == claim["claim_id"])
        trace = assessments[claim["claim_id"]]["execution_trace"]
        client.first = None if live_first else first_turn(
            next(s["raw_response"] for s in trace if s["step"] == "model_decision"))
        started = time.time()
        result = fact_agent_v.run_fact_agent(graphs[trial], summary_claim, client)
        row = {"key": claim["key"], "config": name, "verdict": result["verdict"],
               "stop_reason": result["stop_reason"], "explanation": result["explanation"],
               "references": result["references"], "seconds": round(time.time() - started, 1),
               "model_calls": result["model_call_count"], "evidence_selection": result.get("evidence_selection"),
               "evidence_id_map": result.get("evidence_id_map"),
               "raw_decisions": [s.get("raw_response") for s in result["execution_trace"] if s["step"] == "model_decision"]}
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{number}/{len(claims)}] {claim['key']} {result['verdict']} ({result['stop_reason']}, {row['seconds']}s)",
              flush=True)


if __name__ == "__main__":
    main()
