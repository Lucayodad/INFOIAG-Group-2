"""Re-derive structured (D) verdicts from the saved raw reports under the counting rules; score them.

    python rescore_d.py RUN.jsonl
"""
import json, sys, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), r"C:\Users\saif-\Desktop\IA\INFOIAG-Group-2"]
import fact_agent_v
from score import load_gold, score


def derive(row, claim_text):
    """Verdict from the last raw structured report; None if it was not a valid finalize."""
    raw = row["raw_decisions"][-1] if row.get("raw_decisions") else None
    try:
        decision = json.loads(raw)
        envelope = fact_agent_v.structured_to_envelope(decision, claim_text, row["evidence_id_map"] or {})
    except (TypeError, ValueError):
        return None
    return envelope.get("verdict") if envelope.get("action") == "finalize" else None


if __name__ == "__main__":
    gold = load_gold()
    claims = {c["key"]: c for c in json.loads((HERE / "claims.json").read_text(encoding="utf-8"))}
    rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
    recorded = {r["key"]: r["verdict"] for r in rows}
    derived = {r["key"]: derive(r, claims[r["key"]]["claim"]) or r["verdict"] for r in rows}
    print("derived == recorded:", sum(derived[k] == recorded[k] for k in recorded), "/", len(recorded))
    score(recorded, gold, list(recorded), "as run")
    score(derived, gold, list(recorded), "re-derived")
