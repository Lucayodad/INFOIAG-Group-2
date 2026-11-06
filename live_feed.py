"""Bounded, read-only observation of real pipeline handoffs and model/tool calls."""

from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock, Thread
from time import perf_counter
import json
import math
import sys


def bounded(value, depth=0):
    """Keep feed previews small; complete evidence remains in normal run results."""
    if depth > 5:
        return "[preview depth limit]"
    if isinstance(value, str):
        return value if len(value) <= 1000 else value[:1000] + " [preview truncated]"
    if isinstance(value, dict):
        result = {str(k): bounded(v, depth + 1) for k, v in list(value.items())[:32]}
        if len(value) > 32:
            result["_omitted_fields"] = len(value) - 32
        return result
    if isinstance(value, (list, tuple)):
        items = [bounded(v, depth + 1) for v in value[:12]]
        if len(value) > 12:
            items.append({"_omitted_items": len(value) - 12})
        return items
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return "[" + type(value).__name__ + "]"


class LiveFeed:
    """Flush JSONL and readable lines immediately without changing agent decisions."""
    def __init__(self, path, run_id, *, console=None, heartbeat_seconds=30):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id, self.sequence, self.started = run_id, 0, perf_counter()
        self.console = sys.stdout if console is None else console
        self.lock, self.stop = RLock(), Event()
        self.active, self.disabled, self.closed = None, False, False
        # Appending preserves the trace if an interrupted explicit output is retried.
        self.file = self.path.open("a", encoding="utf-8", buffering=1)
        self.thread = None
        if heartbeat_seconds > 0:
            self.thread = Thread(target=self._heartbeats, args=(heartbeat_seconds,), daemon=True)
            self.thread.start()

    def emit(self, kind, actor, recipient, summary, **details):
        with self.lock:
            if self.disabled or self.closed:
                return
            self.sequence += 1
            event = {"schema_version": 1, "run_id": self.run_id, "sequence": self.sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(perf_counter() - self.started, 3),
                "kind": kind, "actor": actor, "recipient": recipient,
                "summary": " ".join(str(summary).split())[:300], "details": bounded(details)}
            try:
                self.file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                self.file.flush()
                stamp = event["timestamp"][11:19] + "Z"
                print(f"[LIVE {stamp} #{self.sequence:04d}] {actor} -> {recipient} | {kind} | {event['summary']}",
                      file=self.console, flush=True)
            except (OSError, ValueError) as exc:
                self.disabled = True
                print(f"Live feed unavailable ({type(exc).__name__}); agent execution continues.", file=sys.stderr, flush=True)

    def begin_call(self, actor, recipient, summary, **details):
        with self.lock:
            self.active = (actor, recipient, summary, perf_counter(),
                           {k: v for k, v in details.items() if k != "request"})
            self.emit("call_started", actor, recipient, summary, **details)

    def end_call(self):
        with self.lock:
            self.active = None

    def heartbeat(self):
        with self.lock:
            if self.active:
                actor, recipient, summary, started, details = self.active
                self.emit("waiting", actor, recipient,
                    f"Still waiting: {summary}; {perf_counter() - started:.0f}s elapsed", **details)

    def _heartbeats(self, seconds):
        while not self.stop.wait(seconds):
            self.heartbeat()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
        with self.lock:
            self.closed = True
            self.file.close()


def emit_event(client, kind, actor, recipient, summary, **details):
    # Read an explicitly attached observer only; Mock clients must not invent one.
    feed = getattr(client, "__dict__", {}).get("live_feed")
    if isinstance(feed, LiveFeed):
        feed.emit(kind, actor, recipient, summary, **details)


def request_context(payload):
    fields = payload.get("format", {}).get("properties", {})
    bodies = []
    for message in payload.get("messages", []):
        if message.get("role") == "user":
            try:
                body = json.loads(message.get("content", ""))
            except (ValueError, TypeError):
                body = {"sentence": message.get("content", "")}
            if isinstance(body, dict):
                bodies.append(body)
    context = next((b for b in bodies if "summary_claim" in b), bodies[0] if bodies else {})
    if "sentences" in fields:
        actor = "summary_generator"
    elif "action" in fields:
        actor = "factual_checker"
    elif "replacements" in fields:
        actor = "reviser"
    else:
        actor = "source_extractor" if "id" in context else "summary_claim_extractor"
    ids = list(context.get("candidates", context.get("incoming", {})))
    claim = context.get("summary_claim", {})
    return actor, {"claim_id": claim.get("claim_id"), "tweet_id": context.get("id"),
        "candidate_ids": ids, "request": context,
        "format_repair": any("format_repair" in b for b in bodies)
                         or any("FORMAT REPAIR:" in m.get("content", "") for m in payload.get("messages", []))}


def node_details(name, state):
    """Describe actual shared state, including candidate versus accepted revisions."""
    if name == "extract_summary_claims":
        return {"claims": [c for r in state["summary_claims"] for c in r["claims"]],
                "extraction_errors": state["extraction_errors"]}
    if name == "factual_check":
        return {"status": state["factual_results"]["status"],
                "verdict_counts": state["factual_results"]["verdict_counts"], "issues": state["factual_issues"]}
    if name == "proportion_check":
        report = state["proportion_results"]
        return {"status": report["status"], "reason": report["reason"], "assessment": report["assessment"],
                "under_represented": report["under_represented"], "sentence_id": report["sentence_id"]}
    if name == "retrieve_revision_evidence":
        result = state["revision_evidence"]
        return {"status": result["status"], "reason": result["reason"],
                "tasks": result["tasks"], "tweets": result["tweets"]}
    return {"route": state["route"], "reason": state["stop_reason"], "status": state["status"],
            "revision_count": state["revision_count"],
            "latest_revision": ({k: state["revisions"][-1].get(k) for k in
                ("status", "reason", "changes", "attempt")} if state["revisions"] else None)}


def observe_node(name, function, client):
    next_node = {"extract_summary_claims": "factual_check", "finalize": "result_writer"}
    def observed(state):
        if not isinstance(getattr(client, "__dict__", {}).get("live_feed"), LiveFeed):
            return function(state)
        emit_event(client, "stage_started", "orchestrator", name,
            f"Starting {name}; revision {state['revision_count']}", revision_count=state["revision_count"])
        try:
            update = function(state)
        except Exception as exc:
            emit_event(client, "stage_failed", name, "orchestrator", str(exc), error_type=type(exc).__name__)
            raise
        current = {**state, **update}
        recipient = next_node.get(name, current["route"] or "orchestrator")
        details = node_details(name, current)
        note = details.get("reason") or details.get("status") or "validated claims ready"
        if details.get("latest_revision"):
            note += "; revision " + str(details["latest_revision"]["status"])
        emit_event(client, "handoff", name, recipient, f"{name} completed: {note}",
                   updated_fields=list(update), **details)
        return update
    return observed
