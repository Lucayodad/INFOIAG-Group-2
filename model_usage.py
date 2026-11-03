"""Count actual chat calls once, including extraction, retries and rechecks."""

from copy import deepcopy


class RecordingClient:
    def __init__(self, client):
        self.client = client
        self.calls = []
        if "live_feed" in getattr(client, "__dict__", {}):
            self.live_feed = client.__dict__["live_feed"]

    def __getattr__(self, name):
        return getattr(self.client, name)

    def chat(self, payload):
        row = {"call": len(self.calls) + 1, "model": payload.get("model"),
               "status": "failed", "usage": None}
        self.calls.append(row)
        try:
            response = self.client.chat(payload)
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            raise
        row.update(status="returned", usage=deepcopy(response.get("usage")))
        return response

    def accounting(self, start=0):
        calls = deepcopy(self.calls[start:])
        keys = ("input_tokens", "output_tokens", "total_tokens")
        complete = bool(calls) and all(
            row["status"] == "returned" and isinstance(row["usage"], dict)
            and all(type(row["usage"].get(key)) is int and row["usage"][key] >= 0 for key in keys)
            for row in calls)
        known = {key: sum(row["usage"].get(key, 0) for row in calls
                         if isinstance(row["usage"], dict)
                         and type(row["usage"].get(key)) is int) for key in keys}
        return {"model_calls": calls, "llm_calls": len(calls), "usage_complete": complete,
                "usage": known if complete else None, "known_usage": known}


def recording_client(client):
    return client if isinstance(client, RecordingClient) else RecordingClient(client)
