"""Synthetic evidence and mocked Qwen replies for the tests; every check itself is real code."""

import json
import re
from unittest.mock import Mock

from rdflib import RDF, Graph, Literal, URIRef

from data_loader import Tweet
from ontology_tools import NS
from populate_ontology import populate_ontology


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


def mock_qwen(draft, verdicts=("SUPPORTED", "CONTRADICTED", "UNKNOWN")):
    """Replies for claim extraction and one lookup_fact + finalize per claim, in order."""
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
    client = Mock()
    client.chat.side_effect = replies
    return client


def mock_embeddings(texts):
    """Gold-topic vectors for deterministic cosine tests, not real embeddings."""
    return [[1.0 if any(w in text.lower() for w in ("speech", "praise", "approve")) else 0.01,
             1.0 if "policy" in text.lower() else 0.01,
             1.0 if "detail" in text.lower() else 0.01] for text in texts]


def many_tweets(per_group=8):
    """(tweets, extraction records) of a synthetic topic with per_group tweets per group."""
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
    return tweets, records


def fixture(kind="both", many=False, per_group=8):
    """(graph, draft): the demo draft, or with many=True a larger topic of per_group tweets per group."""
    if not many:
        graph, draft = demo_fixture(contradiction=kind in ("both", "factual"))
        draft["sentences"][2]["text"] = "A tweet expresses praise for the speech."
        draft["summary"] = " ".join(s["text"] for s in draft["sentences"])
        return graph, draft
    tweets, records = many_tweets(per_group)
    sentences = [{"text": "One author likes Obama's speech.", "source_ids": ["AA_00"]}]
    sentences += [{"text": f"End of briefing {i}.", "source_ids": ["AA_00"]} for i in range(2, 6)]
    return populate_ontology(tweets, records, run_id="revision_many"), {
        "topic": "Obama", "summary_prompt": "Controlled revision example.",
        "sentences": sentences, "summary": " ".join(s["text"] for s in sentences)}


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


def mock_client(transform_revision=None):
    """Schema-valid model replies for extraction, fact checking and revision; real Python does all checks."""
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
                task = next(t for t in body["tasks"] if t["sentence_id"] == sentence_id)
                key = task["evidence_ids"][0]
                quote = by_id[key]["quote"]
                text = ("One author requests detail item " + re.search(r"item (\d+)", quote).group(1) + "."
                        if "item " in quote else "One author requests more detail." if "detail" in quote else
                        "One author dislikes the policy." if "policy" in quote else "One author likes Obama's speech.")
                replacements.append({"sentence_id": sentence_id, "text": text, "source_ids": [key],
                                     "evidence": [{"tweet_id": key, "quote": quote}]})
            data = {"replacements": replacements}
            return response(transform_revision(data) if transform_revision else data)
        if fields == {"status", "claims"}:
            text = raw
            entries = [] if text.startswith(("End of briefing", "Further context")) else [{
                "quote": text, "claim": text, "target": meaning(text),
                "stance": "negative" if "dislikes" in text else "positive" if meaning(text) == "speech" else "neutral"}]
            return response({"status": "extracted" if entries else "no_claim", "claims": entries})
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
    client.chat.side_effect = chat
    return client


# A small labelled topic for the proportion layer: 10 tweets, 4 positive, 4 negative,
# 1 neutral and 1 without any extracted claim.
STANCES = ["positive"] * 4 + ["negative"] * 4 + ["neutral", None]
TEXTS = ["Obama gave a hopeful and inspiring speech about jobs today",
         "Proud of the president for standing up for healthcare reform",
         "Great to see Obama visiting the schools in our town",
         "The new policy from the White House finally helps students",
         "Obama keeps failing the middle class with these taxes",
         "Another broken promise from this administration on immigration",
         "The president is weak on foreign policy and everyone knows it",
         "Obamacare premiums doubled again this year, terrible",
         "Obama will speak in Chicago on Tuesday evening",
         "Watching the news tonight"]


def topic_graph(labelled=False):
    graph = Graph()
    for number, (stance, text) in enumerate(zip(STANCES, TEXTS), 1):
        tweet, claim = URIRef(f"http://example.org/tweet/{number}"), URIRef(f"http://example.org/claim/{number}")
        graph.add((tweet, RDF.type, NS.Tweet))
        graph.add((tweet, NS.tweetId, Literal(f"Obama_{number:03d}")))
        graph.add((tweet, NS.aboutTopic, NS.Topic_Obama))
        graph.add((tweet, NS.tweetHasText, Literal(text)))
        if labelled:
            graph.add((tweet, NS.dialectLabel, Literal("AA")))
        if stance:
            graph.add((tweet, NS.containsClaim, claim))
            graph.add((claim, NS.stance, Literal(stance)))
    return graph


def record(sentence_id, claims, status="extracted"):
    quotes = [f"part {n} of {sentence_id}" for n in range(1, len(claims) + 1)]
    return {"sentence_id": sentence_id, "sentence": f"Sentence {sentence_id}: {', '.join(quotes) or 'none'}.",
            "status": status,
            "claims": [{"claim_id": f"{sentence_id}_C{n:02d}", "summary_quote": quotes[n - 1],
                        "claim": f"claim {n} of {sentence_id}", "target": "Obama", "stance": stance,
                        "cited_source_ids": cited, "verification_status": "unverified"}
                       for n, (stance, cited) in enumerate(claims, 1)]}


def checked(records, verdicts):
    """A checked iteration: verdicts by claim id (UNKNOWN when absent)."""
    assessments = [{"summary_claim_id": c["claim_id"], "sentence_id": r["sentence_id"],
                    "verdict": verdicts.get(c["claim_id"], "UNKNOWN"), "evidence_complete": True}
                   for r in records for c in r["claims"]]
    issues = [{"claim_id": a["summary_claim_id"], "sentence_id": a["sentence_id"], "type": "factual",
               "explanation": "checked", "severity": "serious" if a["verdict"] == "CONTRADICTED" else "unresolved"}
              for a in assessments if a["verdict"] != "SUPPORTED"]
    facts = {"assessments": assessments, "issues": issues,
             "status": "FAIL" if any(i["severity"] == "serious" for i in issues) else "UNKNOWN" if issues else "PASS"}
    return {"summary_claims": records, "factual_results": facts}


def draft_records():
    """S01 is the only negative sentence; nothing reports the positive viewpoint."""
    return [record("S01", [("negative", ["Obama_005"])]),
            record("S02", [("neutral", ["Obama_009"]), ("neutral", ["Obama_010"])]),
            record("S03", [("neutral", ["Obama_009"])]),
            record("S04", [("neutral", ["Obama_009"])]),
            record("S05", [("neutral", ["Obama_009"])])]


SUPPORTED_DRAFT = {"S01_C01": "SUPPORTED", "S02_C01": "SUPPORTED", "S03_C01": "SUPPORTED",
                   "S04_C01": "SUPPORTED", "S05_C01": "SUPPORTED"}  # S02_C02 stays UNKNOWN


def rewrite_evidence():
    """Retrieved evidence for rewriting S05: its own tweet Obama_009 and the new Obama_001."""
    tweets = [{"tweet_id": key, "quote": TEXTS[int(key[-3:]) - 1], "source_file": "tweets.csv",
               "source_line": int(key[-3:])} for key in ("Obama_009", "Obama_001")]
    return {"status": "ready", "tweets": tweets, "tasks": []}


def reviser_reply(ids):
    quotes = {"Obama_009": "speak in Chicago", "Obama_001": "hopeful and inspiring speech"}
    return {"output": [{"content": [{"text": json.dumps({"replacements": [{
        "sentence_id": "S05", "text": "Some users announce an upcoming Chicago appearance, while others "
                                      "praise a recent address by the president as uplifting.",
        "source_ids": ids, "evidence": [{"tweet_id": i, "quote": quotes[i]} for i in ids]}]})}]}], "usage": {}}
