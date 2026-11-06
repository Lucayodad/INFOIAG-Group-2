"""Citation repair: re-check UNKNOWN claims with the topic's best-matching uncited tweets added.

    python lab_recite.py CONFIG PREDICTIONS.jsonl OUT.jsonl [K]

Candidates are ranked by the sum of two ranks: all-minilm cosine to the claim
and idf-weighted word overlap (both computed without any label).
"""
import collections, html, json, math, re, sys, time
from pathlib import Path

import lab_run  # sets sys.path and the working directory
from lab_run import REPO, HERE, ReplayFirstTurn, first_turn, OllamaClient, LabelFreeClient, label_free_graph, first_check
from lab_run import flatten_summary_claims, fact_check_v, fact_agent_v, configs
from data_loader import load_tweets

STOP = set(("the a an and or of to in on for with is are was were be been their they them this that these those "
            "it its as at by from some many several users user people tweets tweet posts post others other there "
            "about include includes including expressed express described describe mention mentioned focus focused "
            "who which have has had not no more most also such like").split())
TOPIC_WORDS = {"McDonalds": {"mcdonalds", "mcdonald", "mcdonald's"}, "Obama": {"obama", "barack", "president"},
               "Superbowl": {"super", "bowl", "superbowl"}}


def stem(word):
    return re.sub(r"(ing|ed|es|s|ly)$", "", word)


def words(text, topic):
    return {stem(w) for w in re.findall(r"[a-z']+", text.lower())
            if w not in STOP and w not in TOPIC_WORDS[topic] and len(w) > 2}


def candidates(claim_text, cited, topic, tweets, vectors, claim_vector, k):
    df = collections.Counter(w for _, text in tweets for w in words(text, topic))
    query = words(claim_text, topic)
    pool = [(key, text) for key, text in tweets if key not in cited]
    cosine = lambda a, b: sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    by_vector = sorted(pool, key=lambda row: (-cosine(claim_vector, vectors[row[0]]), row[0]))
    by_words = sorted(pool, key=lambda row: (-sum(math.log(len(tweets) / df[w]) for w in query & words(row[1], topic)),
                                             row[0]))
    rank = {row[0]: by_vector.index(row) + by_words.index(row) for row in pool}
    return sorted(rank, key=lambda key: (rank[key], key))[:k]


def main():
    name, predictions, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    configs.apply(name, fact_check_v, fact_agent_v)
    claims = {c["key"]: c for c in json.loads((HERE / "claims.json").read_text(encoding="utf-8"))}
    vectors = json.loads((HERE / "embeddings.json").read_text(encoding="utf-8"))
    tweets = {t: [(x.id, html.unescape(x.text)) for x in load_tweets(t)] for t in ("McDonalds", "Obama", "Superbowl")}
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    todo = [row["key"] for row in rows if row["verdict"] == "UNKNOWN"]
    done = set()
    if out.exists():
        done = {json.loads(line)["key"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    client = ReplayFirstTurn(LabelFreeClient(OllamaClient(timeout=600)))
    graphs, recorded = {}, {}
    for number, key in enumerate(todo, 1):
        if key in done:
            continue
        claim = claims[key]
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
        summary_claim = dict(next(c for c in flat if c["claim_id"] == claim["claim_id"]))
        added = candidates(claim["claim"], set(claim["cited"]), claim["topic"], tweets[claim["topic"]],
                           vectors, vectors[key], k)
        summary_claim["cited_source_ids"] = list(summary_claim["cited_source_ids"]) + added
        trace = assessments[claim["claim_id"]]["execution_trace"]
        client.first = first_turn(next(s["raw_response"] for s in trace if s["step"] == "model_decision"))
        started = time.time()
        result = fact_agent_v.run_fact_agent(graphs[trial], summary_claim, client)
        row = {"key": key, "config": name + "+recite", "added": added, "verdict": result["verdict"],
               "stop_reason": result["stop_reason"], "explanation": result["explanation"],
               "references": result["references"], "seconds": round(time.time() - started, 1)}
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{number}/{len(todo)}] {key} {result['verdict']} ({row['seconds']}s) added {added}", flush=True)


if __name__ == "__main__":
    main()
