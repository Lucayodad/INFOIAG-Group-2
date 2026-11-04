"""The label-free viewpoint check: the agent's reflection on its own bias.

A viewpoint (positive/negative stance of the claims extracted from the
tweets' text) held by enough of the topic's tweets must be reported by the
summary. The check reads no dialect label, follows the open world assumption
(an unextracted sentence makes a viewpoint UNKNOWN, never missing), repairs a
missing viewpoint only after every contradiction, and its gate adds a
viewpoint only when a SUPPORTED claim reports it. Off by default.

    python -m unittest tests.test_viewpoint_check -v
"""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock

from rdflib import RDF, Graph, Literal, URIRef

from ontology_tools import NS
from revision_retrieval import retrieve_revision_evidence, viewpoint_task
from run_final_agent import LABEL_LEAK, label_free_graph
from summary_agent import REPRESENTATION_NOT_RUN, build_summary_agent, compare_revision, run_summary_agent, \
    seed_check_cache
from summary_revision import REVISER_PROFILE, REVISION_INSTRUCTIONS, revise_summary
from tests.test_label_free import RecordingQwen, embed, first_check
from tests.test_real_pieces import graph_path, recorded
from viewpoint_check import VIEWPOINT_NOT_RUN, check_viewpoints, topic_stances


# 10 tweets: 4 positive, 4 negative, 1 neutral, 1 without any claim.
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


def checked(graph, records, verdicts):
    """A checked iteration: verdicts by claim id (UNKNOWN when absent)."""
    assessments = [{"summary_claim_id": c["claim_id"], "sentence_id": r["sentence_id"],
                    "verdict": verdicts.get(c["claim_id"], "UNKNOWN"), "evidence_complete": True}
                   for r in records for c in r["claims"]]
    issues = [{"claim_id": a["summary_claim_id"], "sentence_id": a["sentence_id"], "type": "factual",
               "explanation": "checked", "severity": "serious" if a["verdict"] == "CONTRADICTED" else "unresolved"}
              for a in assessments if a["verdict"] != "SUPPORTED"]
    facts = {"assessments": assessments, "issues": issues,
             "status": "FAIL" if any(i["severity"] == "serious" for i in issues) else "UNKNOWN" if issues else "PASS"}
    views = check_viewpoints(graph, "Obama", records, facts)
    return {"summary_claims": records, "factual_results": facts, "representation_results":
            deepcopy(REPRESENTATION_NOT_RUN), "viewpoint_results": views, "viewpoint_issues": views["issues"]}


def draft_records():
    """S01 is the only negative sentence; nothing reports the positive viewpoint."""
    return [record("S01", [("negative", ["Obama_005"])]),
            record("S02", [("neutral", ["Obama_009"]), ("neutral", ["Obama_010"])]),
            record("S03", [("neutral", ["Obama_009"])]),
            record("S04", [("neutral", ["Obama_009"])]),
            record("S05", [("neutral", ["Obama_009"])])]


SUPPORTED_DRAFT = {"S01_C01": "SUPPORTED", "S02_C01": "SUPPORTED", "S03_C01": "SUPPORTED",
                   "S04_C01": "SUPPORTED", "S05_C01": "SUPPORTED"}  # S02_C02 stays UNKNOWN


class ViewpointCheck(unittest.TestCase):
    def setUp(self):
        self.graph = topic_graph()

    def test_stances_come_from_the_tweet_claims(self):
        stances = topic_stances(self.graph, "Obama")
        self.assertEqual(10, len(stances))  # the claimless tweet counts in the total
        self.assertEqual({"positive"}, stances["Obama_001"])
        self.assertEqual(set(), stances["Obama_010"])

    def test_missing_viewpoint_fails(self):
        result = checked(self.graph, draft_records(), SUPPORTED_DRAFT)["viewpoint_results"]
        self.assertEqual(("FAIL", ["positive"]), (result["status"], result["missing"]))
        self.assertEqual({"negative": 0.4, "positive": 0.4}, result["share_by_stance"])
        self.assertEqual(["Obama_001", "Obama_002", "Obama_003", "Obama_004"], result["tweets_by_stance"]["positive"])
        self.assertIn("4 of the topic's 10 tweets", result["issues"][0]["explanation"])
        self.assertEqual("serious", result["issues"][0]["severity"])

    def test_a_contradicted_claim_reports_no_viewpoint(self):
        records = draft_records()
        result = checked(self.graph, records, {**SUPPORTED_DRAFT, "S01_C01": "CONTRADICTED"})["viewpoint_results"]
        self.assertEqual(["negative", "positive"], result["missing"])
        unknown = checked(self.graph, records, {**SUPPORTED_DRAFT, "S01_C01": "UNKNOWN"})["viewpoint_results"]
        self.assertEqual((["S01_C01"], []), (unknown["reported"]["negative"], unknown["supported"]["negative"]))

    def test_an_unextracted_sentence_makes_it_unknown_not_missing(self):
        records = draft_records()
        records[4] = {**records[4], "status": "unclear", "claims": []}
        result = checked(self.graph, records, SUPPORTED_DRAFT)["viewpoint_results"]
        self.assertEqual(("UNKNOWN", [], ["positive"], []),
                         (result["status"], result["missing"], result["uncertain"], result["issues"]))

    def test_a_rare_viewpoint_is_not_required(self):
        result = check_viewpoints(self.graph, "Obama", draft_records(),
                                  checked(self.graph, draft_records(), SUPPORTED_DRAFT)["factual_results"], min_share=0.5)
        self.assertEqual(("PASS", []), (result["status"], result["missing"]))
        with self.assertRaises(ValueError):
            check_viewpoints(self.graph, "Obama", draft_records(), {}, min_share=0)

    def test_reads_no_label(self):
        result = checked(topic_graph(labelled=True), draft_records(), SUPPORTED_DRAFT)["viewpoint_results"]
        self.assertIsNone(LABEL_LEAK.search(json.dumps(result)))
        self.assertEqual(result, checked(self.graph, draft_records(), SUPPORTED_DRAFT)["viewpoint_results"])


class ViewpointRepairTask(unittest.TestCase):
    def setUp(self):
        self.graph = topic_graph()

    def test_targets_the_least_supported_sentence_never_the_only_carrier(self):
        state = checked(self.graph, draft_records(), {**SUPPORTED_DRAFT, "S01_C01": "UNKNOWN"})
        # S01 has no SUPPORTED claim but is the only negative sentence; S02 has an UNKNOWN claim.
        task = viewpoint_task(state["summary_claims"], state["factual_results"], state["viewpoint_results"])
        self.assertEqual(("viewpoint", "positive", "S02"), (task["kind"], task["stance"], task["sentence_id"]))
        self.assertEqual(["Obama_009", "Obama_010"], task["keep_ids"])
        records = draft_records()
        records[2] = {**records[2], "status": "no_claim", "claims": []}
        state = checked(self.graph, records, SUPPORTED_DRAFT)
        self.assertEqual("S03", viewpoint_task(state["summary_claims"], state["factual_results"],
                                               state["viewpoint_results"])["sentence_id"])

    def test_retrieval_keeps_the_sentence_tweets_and_adds_the_viewpoint(self):
        state = checked(self.graph, draft_records(), SUPPORTED_DRAFT)
        retrieved = retrieve_revision_evidence(self.graph, "Obama", state["summary_claims"], state["factual_results"],
                                               deepcopy(REPRESENTATION_NOT_RUN), embed, top_k=2,
                                               viewpoint_results=state["viewpoint_results"])
        self.assertEqual("ready", retrieved["status"])
        [task] = retrieved["tasks"]
        self.assertEqual(["Obama_009", "Obama_010"], task["evidence_ids"][:2])
        self.assertEqual(2, len(task["viewpoint_ids"]))
        self.assertTrue(set(task["viewpoint_ids"]) <= {"Obama_001", "Obama_002", "Obama_003", "Obama_004"})
        self.assertEqual(set(task["evidence_ids"]), {t["tweet_id"] for t in retrieved["tweets"]})
        self.assertIsNone(LABEL_LEAK.search(json.dumps(retrieved)))

    def test_no_task_when_the_check_did_not_fail(self):
        state = checked(self.graph, draft_records(), SUPPORTED_DRAFT)
        for views in (None, VIEWPOINT_NOT_RUN, {**state["viewpoint_results"], "status": "UNKNOWN"}):
            retrieved = retrieve_revision_evidence(self.graph, "Obama", state["summary_claims"], state["factual_results"],
                                                   deepcopy(REPRESENTATION_NOT_RUN), embed, viewpoint_results=views)
            self.assertEqual(([], "insufficient"), (retrieved["tasks"], retrieved["status"]))


class ViewpointGate(unittest.TestCase):
    def setUp(self):
        self.graph = topic_graph()
        self.before = checked(self.graph, draft_records(), SUPPORTED_DRAFT)

    def after(self, sentence_id, claims, verdicts):
        records = deepcopy(draft_records())
        index = int(sentence_id[1:]) - 1
        records[index] = record(sentence_id, claims)
        verdicts = {**{k: v for k, v in SUPPORTED_DRAFT.items() if not k.startswith(sentence_id)}, **verdicts}
        return checked(self.graph, records, verdicts), [{"sentence_id": sentence_id}]

    def test_a_supported_added_viewpoint_is_accepted(self):
        after, changes = self.after("S02", [("neutral", ["Obama_009"]), ("positive", ["Obama_001"])],
                                    {"S02_C01": "SUPPORTED", "S02_C02": "SUPPORTED"})
        self.assertEqual((True, "measured_improvement"), compare_revision(self.before, after, changes))

    def test_an_unsupported_added_viewpoint_is_rejected(self):
        after, changes = self.after("S02", [("neutral", ["Obama_009"]), ("positive", ["Obama_001"])],
                                    {"S02_C01": "SUPPORTED"})
        self.assertEqual((False, "viewpoint_not_supported"), compare_revision(self.before, after, changes))

    def test_losing_a_viewpoint_is_rejected(self):
        after, changes = self.after("S01", [("neutral", ["Obama_009"]), ("positive", ["Obama_001"])],
                                    {"S01_C01": "SUPPORTED", "S01_C02": "SUPPORTED"})
        self.assertEqual((False, "viewpoint_regression"), compare_revision(self.before, after, changes))

    def test_a_still_missing_viewpoint_is_rejected_and_retried(self):
        after, changes = self.after("S02", [("neutral", ["Obama_009"]), ("neutral", ["Obama_010"])],
                                    {"S02_C01": "SUPPORTED", "S02_C02": "SUPPORTED"})
        self.assertEqual((False, "viewpoint_still_missing"), compare_revision(self.before, after, changes))
        # Without the viewpoint check the same valid rewrite simply brings no measured gain.
        off = [{k: v for k, v in s.items() if k != "viewpoint_results"} for s in (self.before, after)]
        self.assertEqual((False, "no_improvement"), compare_revision(*off, changes))


class ViewpointReviser(unittest.TestCase):
    SUMMARY = {"sentences": [{"text": f"Sentence {n} summarizes a separate part of the discussion.",
                              "source_ids": ["Obama_009"]} for n in range(1, 6)]}

    def retrieved(self):
        tweets = [{"tweet_id": key, "quote": TEXTS[int(key[-3:]) - 1], "source_file": "tweets.csv",
                   "source_line": int(key[-3:])} for key in ("Obama_009", "Obama_001")]
        return {"status": "ready", "tweets": tweets, "tasks": [{
            "task_id": "viewpoint:positive", "kind": "viewpoint", "sentence_id": "S05", "stance": "positive",
            "query": "Sentence 5", "preferred_ids": ["Obama_001"], "keep_ids": ["Obama_009"],
            "evidence_ids": ["Obama_009", "Obama_001"], "viewpoint_ids": ["Obama_001"], "explanation": "add it"}]}

    @staticmethod
    def reply(ids):
        quotes = {"Obama_009": "speak in Chicago", "Obama_001": "hopeful and inspiring speech"}
        return {"output": [{"content": [{"text": json.dumps({"replacements": [{
            "sentence_id": "S05", "text": "Some users announce an upcoming Chicago appearance, while others "
                                          "praise a recent address by the president as uplifting.",
            "source_ids": ids, "evidence": [{"tweet_id": i, "quote": quotes[i]} for i in ids]}]})}]}], "usage": {}}

    def test_the_replacement_must_cite_a_tweet_with_the_viewpoint(self):
        client = Mock()
        client.chat.side_effect = [self.reply(["Obama_009"]), self.reply(["Obama_009", "Obama_001"])]
        result = revise_summary(self.SUMMARY, [], self.retrieved(), client)
        self.assertEqual("candidate", result["status"])
        self.assertIn("expressing the positive viewpoint: Obama_001", result["attempt_errors"][0])
        self.assertEqual(REVISION_INSTRUCTIONS, client.chat.call_args_list[0].args[0]["messages"][0]["content"])


class ViewpointAgent(unittest.TestCase):
    def test_off_by_default(self):
        self.assertNotIn("viewpoint_check", set(build_summary_agent(Graph(), Mock()).get_graph().nodes))
        nodes = build_summary_agent(Graph(), Mock(), viewpoint_check=True).get_graph()
        self.assertIn("viewpoint_check", set(nodes.nodes))
        self.assertIn(("factual_check", "viewpoint_check"), {(e.source, e.target) for e in nodes.edges})
        self.assertIn(("viewpoint_check", "route_checks"), {(e.source, e.target) for e in nodes.edges})
        for bad in ({"viewpoint_check": 1}, {"min_viewpoint_share": 0}, {"min_viewpoint_share": 1.5}):
            with self.assertRaises(ValueError):
                build_summary_agent(Graph(), Mock(), **bad)

    def run_recorded(self, name, **options):
        run, client = recorded(name), RecordingQwen()
        result = run_summary_agent(label_free_graph(graph_path(name)), run["initial_summary"], topic=run["topic"],
                                   prompt=run["prompt"], client=client, embedder=embed,
                                   check_cache=seed_check_cache(run["initial_summary"], first_check(name)), **options)
        return result, client

    def test_recorded_summary_missing_the_positive_viewpoint(self):
        # Obama P1 s101: 33 of 90 tweets hold a positive claim, the summary none.
        result, client = self.run_recorded("Obama_prompt1_seed101", viewpoint_check=True)
        views = result["iterations"][0]["viewpoint_results"]
        self.assertEqual(("FAIL", ["positive"]), (views["status"], views["missing"]))
        task = result["retrieval_history"][0]["tasks"][0]
        self.assertEqual(("viewpoint", "positive"), (task["kind"], task["stance"]))
        self.assertTrue(task["viewpoint_ids"] and set(task["viewpoint_ids"]) <= set(views["tweets_by_stance"]["positive"]))
        revisions = [p for p in client.payloads if "replacements" in json.dumps(p["format"])]
        self.assertTrue(revisions)
        self.assertTrue(all(p["messages"][0]["content"].startswith(REVISER_PROFILE) for p in revisions))
        self.assertTrue(all(LABEL_LEAK.search(json.dumps(p, ensure_ascii=False)) is None for p in client.payloads))
        self.assertNotIn("accepted", [r["status"] for r in result["revisions"]])
        self.assertEqual("stopped_with_issues", result["status"])

    def test_recorded_default_run_is_unchanged(self):
        result, client = self.run_recorded("Obama_prompt1_seed101")
        self.assertEqual("NOT_RUN", result["viewpoint_results"]["status"])
        self.assertEqual(([], [], "completed_with_uncertainty"),
                         (client.payloads, result["revisions"], result["status"]))

    def test_contradictions_are_repaired_first(self):
        result, _ = self.run_recorded("Superbowl_prompt1_seed202", viewpoint_check=True, max_revisions=1)
        self.assertEqual({"factual"}, {t["kind"] for t in result["retrieval_history"][0]["tasks"]})


if __name__ == "__main__":
    unittest.main()
