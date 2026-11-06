"""The label-free proportion feedback layer.

Qwen reads every tweet of the topic and the summary and judges whether the
summary represents all the tweets in proportion. No dialect label is read or
named. Qwen states an under-represented opinion in a few words but names no
tweets: the agent finds the topic's tweets closest to it (wherever they are in
the list) and, if a SUPPORTED summary claim already quotes one, the opinion is
already reported and nothing is rewritten. Feedback naming no opinion is
UNKNOWN, which never triggers a revision; contradictions are repaired first;
and the gate adds the opinion only when a SUPPORTED claim of the rewritten
sentence quotes one of the retrieved tweets. Off by default.

    python -m unittest tests.test_proportion_check -v
"""

from copy import deepcopy
import json
import re
import unittest
from unittest.mock import Mock

from rdflib import Graph

from proportion_check import PROPORTION_INSTRUCTIONS, PROPORTION_NOT_RUN, check_proportion, verify_feedback
from revision_retrieval import closest_topic_tweets, proportion_task, retrieve_revision_evidence
from run_final_agent import LABEL_LEAK, label_free_graph
from summary_agent import REPRESENTATION_NOT_RUN, build_summary_agent, compare_revision, run_summary_agent, \
    seed_check_cache
from summary_revision import REVISER_PROFILE, revise_summary
from tests import test_viewpoint_check as viewpoint_tests
from tests.test_label_free import embed, first_check
from tests.test_real_pieces import graph_path, recorded
from tests.test_viewpoint_check import SUPPORTED_DRAFT, TEXTS, checked, draft_records, record, topic_graph
from viewpoint_check import VIEWPOINT_NOT_RUN


SUMMARY = {"sentences": [{"text": f"Sentence {n} summarizes a separate part of the discussion.",
                          "source_ids": ["Obama_009"]} for n in range(1, 6)]}
POSITIVE = ["Obama_001", "Obama_002", "Obama_003", "Obama_004"]


def feedback(proportional=False, viewpoint="praise for the president's speeches and policies", sentence="S02",
             assessment="Most sentences report criticism; the praise is missing."):
    return json.dumps({"assessment": assessment, "proportional": proportional,
                       "under_represented_viewpoint": "" if proportional else viewpoint,
                       "sentence_id": "none" if proportional else sentence})


def reply(text):
    return {"output": [{"content": [{"text": text}]}], "usage": {}}


class Critic:
    """Answers the proportion feedback request; every other request gets an invalid answer."""

    def __init__(self, *answers):
        self.answers, self.payloads = list(answers), []

    def chat(self, payload):
        self.payloads.append(payload)
        if "proportional" in payload["format"]["properties"]:
            return reply(self.answers.pop(0) if len(self.answers) > 1 else self.answers[0])
        return reply("{}")

    def feedback_requests(self):
        return [p for p in self.payloads if "proportional" in p["format"]["properties"]]


class ProportionCheck(unittest.TestCase):
    def setUp(self):
        self.graph = topic_graph()

    def test_proportional_summary_passes(self):
        result = check_proportion(self.graph, "Obama", SUMMARY, Critic(feedback(proportional=True)))
        self.assertEqual(("PASS", True, [], 1), (result["status"], result["proportional"], result["issues"],
                                                 result["model_call_count"]))

    def test_a_named_opinion_fails(self):
        result = check_proportion(self.graph, "Obama", SUMMARY, Critic(feedback()))
        self.assertEqual(("FAIL", "S02", {"viewpoint": "praise for the president's speeches and policies"}),
                         (result["status"], result["sentence_id"], result["under_represented"]))
        [issue] = result["issues"]
        self.assertEqual(("disproportionate_summary", "serious"), (issue["type"], issue["severity"]))

    def test_no_named_opinion_stays_unknown(self):
        result = check_proportion(self.graph, "Obama", SUMMARY, Critic(feedback(viewpoint="  ")))
        self.assertEqual(("UNKNOWN", "no_viewpoint_named", []), (result["status"], result["reason"], result["issues"]))

    def test_a_long_description_is_repaired(self):
        client = Critic(feedback(viewpoint=" ".join(["absurdity"] * 16)), feedback())
        result = check_proportion(self.graph, "Obama", SUMMARY, client)
        self.assertEqual(("FAIL", 2), (result["status"], result["model_call_count"]))
        self.assertIn("at most 15 words", client.payloads[1]["messages"][-1]["content"])

    def test_invalid_feedback_is_repaired_once_then_unknown(self):
        client = Critic("{}", "{}")
        result = check_proportion(self.graph, "Obama", SUMMARY, client)
        self.assertEqual(("UNKNOWN", "invalid_feedback", 2), (result["status"], result["reason"],
                                                               result["model_call_count"]))
        self.assertIn("Rejected", client.payloads[1]["messages"][-1]["content"])

    def test_a_named_group_is_rejected_and_never_echoed(self):
        client = Critic(feedback(viewpoint="what Hispanic users think of the speech"), feedback())
        result = check_proportion(self.graph, "Obama", SUMMARY, client)
        self.assertEqual("FAIL", result["status"])
        self.assertNotIn("Hispanic", json.dumps(client.payloads[1]))
        self.assertIn("never name a group", client.payloads[1]["messages"][-1]["content"])

    def test_request_holds_every_tweet_and_no_label(self):
        client = Critic(feedback())
        check_proportion(topic_graph(labelled=True), "Obama", SUMMARY, client)
        [payload] = client.payloads
        body = json.loads(payload["messages"][1]["content"])
        self.assertEqual([f"Obama_{n:03d}" for n in range(1, 11)], [t["id"] for t in body["tweets"]])
        self.assertEqual(TEXTS, [t["text"] for t in body["tweets"]])
        self.assertEqual({"sentence_id", "text"}, set(body["summary"][0]))  # no citations: content is judged
        self.assertNotIn("tweet_ids", payload["format"]["properties"])  # Qwen names no tweets
        self.assertEqual(PROPORTION_INSTRUCTIONS, payload["messages"][0]["content"])
        self.assertIsNone(LABEL_LEAK.search(json.dumps(payload, ensure_ascii=False)))
        self.assertIn("Every tweet counts equally", PROPORTION_INSTRUCTIONS)

    def test_an_identical_summary_reuses_its_judgment(self):
        client, cache = Critic(feedback()), {}
        first = check_proportion(self.graph, "Obama", SUMMARY, client, cache=cache)
        again = check_proportion(self.graph, "Obama", deepcopy(SUMMARY), client, cache=cache)
        self.assertEqual((1, 0, True), (len(client.payloads), again["model_call_count"], again["reused"]))
        self.assertEqual(first["under_represented"], again["under_represented"])


def failed_feedback(sentence_id="S02"):
    return {**deepcopy(PROPORTION_NOT_RUN), "status": "FAIL", "proportional": False, "topic_tweet_count": 10,
            "under_represented": {"viewpoint": "praise for the president"}, "sentence_id": sentence_id,
            "issues": [{"type": "disproportionate_summary", "severity": "serious",
                        "explanation": "praise is under-represented"}]}


class ProportionRepairTask(unittest.TestCase):
    def setUp(self):
        self.graph = topic_graph()
        self.state = checked(self.graph, draft_records(), SUPPORTED_DRAFT)

    def test_targets_the_sentence_the_feedback_chose(self):
        task = proportion_task(self.state["summary_claims"], self.state["factual_results"], failed_feedback("S04"))
        self.assertEqual(("proportion", "S04", "praise for the president", [], ["Obama_009"]),
                         (task["kind"], task["sentence_id"], task["query"], task["preferred_ids"], task["keep_ids"]))
        self.assertIn("do not copy this description", task["explanation"])
        # Without a usable choice the least supported sentence is rewritten (S02 holds an UNKNOWN claim).
        fallback = proportion_task(self.state["summary_claims"], self.state["factual_results"], failed_feedback(None))
        self.assertEqual("S02", fallback["sentence_id"])

    def test_retrieval_finds_the_closest_tweets_then_keeps_the_sentence_tweets(self):
        retrieved = retrieve_revision_evidence(self.graph, "Obama", self.state["summary_claims"],
                                               self.state["factual_results"], deepcopy(REPRESENTATION_NOT_RUN), embed,
                                               top_k=2, proportion_results=failed_feedback("S02"))
        self.assertEqual("ready", retrieved["status"])
        [task] = retrieved["tasks"]
        # The closest tweets of the whole topic, exactly as the verification ranks them.
        closest, calls = closest_topic_tweets(self.graph, "Obama", "praise for the president", embed, top_k=2)
        self.assertEqual(([key for key, _ in closest], 1), (task["proportion_ids"], calls))
        self.assertEqual(task["proportion_ids"], task["preferred_ids"])
        self.assertEqual(task["proportion_ids"], task["evidence_ids"][:2])
        own = {"Obama_009", "Obama_010"} - set(task["proportion_ids"])
        self.assertEqual(sorted(own), sorted(task["evidence_ids"][2:]))
        self.assertIsNone(LABEL_LEAK.search(json.dumps(retrieved)))

    def test_contradictions_come_first_and_other_statuses_do_nothing(self):
        state = checked(self.graph, draft_records(), {**SUPPORTED_DRAFT, "S01_C01": "CONTRADICTED"})
        retrieved = retrieve_revision_evidence(self.graph, "Obama", state["summary_claims"], state["factual_results"],
                                               deepcopy(REPRESENTATION_NOT_RUN), embed,
                                               proportion_results=failed_feedback())
        self.assertEqual({"factual"}, {t["kind"] for t in retrieved["tasks"]})
        for result in (None, PROPORTION_NOT_RUN, {**failed_feedback(), "status": "UNKNOWN"}):
            retrieved = retrieve_revision_evidence(self.graph, "Obama", self.state["summary_claims"],
                                                   self.state["factual_results"], deepcopy(REPRESENTATION_NOT_RUN),
                                                   embed, proportion_results=result)
            self.assertEqual([], retrieved["tasks"])


class ProportionVerification(unittest.TestCase):
    """Qwen proposes, the verified evidence decides."""

    def facts(self, verdict, reference):
        return {"assessments": [{"summary_claim_id": "S02_C01", "sentence_id": "S02", "verdict": verdict,
                                 "evidence_complete": True,
                                 "references": [{"evidence_id": reference, "quote": "x"}]}]}

    CLOSEST = [("Obama_001", 0.71), ("Obama_003", 0.65)]

    def test_an_opinion_a_supported_claim_already_quotes_is_already_reported(self):
        for reference in ("http://example.org/run/tweet/Obama_003", "http://example.org/run/claim/Obama_001_C02"):
            result = verify_feedback(failed_feedback(), self.facts("SUPPORTED", reference), self.CLOSEST)
            self.assertEqual(("UNKNOWN", "already_reported", [], ["S02_C01"]),
                             (result["status"], result["reason"], result["issues"], result["reported_by"]))

    def test_otherwise_the_finding_stands(self):
        for verdict, reference in (("SUPPORTED", "http://example.org/run/tweet/Obama_009"),
                                   ("UNKNOWN", "http://example.org/run/tweet/Obama_001")):
            result = verify_feedback(failed_feedback(), self.facts(verdict, reference), self.CLOSEST)
            self.assertEqual("FAIL", result["status"])
            self.assertEqual(["Obama_001", "Obama_003"], [row["tweet_id"] for row in result["closest_tweets"]])
        passed = {**failed_feedback(), "status": "PASS"}
        self.assertIs(passed, verify_feedback(passed, self.facts("SUPPORTED", "x/tweet/Obama_001"), self.CLOSEST))


# The gate tests isolate the proportion rule: the viewpoint check is off.
WITHOUT_VIEWPOINTS = {"viewpoint_results": deepcopy(VIEWPOINT_NOT_RUN), "viewpoint_issues": []}


class ProportionGate(unittest.TestCase):
    def setUp(self):
        self.graph = topic_graph()
        self.before = {**checked(self.graph, draft_records(), SUPPORTED_DRAFT), **WITHOUT_VIEWPOINTS,
                       "proportion_results": failed_feedback()}

    def after(self, reference):
        records = deepcopy(draft_records())
        records[1] = record("S02", [("neutral", ["Obama_009"]), ("positive", ["Obama_001"])])
        state = {**checked(self.graph, records, {**SUPPORTED_DRAFT, "S02_C01": "SUPPORTED", "S02_C02": "SUPPORTED"}),
                 **WITHOUT_VIEWPOINTS}
        for assessment in state["factual_results"]["assessments"]:
            assessment["references"] = [{"evidence_id": "http://example.org/run/tweet/Obama_009", "quote": "x"}]
            if assessment["summary_claim_id"] == "S02_C02":
                assessment["references"] = [{"evidence_id": reference, "quote": "x"}]
        return state, [{"sentence_id": "S02"}]

    def test_a_supported_claim_quoting_a_retrieved_tweet_is_accepted(self):
        for reference in ("http://example.org/run/tweet/Obama_001", "http://example.org/run/claim/Obama_003_C01"):
            after, changes = self.after(reference)
            self.assertEqual((True, "measured_improvement"), compare_revision(self.before, after, changes, POSITIVE))

    def test_new_citations_on_an_unchanged_text_add_nothing(self):
        # The third live trial: attempt 2 only added citations to an unchanged sentence.
        after, _ = self.after("http://example.org/run/tweet/Obama_001")
        same = {"text": "Some users praise the president.", "source_ids": ["Obama_009"]}
        changes = [{"sentence_id": "S02", "before": same, "after": {**same, "source_ids": ["Obama_009", "Obama_001"]}}]
        self.assertEqual((False, "proportion_not_supported"), compare_revision(self.before, after, changes, POSITIVE))
        changes[0]["after"] = {**changes[0]["after"], "text": "Some users praise the president's speeches."}
        self.assertEqual((True, "measured_improvement"), compare_revision(self.before, after, changes, POSITIVE))

    def test_support_from_other_tweets_is_rejected_and_retried(self):
        after, changes = self.after("http://example.org/run/tweet/Obama_009")
        self.assertEqual((False, "proportion_not_supported"), compare_revision(self.before, after, changes, POSITIVE))
        # Without the feedback layer the same rewrite simply brings no measured gain.
        off = {k: v for k, v in self.before.items() if k != "proportion_results"}
        self.assertEqual((False, "no_improvement"), compare_revision(off, after, changes, POSITIVE))


class ProportionReviser(unittest.TestCase):
    def test_the_replacement_must_cite_a_retrieved_tweet(self):
        retrieved = viewpoint_tests.ViewpointReviser().retrieved()
        retrieved["tasks"] = [{"task_id": "proportion", "kind": "proportion", "sentence_id": "S05",
                               "viewpoint": "praise", "query": "praise", "preferred_ids": ["Obama_001"],
                               "keep_ids": ["Obama_009"], "evidence_ids": ["Obama_009", "Obama_001"],
                               "proportion_ids": ["Obama_001"], "explanation": "add it"}]
        client = Mock()
        client.chat.side_effect = [viewpoint_tests.ViewpointReviser.reply(["Obama_009"]),
                                   viewpoint_tests.ViewpointReviser.reply(["Obama_009", "Obama_001"])]
        result = revise_summary(SUMMARY, [], retrieved, client)
        self.assertEqual("candidate", result["status"])
        self.assertIn("under-represented viewpoint: Obama_001", result["attempt_errors"][0])

    def test_the_replacement_must_change_the_text(self):
        retrieved = viewpoint_tests.ViewpointReviser().retrieved()
        retrieved["tasks"] = [{"task_id": "proportion", "kind": "proportion", "sentence_id": "S05",
                               "viewpoint": "praise", "query": "praise", "preferred_ids": ["Obama_001"],
                               "keep_ids": ["Obama_009"], "evidence_ids": ["Obama_009", "Obama_001"],
                               "proportion_ids": ["Obama_001"], "explanation": "add it"}]
        reply = viewpoint_tests.ViewpointReviser.reply(["Obama_009", "Obama_001"])
        text = json.loads(reply["output"][0]["content"][0]["text"])["replacements"][0]["text"]
        summary = deepcopy(SUMMARY)
        summary["sentences"][4] = {"text": text, "source_ids": ["Obama_009"]}
        client = Mock()
        client.chat.side_effect = [reply, reply]
        result = revise_summary(summary, [], retrieved, client)
        self.assertEqual(("rejected", "invalid_revision"), (result["status"], result["reason"]))
        self.assertIn("new citations alone change nothing", result["attempt_errors"][0])


class ProportionAgent(unittest.TestCase):
    def test_off_by_default(self):
        self.assertNotIn("proportion_check", set(build_summary_agent(Graph(), Mock()).get_graph().nodes))
        graph = build_summary_agent(Graph(), Mock(), proportion_check=True).get_graph()
        edges = {(e.source, e.target) for e in graph.edges}
        self.assertTrue({("factual_check", "proportion_check"), ("proportion_check", "route_checks")} <= edges)
        both = build_summary_agent(Graph(), Mock(), proportion_check=True, viewpoint_check=True).get_graph()
        self.assertIn(("viewpoint_check", "proportion_check"), {(e.source, e.target) for e in both.edges})
        with self.assertRaises(ValueError):
            build_summary_agent(Graph(), Mock(), proportion_check=1)

    @staticmethod
    def devil_embed(texts):
        """Tweets naming the devil sit together, every other tweet elsewhere."""
        return [[1.0, 0.01] if re.search(r"devil|satan|anti-?christ", text, re.I) else [0.01, 1.0] for text in texts]

    def run_recorded(self, name, client, **options):
        run = recorded(name)
        return run_summary_agent(label_free_graph(graph_path(name)), run["initial_summary"], topic=run["topic"],
                                 prompt=run["prompt"], client=client, embedder=self.devil_embed,
                                 check_cache=seed_check_cache(run["initial_summary"], first_check(name), any_checker=True), **options)

    def test_recorded_opinion_already_reported_is_not_rewritten(self):
        # The second live trial: Qwen flagged the devil comparisons, which the verified
        # claim S02_C02 already reports, quoting Obama_008, Obama_044 and Obama_061.
        client = Critic(feedback(viewpoint="Obama is frequently compared to the devil, Satan, or the Anti-Christ."))
        result = self.run_recorded("Obama_prompt1_seed101", client, proportion_check=True)
        found = result["proportion_results"]
        self.assertEqual(("UNKNOWN", "already_reported", ["S02_C02"]),
                         (found["status"], found["reason"], found["reported_by"]))
        self.assertEqual(["Obama_008", "Obama_044", "Obama_061"], [row["tweet_id"] for row in found["closest_tweets"]])
        self.assertEqual(([], [], "completed_with_uncertainty"),
                         (result["retrieval_history"], result["revisions"], result["status"]))
        self.assertIn("proportion_uncertainty", result["uncertainty_flags"])

    def test_recorded_run_repairs_the_named_opinion(self):
        # Obama P1 s101 has no contradiction and no verified claim quotes the praise.
        client = Critic(feedback(sentence="S05"))
        result = self.run_recorded("Obama_prompt1_seed101", client, proportion_check=True)
        self.assertEqual("FAIL", result["iterations"][0]["proportion_results"]["status"])
        self.assertEqual(90, len(json.loads(client.feedback_requests()[0]["messages"][1]["content"])["tweets"]))
        task = result["retrieval_history"][0]["tasks"][0]
        self.assertEqual(("proportion", "S05", ["Obama_001", "Obama_002", "Obama_003"]),
                         (task["kind"], task["sentence_id"], task["proportion_ids"]))
        revisions = [p for p in client.payloads if "replacements" in json.dumps(p["format"])]
        self.assertTrue(revisions)
        self.assertTrue(all(p["messages"][0]["content"].startswith(REVISER_PROFILE) for p in revisions))
        self.assertTrue(all(LABEL_LEAK.search(json.dumps(p, ensure_ascii=False)) is None for p in client.payloads))
        # The stub reviser's answers are invalid, so nothing is accepted and the finding remains.
        self.assertNotIn("accepted", [r["status"] for r in result["revisions"]])
        self.assertEqual("stopped_with_issues", result["status"])

    def test_a_proportional_judgment_changes_nothing(self):
        client = Critic(feedback(proportional=True))
        result = self.run_recorded("Obama_prompt1_seed101", client, proportion_check=True)
        self.assertEqual(("PASS", [], "completed_with_uncertainty"),
                         (result["proportion_results"]["status"], result["revisions"], result["status"]))
        self.assertEqual(1, len(client.payloads))

    def test_contradictions_are_repaired_first(self):
        result = self.run_recorded("Superbowl_prompt1_seed202", Critic(feedback()), proportion_check=True,
                                   max_revisions=1)
        self.assertEqual({"factual"}, {t["kind"] for t in result["retrieval_history"][0]["tasks"]})


if __name__ == "__main__":
    unittest.main()
