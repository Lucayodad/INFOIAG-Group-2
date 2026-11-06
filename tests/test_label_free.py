"""Dialect labels are evaluation-only: the agent must never read or send one.

Feedback on Intermediate Report 2: the dialect annotations may not be fed to
the LLM, and using them inside the agent uses test data for training. These
tests pin the label-free agent on the recorded graphs and first checks of the
18 reruns (results/agent_rerun): no group check, no label in its graph,
state or any model request, and revision only on contradicted claims.

    python -m unittest tests.test_label_free -v
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
import unittest.mock
from unittest.mock import Mock
import zlib

from rdflib import Graph
from rdflib.graph import ReadOnlyGraphAggregate

from data_loader import DIALECT_LABELS, load_tweets
from fact_check import compact_evidence
from ontology_tools import NS
from revision_retrieval import retrieve_revision_evidence, topic_tweets
from run_final_agent import LABEL_LEAK, LabelFreeClient, first_check as reused_check, label_free_graph
from summary_agent import REPRESENTATION_NOT_RUN, build_summary_agent, compare_revision, run_summary_agent, \
    seed_check_cache
from summary_revision import REVISION_INSTRUCTIONS, revise_summary
from tests.test_real_pieces import graph_path, recorded


ROOT = Path(__file__).resolve().parents[1]
CONTRADICTED = ("McDonalds_prompt1_seed202", "Obama_prompt1_seed202", "Superbowl_prompt1_seed202")
ONE_PER_TOPIC = ("McDonalds_prompt1_seed101", "Obama_prompt1_seed101", "Superbowl_prompt1_seed101")


def embed(texts):
    """Deterministic stand-in for MiniLM: the ranking logic, not the model, is tested."""
    return [[1.0 + (zlib.crc32(f"{text}|{i}".encode()) % 1000) / 1000 for i in range(8)] for text in texts]


class RecordingQwen:
    """Records every request; answers with an invalid revision so nothing is accepted."""

    def __init__(self):
        self.payloads = []

    def chat(self, payload):
        self.payloads.append(payload)
        return {"output": [{"content": [{"text": "{}"}]}], "usage": {}}


def first_check(name):
    return reused_check(recorded(name)["iterations"][0])


def no_group(state):
    state = deepcopy(state)
    state.update(representation_results=deepcopy(REPRESENTATION_NOT_RUN), representation_issues=[])
    return state


def dry_run(name, client):
    run = recorded(name)
    return run_summary_agent(label_free_graph(graph_path(name)), run["initial_summary"], topic=run["topic"],
                             prompt=run["prompt"], client=client, embedder=embed,
                             check_cache=seed_check_cache(run["initial_summary"], first_check(name), any_checker=True))


class LabelFreeAgent(unittest.TestCase):
    def test_default_agent_has_no_group_check(self):
        nodes = set(build_summary_agent(Graph(), Mock()).get_graph().nodes)
        self.assertNotIn("representation_check", nodes)
        self.assertTrue({"extract_summary_claims", "factual_check", "route_checks",
                         "retrieve_revision_evidence", "revise_summary", "finalize"} <= nodes)
        # There is no option to switch a label-based group check on.
        with self.assertRaises(TypeError):
            build_summary_agent(Graph(), Mock(), representation_metric="mnli")

    def test_agent_drops_labels_from_any_callers_graph(self):
        # Entry points that pass a labelled graph (run_pipeline, experiment_runner) are covered too.
        name = "Obama_prompt1_seed202"
        run, labelled = recorded(name), Graph().parse(graph_path(name), format="turtle")
        before, seen = set(labelled), []
        import summary_agent
        original = summary_agent.fact_check_summary

        def spy(graph, *args, **kwargs):
            seen.append(len(list(graph.triples((None, NS.dialectLabel, None)))))
            return original(graph, *args, **kwargs)
        with unittest.mock.patch("summary_agent.fact_check_summary", side_effect=spy):
            run_summary_agent(labelled, run["initial_summary"], topic=run["topic"], prompt=run["prompt"],
                              client=RecordingQwen(), embedder=embed, max_revisions=0,
                              check_cache=seed_check_cache(run["initial_summary"], first_check(name), any_checker=True))
        self.assertEqual([0], seen)
        self.assertEqual(before, set(labelled))  # the caller's graph is unchanged

    def test_demo_run_output_holds_no_label(self):
        # A labelled synthetic graph and the agent with its defaults.
        from fakes import fixture, mock_client
        graph, draft = fixture("factual")
        self.assertTrue(any(True for _ in graph.triples((None, NS.dialectLabel, None))))
        result = run_summary_agent(graph, draft, topic="Obama", prompt=draft["summary_prompt"], client=mock_client())
        rendered = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("dialect_label", rendered)
        # Everything the reviser and the checks worked from (the NOT_RUN note itself names the labels).
        self.assertIsNone(LABEL_LEAK.search(json.dumps(
            [result["revisions"], result["retrieval_history"],
             [{key: value for key, value in i.items() if key != "representation_results"} for i in result["iterations"]]],
            ensure_ascii=False)))
        self.assertEqual(["accepted"], [r["status"] for r in result["revisions"]])

    def test_agent_graph_has_no_label_and_the_file_is_unchanged(self):
        for name in ONE_PER_TOPIC:
            path = graph_path(name)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            graph = label_free_graph(path)
            self.assertEqual([], list(graph.triples((None, NS.dialectLabel, None))))
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
            labelled = Graph().parse(path, format="turtle")
            self.assertEqual(90, len(list(labelled.triples((None, NS.dialectLabel, None)))))
            self.assertEqual(len(labelled) - 90, len(graph))

    def test_label_free_tweets_are_the_same_tweets(self):
        for name in ONE_PER_TOPIC:
            topic = recorded(name)["topic"]
            dataset = {t.id: t.text for t in load_tweets(topic) if t.dialect_label in DIALECT_LABELS}
            tweets, problems = topic_tweets(ReadOnlyGraphAggregate([label_free_graph(graph_path(name))]), topic)
            self.assertEqual([], problems)
            self.assertEqual(dataset, {t["tweet_id"]: t["text"] for t in tweets})
            self.assertEqual(90, len(tweets))
            self.assertTrue(all(set(t) == {"tweet_id", "text", "source_file", "source_line"} for t in tweets))

    def test_seeded_first_checks_hold_no_label(self):
        for name in CONTRADICTED + ONE_PER_TOPIC:
            raw = recorded(name)["iterations"][0]
            self.assertIsNotNone(LABEL_LEAK.search(json.dumps(raw)))  # the recording holds labels
            cleaned = first_check(name)
            self.assertEqual({"summary_claims", "factual_results"}, set(cleaned))
            self.assertIsNone(LABEL_LEAK.search(json.dumps(cleaned, ensure_ascii=False)), name)
            self.assertEqual(raw["summary_claims"], cleaned["summary_claims"])
            self.assertEqual([a["verdict"] for a in raw["factual_results"]["assessments"]],
                             [a["verdict"] for a in cleaned["factual_results"]["assessments"]])
            seed_check_cache(recorded(name)["initial_summary"], cleaned, any_checker=True)

    def test_fact_checker_never_showed_qwen_a_label(self):
        # The recorded evidence carries labels; the view sent to Qwen must not.
        seen = 0
        for name in CONTRADICTED:
            for assessment in recorded(name)["iterations"][0]["factual_results"]["assessments"]:
                if "dialect_label" in json.dumps(assessment.get("evidence", {})):
                    seen += 1
                    view = compact_evidence(assessment)
                    self.assertIsNone(LABEL_LEAK.search(json.dumps(view["evidence_catalogue"], ensure_ascii=False)))
        self.assertGreater(seen, 0)

    def test_retrieval_and_reviser_prompt_are_label_free(self):
        for name in CONTRADICTED:
            run, checked = recorded(name), first_check(name)
            graph = ReadOnlyGraphAggregate([label_free_graph(graph_path(name))])
            retrieved = retrieve_revision_evidence(graph, run["topic"], checked["summary_claims"],
                                                   checked["factual_results"], embed)
            self.assertEqual("ready", retrieved["status"], name)
            self.assertEqual({"factual"}, {task["kind"] for task in retrieved["tasks"]})
            for task in retrieved["tasks"]:
                # The contradicted claim's own cited tweets come first.
                self.assertTrue(set(task["evidence_ids"]) <= set(task["preferred_ids"]) or
                                len(task["preferred_ids"]) < len(task["evidence_ids"]))
            client = RecordingQwen()
            issues = [{key: i[key] for key in ("type", "severity", "explanation")}
                      for i in checked["factual_results"]["issues"] if i["severity"] == "serious"]
            revise_summary(run["initial_summary"], issues, retrieved, client)
            self.assertEqual(2, len(client.payloads))  # proposal + one format repair
            for payload in client.payloads:
                self.assertIsNone(LABEL_LEAK.search(json.dumps(payload, ensure_ascii=False)), name)
        self.assertNotIn("dialect", REVISION_INSTRUCTIONS.lower())

    def test_factual_repair_keeps_the_sentences_own_citations(self):
        # Superbowl P1 s202: the uncontested claim ("indifference") was SUPPORTED by
        # tweets 051 and 084, outside the 3 most similar to the contradicted claim.
        name = "Superbowl_prompt1_seed202"
        run, checked = recorded(name), first_check(name)
        graph = ReadOnlyGraphAggregate([label_free_graph(graph_path(name))])
        retrieved = retrieve_revision_evidence(graph, run["topic"], checked["summary_claims"],
                                               checked["factual_results"], embed)
        task = retrieved["tasks"][0]
        cited = run["initial_summary"]["sentences"][int(task["sentence_id"][1:]) - 1]["source_ids"]
        self.assertEqual(set(cited), set(task["evidence_ids"]))
        self.assertTrue({"Superbowl_051", "Superbowl_084"} <= set(task["evidence_ids"]))
        self.assertLessEqual(len(retrieved["tweets"]), 9)

    def test_guard_refuses_labels_but_not_ordinary_words(self):
        backend = Mock()
        guard = LabelFreeClient(backend)
        for leak in ({"tweets": [{"dialect_label": None}]}, {"group": "White"}, {"g": "AA"}, {"text": "Dialect group"}):
            with self.assertRaises(RuntimeError):
                guard.chat(leak)
        guard.chat({"text": "The White House and the AAA hotline"})
        guard.embed(["Obama at the White House"], model="all-minilm")
        self.assertEqual(1, backend.chat.call_count)
        self.assertEqual(6, guard.checked_requests)

    def test_only_contradictions_trigger_revision(self):
        for name in ONE_PER_TOPIC:
            client = RecordingQwen()
            result = dry_run(name, client)
            # Seeded first check: the only model calls redo a recorded failed extraction.
            self.assertTrue(all(p["messages"][0]["content"].lstrip().startswith("Extract individual claims")
                                for p in client.payloads), name)
            failed = sum(r["status"] == "unclear" for r in first_check(name)["summary_claims"])
            self.assertEqual(2 * failed, len(client.payloads), name)
            self.assertEqual("NOT_RUN", result["representation_results"]["status"])
            self.assertEqual([], result["revisions"])
            self.assertIn(result["stop_reason"], ("uncertainty_only", "no_confirmed_serious_issues"))
        for name in CONTRADICTED:
            client = RecordingQwen()
            result = dry_run(name, client)
            self.assertEqual("confirmed_serious_issues", result["routing_decisions"][0]["reason"], name)
            self.assertEqual(3, len(result["revisions"]))  # invalid proposals retried up to the bound
            self.assertEqual(result["initial_summary"], result["current_summary"])
            for payload in client.payloads:
                self.assertIsNone(LABEL_LEAK.search(json.dumps(payload, ensure_ascii=False)))
            self.assertNotIn("dialect_label", json.dumps(result["retrieval_history"]))


class GateWithoutGroups(unittest.TestCase):
    """Summary-level rule: no new contradiction, and no more issues in total."""

    def setUp(self):
        self.before = no_group(first_check("Obama_prompt1_seed202"))
        contradiction = next(i for i in self.before["factual_results"]["issues"] if i["severity"] == "serious")
        self.sentence = contradiction["sentence_id"]
        self.changes = [{"sentence_id": self.sentence}]
        # The repaired sentence: one SUPPORTED claim, the contradiction now UNKNOWN.
        self.after = deepcopy(self.before)
        for assessment in self.after["factual_results"]["assessments"]:
            if assessment["sentence_id"] == self.sentence:
                assessment.update(verdict="SUPPORTED", evidence_complete=True)
        for issue in self.after["factual_results"]["issues"]:
            if issue["sentence_id"] == self.sentence and issue["severity"] == "serious":
                issue["severity"] = "unresolved"

    def test_contradiction_becoming_unknown_is_accepted(self):
        self.assertEqual((True, "measured_improvement"), compare_revision(self.before, self.after, self.changes))

    def test_a_new_unknown_elsewhere_is_a_regression(self):
        extra = deepcopy(next(i for i in self.after["factual_results"]["issues"] if i["severity"] == "unresolved"))
        extra["sentence_id"] = next(s for s in ("S01", "S02", "S03", "S04", "S05") if s != self.sentence)
        self.after["factual_results"]["issues"].append(extra)
        self.assertEqual((False, "factual_regression"), compare_revision(self.before, self.after, self.changes))

    def test_a_still_contradicted_sentence_is_rejected(self):
        self.assertEqual((False, "unsupported_revision"),
                         compare_revision(self.before, deepcopy(self.before), self.changes))

    def test_recorded_mcdonalds_p1_s202_fixes_are_accepted(self):
        path = ROOT / "results" / "agent_final" / "McDonalds_prompt1_seed202.json"
        if not path.exists():
            self.skipTest("local results/agent_final not available")
        run = json.loads(path.read_text(encoding="utf-8"))
        outcomes = [compare_revision(no_group(run["iterations"][0]), no_group(run["iterations"][r["checked_iteration"]]),
                                     r["changes"])[0] for r in run["revisions"]]
        self.assertEqual([False, True, True], outcomes)  # attempt 1 adds an issue; 2 and 3 remove the contradiction


if __name__ == "__main__":
    unittest.main()
