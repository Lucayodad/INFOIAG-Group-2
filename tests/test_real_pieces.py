"""Each agent piece against the real recorded runs, not synthetic fixtures.

The other test files check the logic on small mocks. These tests feed every
piece the recorded states, revision candidates and RDF graphs of the 18 agent
reruns in results/agent_rerun, and each pins a failure seen in those runs.
Tests marked live call the local Ollama and MNLI models (several minutes):

    python -m unittest tests.test_real_pieces -v
    INFOIAG_LIVE=1 python -m unittest tests.test_real_pieces -v   (PowerShell: $env:INFOIAG_LIVE=1)
"""

from copy import deepcopy
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import unittest
from unittest.mock import Mock
import zlib

from rdflib import Graph
from rdflib.graph import ReadOnlyGraphAggregate

from demo_revision_loop import fixture, mock_client
from representation_check import representation_check
from representation_tools import get_group_claims
from revision_retrieval import cosine, retrieve_revision_evidence
from summarizer import GenerationError, validate_abstraction
from equal_coverage import CachedScorer
from summary_agent import _metrics, _worst_coverage, compare_revision, run_summary_agent, seed_check_cache
from summary_claims import extract_sentence_with_retry
from summary_revision import revise_summary


ROOT = Path(__file__).resolve().parents[1]
RERUN = ROOT / "results" / "agent_rerun"
LIVE = os.environ.get("INFOIAG_LIVE") == "1"
live = unittest.skipUnless(LIVE, "set INFOIAG_LIVE=1 to call the local Ollama and MNLI models")


@lru_cache(maxsize=None)
def recorded(name):
    path = RERUN / f"{name}.json"
    if not path.exists():
        raise unittest.SkipTest(f"recorded rerun {name} is not available")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("agent_result", data)


def candidate(name, attempt=1):
    """(first check, recheck of the candidate, revision record) for one recorded attempt."""
    run = recorded(name)
    revision = next(r for r in run["revisions"] if r.get("attempt") == attempt and "checked_iteration" in r)
    return run["iterations"][0], run["iterations"][revision["checked_iteration"]], revision


@lru_cache(maxsize=None)
def graph(name):
    path = Path(recorded(name)["pipeline"]["graph"])
    path = path if path.is_absolute() else ROOT / path
    if not path.exists():
        raise unittest.SkipTest(f"graph {path} is not available")
    return ReadOnlyGraphAggregate([Graph().parse(path, format="turtle")])


def trials():
    return sorted(path.stem for path in RERUN.glob("*_prompt*_seed*.json"))


def is_copy(text, quotes):
    try:
        validate_abstraction(text, quotes)
    except GenerationError:
        return True
    return False


def bag_of_words(texts):
    """Deterministic stand-in for the embedding model."""
    vectors = []
    for text in texts:
        vector = [0.001] * 64
        for word in re.findall(r"\w+", text.casefold()):
            vector[zlib.crc32(word.encode()) % 64] += 1.0
        vectors.append(vector)
    return vectors


def response(text):
    return {"output": [{"content": [{"text": text}]}], "usage": {"input_tokens": 1, "output_tokens": 1}}


class AcceptanceGateOnRecordedCandidates(unittest.TestCase):
    """compare_revision on the rechecks the reruns actually produced."""

    def decide(self, name, attempt=1):
        before, after, revision = candidate(name, attempt)
        return compare_revision(before, after, revision["changes"]), _metrics(before), _metrics(after)

    def test_scenario_1_contradiction_removed_despite_a_leftover_unknown_phrase(self):
        # The old all-SUPPORTED gate rolled this back for "unnecessary or harmful".
        (accepted, reason), before, after = self.decide("Obama_prompt1_seed202")
        self.assertEqual((accepted, reason), (True, "measured_improvement"))
        self.assertEqual((before["serious_factual_count"], after["serious_factual_count"]), (1, 0))

    def test_scenario_2_group_fix_that_raises_the_worst_group_is_accepted(self):
        (accepted, _), before, after = self.decide("Obama_prompt2_seed202")
        self.assertTrue(accepted)
        self.assertGreater(_worst_coverage(after), _worst_coverage(before))

    def test_fix_that_lifts_every_group_is_accepted_although_the_gap_widens(self):
        # Four-fifths paradox: Hisp 0.47% -> 1.33% overtakes the others, so
        # 1 - min/max grows even though every group gained coverage.
        (accepted, _), before, after = self.decide("McDonalds_prompt1_seed303")
        self.assertTrue(accepted)
        self.assertTrue(all(after["coverage"][g] > before["coverage"][g] for g in before["coverage"]))
        self.assertGreater(after["disparity"], before["disparity"])

    def test_levelling_down_is_rejected_although_disparity_improves(self):
        # The old gate accepted this: disparity 0.25 -> 0.24 because every group lost coverage.
        (accepted, reason), before, after = self.decide("Superbowl_prompt2_seed101")
        self.assertEqual((accepted, reason), (False, "representation_regression_or_incomparable"))
        self.assertLess(after["disparity"], before["disparity"])
        self.assertTrue(all(after["coverage"][g] < before["coverage"][g] for g in before["coverage"]))

    def test_rewrite_without_any_supported_claim_is_rejected(self):
        # Same UNKNOWN count before and after, citations unrelated: not a repair.
        (accepted, reason), _, _ = self.decide("Obama_prompt1_seed101")
        self.assertEqual((accepted, reason), (False, "unsupported_revision"))

    def test_rewrite_that_adds_a_contradiction_is_rejected(self):
        (accepted, reason), _, _ = self.decide("Obama_prompt2_seed101")
        self.assertEqual((accepted, reason), (False, "unsupported_revision"))

    def test_rewrite_that_loses_its_only_supported_claim_is_rejected(self):
        (accepted, reason), _, _ = self.decide("McDonalds_prompt1_seed101")
        self.assertEqual((accepted, reason), (False, "unsupported_revision"))

    def test_replay_of_every_recorded_first_acceptance(self):
        """The first non-copy candidate each trial would accept, walking recorded attempts in order."""
        accepted = {}
        for name in trials():
            run = recorded(name)
            for revision in run["revisions"]:
                if "checked_iteration" not in revision:
                    continue
                ok, reason = compare_revision(run["iterations"][0], run["iterations"][revision["checked_iteration"]],
                                              revision["changes"])
                quotes = {row["tweet_id"]: row["quote"] for row in revision["evidence"]["tweets"]}
                if ok and not any(is_copy(c["after"]["text"], [quotes[i] for i in c["after"]["source_ids"]])
                                  for c in revision["changes"]):
                    accepted[name] = revision["attempt"]
                    break
                if reason == "no_improvement":
                    break
        self.assertEqual(accepted, {"McDonalds_prompt1_seed303": 1, "Obama_prompt1_seed202": 1,
                                    "Obama_prompt2_seed202": 1, "Superbowl_prompt2_seed303": 1})


class CopyGuardOnRecordedText(unittest.TestCase):
    def test_the_accepted_pasted_tweet_is_a_copy(self):
        _, _, revision = candidate("McDonalds_prompt2_seed101")
        change = revision["changes"][0]["after"]
        quotes = {row["tweet_id"]: row["quote"] for row in revision["evidence"]["tweets"]}
        self.assertTrue(is_copy(change["text"], [quotes[i] for i in change["source_ids"]]))

    def test_real_paraphrases_are_not_copies(self):
        for name in ("McDonalds_prompt1_seed303", "Superbowl_prompt2_seed101", "Obama_prompt2_seed202"):
            with self.subTest(trial=name):
                _, _, revision = candidate(name)
                quotes = {row["tweet_id"]: row["quote"] for row in revision["evidence"]["tweets"]}
                for change in revision["changes"]:
                    self.assertFalse(is_copy(change["after"]["text"], [quotes[i] for i in change["after"]["source_ids"]]))

    def test_reviser_rejects_the_recorded_paste_instead_of_proposing_it(self):
        _, _, revision = candidate("McDonalds_prompt2_seed101")
        client = Mock()
        client.chat.return_value = response(revision["raw_response"])
        proposal = revise_summary(revision["summary_before"], [], revision["evidence"], client)
        self.assertEqual((proposal["status"], proposal["reason"]), ("rejected", "invalid_revision"))
        self.assertIn("copies", proposal["error"])
        self.assertEqual(client.chat.call_count, 2)  # one repair request, then rejection
        self.assertIn("own words", client.chat.call_args_list[1].args[0]["messages"][-1]["content"])


class GroupCheckOnRecordedSummaries(unittest.TestCase):
    def test_a_failed_claim_extraction_no_longer_blanks_the_group_check(self):
        # 4 of 6 Superbowl reruns had representation UNKNOWN only because one
        # sentence's claim extraction failed; MNLI needs just the sentence text.
        name = "Superbowl_prompt1_seed101"
        state = recorded(name)["iterations"][0]
        self.assertEqual(state["representation_results"]["status"], "UNKNOWN")
        self.assertTrue(any(r["status"] == "unclear" for r in state["summary_claims"]))
        scorer = Mock()
        scorer.score_pairs.side_effect = lambda pairs: [0.05] * len(pairs)
        result = representation_check(graph(name), state["summary_claims"], Mock(), "Superbowl", 0.2,
                                      mnli_scorer=scorer)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["data_complete"])
        self.assertEqual(sum(len(call.args[0]) for call in scorer.score_pairs.call_args_list), 90 * 5)

    @live
    def test_mnli_reproduces_the_recorded_coverage(self):
        # The gate compares coverages across checks, so the metric must be deterministic.
        from equal_coverage import TransformersEntailmentScorer
        name = "Obama_prompt2_seed202"
        state = recorded(name)["iterations"][0]
        result = representation_check(graph(name), state["summary_claims"], Mock(), "Obama", 0.2,
                                      mnli_scorer=TransformersEntailmentScorer())
        for group, row in state["representation_results"]["coverage_by_group"].items():
            self.assertAlmostEqual(result["coverage_by_group"][group]["coverage"], row["coverage"], places=4)


class GroupRetrievalOnRecordedStates(unittest.TestCase):
    def group_tasks(self):
        rows = []
        for name in trials():
            run = recorded(name)
            revision = next((r for r in run["revisions"] if any(
                t["kind"] == "representation" for t in r["evidence"].get("tasks", []))), None)
            if revision:
                task = next(t for t in revision["evidence"]["tasks"] if t["kind"] == "representation")
                rows.append((name, run, revision, task))
        return rows

    def test_group_task_searches_the_groups_tweets_with_the_sentence_to_rewrite(self):
        name = "Obama_prompt2_seed202"
        state = recorded(name)["iterations"][0]
        result = retrieve_revision_evidence(graph(name), "Obama", state["summary_claims"], state["factual_results"],
                                            state["representation_results"], bag_of_words)
        task = next(t for t in result["tasks"] if t["kind"] == "representation")
        target = next(r["sentence"] for r in state["summary_claims"] if r["sentence_id"] == task["sentence_id"])
        self.assertEqual(task["query"], target)
        # The evidence is the group's tweets closest to that sentence, in order.
        group = get_group_claims(graph(name), "Obama", task["dialect_label"])["tweets"]
        query = bag_of_words([target])[0]
        ranked = sorted(group, key=lambda t: (-cosine(query, bag_of_words([t["text"]])[0]), t["tweet_id"]))
        self.assertEqual(task["evidence_ids"], [t["tweet_id"] for t in ranked[:len(task["evidence_ids"])]])

    @live
    def test_sentence_query_finds_group_tweets_on_the_sentence_topic(self):
        # Recorded evidence came from "Represent viewpoints from the X group."
        from ollama_backend import OllamaClient
        client = OllamaClient()
        better = 0
        rows = self.group_tasks()
        for name, run, revision, task in rows:
            state = run["iterations"][0]
            target = next(r["sentence"] for r in state["summary_claims"] if r["sentence_id"] == task["sentence_id"])
            new = retrieve_revision_evidence(graph(name), run["topic"], state["summary_claims"], state["factual_results"],
                                             state["representation_results"], lambda t: client.embed(t))
            new_task = next(t for t in new["tasks"] if t["kind"] == "representation")
            texts = {row["tweet_id"]: row["quote"] for row in new["tweets"] + revision["evidence"]["tweets"]}
            vectors = client.embed([target] + [texts[i] for i in task["evidence_ids"] + new_task["evidence_ids"]])
            sims = [cosine(vectors[0], v) for v in vectors[1:]]
            old, fresh = sims[:len(task["evidence_ids"])], sims[len(task["evidence_ids"]):]
            better += sum(fresh) / len(fresh) > sum(old) / len(old)
        self.assertGreaterEqual(better, len(rows) - 1, f"sentence query was more relevant in {better}/{len(rows)}")


class SummaryClaimExtraction(unittest.TestCase):
    SENTENCE = {"text": "There is a recurring theme of uncertainty regarding whether the game will happen.",
                "source_ids": ["Superbowl_001"]}

    def test_a_paraphrased_quote_gets_one_retry_with_its_error(self):
        bad = {"status": "extracted", "claims": [{"quote": "uncertainty about the game happening",
                                                   "claim": "Users are uncertain.", "target": "game", "stance": "unclear"}]}
        good = deepcopy(bad)
        good["claims"][0]["quote"] = "uncertainty regarding whether the game will happen"
        client = Mock()
        client.chat.side_effect = [response(json.dumps(bad)), response(json.dumps(good))]
        record, calls = extract_sentence_with_retry(self.SENTENCE, 4, client)
        self.assertEqual((record["status"], calls), ("extracted", 2))
        retry = client.chat.call_args_list[1].args[0]["messages"][-1]["content"]
        self.assertIn("quotation not found", retry)

    def test_a_second_failure_still_raises(self):
        client = Mock()
        client.chat.side_effect = [response("{}")] * 2
        with self.assertRaises(ValueError):
            extract_sentence_with_retry(self.SENTENCE, 4, client)
        self.assertEqual(client.chat.call_count, 2)

    @live
    def test_recorded_extraction_failures_are_retried_live(self):
        from ollama_backend import OllamaClient
        client = OllamaClient()
        outcomes = {}
        for name in trials():
            run = recorded(name)
            for error in run["iterations"][0]["extraction_errors"]:
                number = int(error["sentence_id"][1:])
                sentence = run["initial_summary"]["sentences"][number - 1]
                try:
                    record, calls = extract_sentence_with_retry(sentence, number, client)
                    outcomes[(name, error["sentence_id"])] = (record["status"], calls)
                except ValueError as exc:
                    outcomes[(name, error["sentence_id"])] = ("failed", str(exc)[:60])
        print("\nRecorded extraction failures, retried:", outcomes)
        self.assertTrue(outcomes)
        self.assertTrue(any(status == "extracted" for status, _ in outcomes.values()), outcomes)


class RetrySampling(unittest.TestCase):
    def test_only_a_retry_after_rollback_samples_and_each_retry_has_its_own_seed(self):
        def invented(data):
            data["replacements"][0]["text"] = "An invented quantity proves the speech was popular."
            return data
        graph_, draft, counts = fixture("both")
        client = mock_client(transform_revision=invented)
        result = run_summary_agent(graph_, draft, topic="Obama", prompt=draft["summary_prompt"], client=client,
                                   expected_group_counts=counts, disparity_threshold=0.25,
                                   representation_metric="internal")
        self.assertEqual([r["status"] for r in result["revisions"]], ["rolled_back"] * 3)
        options = [(call.args[0]["options"]["temperature"], call.args[0]["options"]["seed"])
                   for call in client.chat.call_args_list
                   if "replacements" in call.args[0]["format"]["properties"]]
        self.assertEqual(options, [(0, 42), (0.7, 43), (0.7, 44)])


class CachingReproducesRecordedChecks(unittest.TestCase):
    """Rechecks redid all five sentences; 133/133 unchanged ones came out identical."""

    def test_seeded_first_check_reproduces_the_recording_without_any_qwen_call(self):
        name = "Obama_prompt2_seed202"
        run = recorded(name)
        first = run["iterations"][0]
        client = Mock()
        client.chat.side_effect = AssertionError("an unchanged sentence was sent to Qwen again")
        scorer = Mock()
        scorer.score_pairs.side_effect = lambda pairs: [0.05] * len(pairs)
        result = run_summary_agent(Graph().parse(ROOT / run["pipeline"]["graph"], format="turtle"),
                                   run["initial_summary"], topic="Obama", prompt=run["prompt"], client=client,
                                   disparity_threshold=0.2, max_revisions=0, mnli_scorer=scorer,
                                   check_cache=seed_check_cache(run["initial_summary"], first))
        client.chat.assert_not_called()
        self.assertEqual(result["summary_claims"], first["summary_claims"])
        self.assertEqual(result["factual_results"]["verdict_counts"], first["factual_results"]["verdict_counts"])
        self.assertEqual([i["severity"] for i in result["factual_issues"]], [i["severity"] for i in first["factual_issues"]])
        self.assertEqual(result["factual_results"]["reused_check_count"], len(first["factual_results"]["assessments"]))

    def test_seed_refuses_a_check_of_another_draft(self):
        other = recorded("Obama_prompt1_seed101")["iterations"][0]
        with self.assertRaises(ValueError):
            seed_check_cache(recorded("Obama_prompt2_seed202")["initial_summary"], other)

    def test_failed_extractions_are_not_seeded_so_they_get_the_retry(self):
        run = recorded("Superbowl_prompt1_seed101")
        cache = seed_check_cache(run["initial_summary"], run["iterations"][0])
        self.assertEqual(len(cache["extraction"]), 4)  # S04 failed in the recording

    def test_a_recheck_only_checks_the_changed_sentence(self):
        graph_, draft, counts = fixture("factual")
        result = run_summary_agent(graph_, draft, topic="Obama", prompt=draft["summary_prompt"], client=mock_client(),
                                   expected_group_counts=counts, disparity_threshold=0.5, representation_metric="internal")
        self.assertEqual(result["revisions"][0]["status"], "accepted")
        changed = {c["sentence_id"] for c in result["revisions"][0]["changes"]}
        recheck = result["iterations"][1]["factual_results"]
        fresh = [a for a in recheck["assessments"] if not a.get("reused_check")]
        self.assertTrue(fresh and all(a["sentence_id"] in changed for a in fresh))
        self.assertEqual(recheck["reused_check_count"], len(recheck["assessments"]) - len(fresh))

    def test_cached_scorer_scores_each_pair_once(self):
        inner = Mock()
        inner.score_pairs.side_effect = lambda pairs: [0.5] * len(pairs)
        scorer = CachedScorer(inner)
        self.assertEqual(scorer.score_pairs([("a", "x"), ("b", "x"), ("a", "x")]), [0.5] * 3)
        self.assertEqual(scorer.score_pairs([("a", "x"), ("c", "x")]), [0.5] * 2)
        self.assertEqual([len(c.args[0]) for c in inner.score_pairs.call_args_list], [2, 1])


class KnownLimitations(unittest.TestCase):
    @unittest.expectedFailure
    def test_each_claim_is_checked_only_against_tweets_cited_for_it(self):
        # Open: every atomic claim inherits its sentence's full citation list,
        # so "others express disgust" is judged next to the tweets cited for
        # "some find it delicious" and the 4B checker calls it mixed -> UNKNOWN.
        shared = 0
        for name in trials():
            for record in recorded(name)["iterations"][0]["summary_claims"]:
                if len(record["claims"]) > 1:
                    shared += len({tuple(c["cited_source_ids"]) for c in record["claims"]}) == 1
        self.assertEqual(shared, 0)


if __name__ == "__main__":
    unittest.main()
