"""Real retrieval, cosine ranking, Qwen protocol, checking agents and LangGraph; mocked models."""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock

from demo_revision_loop import fixture, mock_client, mock_embeddings
from fact_check import quote_in
from ontology_tools import NS
from revision_retrieval import cosine, retrieve_revision_evidence
from summary_agent import compare_revision, run_summary_agent
from summary_revision import revise_summary


class RevisionLoopTests(unittest.TestCase):
    def run_fixture(self, kind="both", client=None, **options):
        options.setdefault("representation_metric", "internal")
        graph, draft, counts = fixture(kind, many=options.pop("many", False), per_group=options.pop("per_group", 8))
        client = mock_client() if client is None else client
        before_graph, before_draft = set(graph), deepcopy(draft)
        threshold = options.pop("disparity_threshold", 0.5 if kind == "factual" else 0.25)
        result = run_summary_agent(graph, draft, topic="Obama", prompt=draft["summary_prompt"],
                                   client=client, expected_group_counts=counts,
                                   disparity_threshold=threshold, **options)
        self.assertEqual(set(graph), before_graph)
        self.assertEqual(draft, before_draft)
        self.assertEqual(result["call_counts"]["model_calls"], client.chat.call_count)
        self.assertEqual(result["call_counts"]["embedding_calls"], client.embed.call_count)
        json.dumps(result)
        return result, client

    def test_no_revision_needed(self):
        result, client = self.run_fixture("representation", disparity_threshold=0.5)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["revision_count"], 0)
        self.assertEqual(result["revisions"], [])
        client.embed.assert_not_called()

    def test_factual_contradiction_repaired(self):
        result, _ = self.run_fixture("factual")
        self.assertEqual(result["revision_count"], 1)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["iterations"][0]["metrics"]["serious_factual_count"], 1)
        self.assertEqual(result["iterations"][-1]["metrics"]["serious_factual_count"], 0)
        self.assertEqual(result["revisions"][0]["status"], "accepted")
        self.assertEqual(result["revisions"][0]["changes"][0]["sentence_id"], "S02")

    def test_underrepresented_group_addressed_with_its_own_evidence(self):
        result, _ = self.run_fixture("representation")
        self.assertEqual(result["status"], "completed")
        self.assertAlmostEqual(result["iterations"][0]["metrics"]["disparity"], 1 / 3)
        self.assertEqual(result["iterations"][-1]["metrics"]["disparity"], 0)
        for retrieval in result["retrieval_history"]:
            by_id = {row["tweet_id"]: row for row in retrieval["tweets"]}
            for task in retrieval["tasks"]:
                if task["kind"] == "representation":
                    self.assertTrue(task["evidence_ids"])
                    self.assertTrue(all(by_id[key]["dialect_label"] == task["dialect_label"] for key in task["evidence_ids"]))

    def test_both_problems_and_exact_preservation_of_unaffected_sentences(self):
        result, _ = self.run_fixture()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["revision_count"], 1)
        self.assertEqual(result["stop_reason"], "no_confirmed_serious_issues")
        self.assertEqual(len(result["summary_versions"]), 2)
        self.assertEqual(len(result["iterations"]), 2)
        self.assertEqual(result["best_iteration"], 1)
        for revision in result["revisions"]:
            changed = {row["sentence_id"] for row in revision["changes"]}
            for i in range(5):
                if f"S{i + 1:02d}" not in changed:
                    self.assertEqual(revision["summary_before"]["sentences"][i], revision["candidate"]["sentences"][i])
        visits = [event["node"] for event in result["execution_history"]]
        for node in ("extract_summary_claims", "factual_check", "representation_check", "route_checks"):
            self.assertEqual(visits.count(node), 2)
        self.assertEqual(visits.count("retrieve_revision_evidence"), 1)
        self.assertEqual(visits.count("revise_summary"), 1)
        self.assertEqual([r["node"] for r in result["routing_decisions"]],
                         ["route_checks", "retrieve_revision_evidence", "revise_summary", "route_checks"])

    def test_unknown_without_confirmed_problems_never_revises(self):
        result, client = self.run_fixture("representation", client=mock_client(unknown_matching=True))
        self.assertEqual(result["status"], "completed_with_uncertainty")
        self.assertEqual(result["stop_reason"], "uncertainty_only")
        self.assertEqual(result["revision_count"], 0)
        client.embed.assert_not_called()

    def test_invalid_qwen_revisions_are_rejected_without_overwriting_summary(self):
        def invented(data):
            data["replacements"][0]["source_ids"] = ["invented"]
            return data
        def unaffected(data):
            data["replacements"][0]["sentence_id"] = "S01"
            return data
        def fake_quote(data):
            data["replacements"][0]["evidence"][0]["quote"] = "Not in any original tweet."
            return data
        def wrong_group(data):
            data["replacements"][0].update(text="One author approves of the speech.", source_ids=["AA_01"],
                evidence=[{"tweet_id": "AA_01", "quote": "I like Obama's speech."}])
            return data
        for transform in (invented, unaffected, fake_quote, wrong_group, lambda data: {"bad": "schema"}):
            with self.subTest(transform=transform):
                result, _ = self.run_fixture(client=mock_client(transform_revision=transform))
                self.assertEqual(result["stop_reason"], "invalid_revision")
                self.assertEqual(result["revisions"][0]["status"], "rejected")
                self.assertEqual(result["current_summary"], result["initial_summary"])
                self.assertEqual(result["revision_count"], 1)
                self.assertEqual(len(result["iterations"]), 1)

    def test_verbatim_qwen_revision_is_rejected(self):
        def copied(data):
            replacement = data["replacements"][0]
            replacement["text"] = replacement["evidence"][0]["quote"]
            return data

        result, _ = self.run_fixture(client=mock_client(transform_revision=copied))
        self.assertEqual(result["revisions"][0]["status"], "rejected")
        self.assertEqual(result["revisions"][0]["reason"], "invalid_revision")
        self.assertIn("verbatim", result["revisions"][0]["error"])
        self.assertEqual(result["current_summary"], result["initial_summary"])

    def test_quotation_wrapped_in_quotation_marks_is_accepted(self):
        def wrapped(data):
            for reference in data["replacements"][0]["evidence"]:
                reference["quote"] = f' "{reference["quote"]}" '
            return data
        result, _ = self.run_fixture("factual", client=mock_client(transform_revision=wrapped))
        self.assertEqual(result["revisions"][0]["status"], "accepted")
        self.assertEqual(result["status"], "completed")

    def test_quote_check_tolerates_only_entities_and_wrapping_marks(self):
        tweet = "Obama in office niggas &amp; bitches get yo act together"
        self.assertTrue(quote_in("niggas & bitches get yo act", tweet))
        self.assertTrue(quote_in('"Obama in office"', tweet))
        self.assertFalse(quote_in("Obama out of office", tweet))
        self.assertFalse(quote_in('""', tweet))

    def test_one_format_repair_stays_within_the_same_revision(self):
        calls = []
        def fail_first(data):
            calls.append(1)
            if len(calls) == 1:
                data["replacements"][0]["evidence"][0]["quote"] = "Not in any original tweet."
            return data
        result, client = self.run_fixture("factual", client=mock_client(transform_revision=fail_first))
        revision = result["revisions"][0]
        self.assertEqual(revision["status"], "accepted")
        self.assertEqual(revision["model_call_count"], 2)
        self.assertEqual(revision["attempt_errors"], ["Quotation is absent from the cited original tweet."])
        self.assertEqual(result["revision_count"], 1)
        repair = [call.args[0] for call in client.chat.call_args_list
                  if "replacements" in call.args[0]["format"]["properties"]][1]
        self.assertIn("Quotation is absent", repair["messages"][-1]["content"])

    def test_unsupported_addition_is_rechecked_and_rolled_back(self):
        def addition(data):
            data["replacements"][0]["text"] = "An invented quantity proves the speech was popular."
            return data
        result, client = self.run_fixture(client=mock_client(transform_revision=addition))
        self.assertEqual(result["stop_reason"], "unsupported_revision")
        # Each rolled-back fix is retried with feedback until the three-revision bound.
        self.assertEqual([r["status"] for r in result["revisions"]], ["rolled_back"] * 3)
        self.assertEqual(result["revision_count"], 3)
        self.assertEqual(result["current_summary"], result["initial_summary"])
        self.assertEqual(len(result["iterations"]), 4)
        self.assertFalse(result["iterations"][-1]["retained"])
        self.assertEqual(result["best_iteration"], 0)
        requests = [json.loads(call.args[0]["messages"][1]["content"]) for call in client.chat.call_args_list
                    if "replacements" in call.args[0]["format"]["properties"]]
        feedback = [i for i in requests[1]["confirmed_issues"] if i["type"] == "rejected_previous_revision"]
        self.assertIn("An invented quantity", feedback[0]["explanation"])

    def test_rolled_back_fix_is_retried_and_can_then_be_accepted(self):
        calls = []
        def addition_first(data):
            calls.append(1)
            if len(calls) == 1:
                data["replacements"][0]["text"] = "An invented quantity proves the speech was popular."
            return data
        result, _ = self.run_fixture("factual", client=mock_client(transform_revision=addition_first))
        self.assertEqual([r["status"] for r in result["revisions"]], ["rolled_back", "accepted"])
        self.assertEqual(result["status"], "completed")
        self.assertNotEqual(result["current_summary"], result["initial_summary"])

    def test_replacement_naming_a_dialect_group_is_rejected(self):
        def group_label(data):
            data["replacements"][0]["text"] = "Some users in the AA group request more detail."
            return data
        result, _ = self.run_fixture("factual", client=mock_client(transform_revision=group_label))
        self.assertEqual(result["revisions"][0]["status"], "rejected")
        self.assertIn("names a dialect group", result["revisions"][0]["error"])
        self.assertEqual(result["current_summary"], result["initial_summary"])

    def test_missing_embedding_evidence_stops_without_a_qwen_revision(self):
        client = mock_client()
        client.embed.side_effect = OSError("Controlled local model unavailable")
        result, _ = self.run_fixture(client=client)
        self.assertEqual(result["stop_reason"], "insufficient_revision_evidence")
        self.assertEqual(result["revision_count"], 0)
        self.assertEqual(result["revisions"], [])
        self.assertEqual(result["current_summary"], result["initial_summary"])

    def test_no_improvement_rolls_back_and_stops(self):
        result, _ = self.run_fixture("representation", client=mock_client(no_improvement=True))
        self.assertEqual(result["stop_reason"], "no_improvement")
        self.assertEqual(result["revisions"][0]["status"], "rolled_back")
        self.assertEqual(result["current_summary"], result["initial_summary"])
        self.assertEqual(result["revision_count"], 1)

    def test_maximum_three_revisions_with_rechecking_and_best_retention(self):
        result, _ = self.run_fixture("representation", many=True, disparity_threshold=0.01)
        self.assertEqual(result["revision_count"], 3)
        self.assertEqual(result["stop_reason"], "revision_limit")
        self.assertEqual(result["status"], "stopped_with_issues")
        self.assertEqual(len(result["iterations"]), 4)
        self.assertTrue(all(r["status"] == "accepted" for r in result["revisions"]))
        disparities = [iteration["metrics"]["disparity"] for iteration in result["iterations"]]
        self.assertEqual(disparities, sorted(disparities, reverse=True))
        self.assertEqual(result["best_iteration"], 3)
        self.assertEqual(result["current_summary"], result["summary_versions"][-1])
        with self.assertRaises(ValueError):
            self.run_fixture(max_revisions=4)

    def test_unknown_values_and_changed_partitions_are_not_improvements(self):
        result, _ = self.run_fixture()
        before = result["iterations"][0]
        after = deepcopy(result["iterations"][1])
        changes = result["revisions"][0]["changes"]
        # Losing a contradiction to UNKNOWN is not a repair.
        after["factual_results"]["assessments"][1]["verdict"] = "UNKNOWN"
        self.assertEqual(compare_revision(before, after, changes), (False, "unsupported_revision"))
        before = result["iterations"][0]
        after = deepcopy(result["iterations"][-1])
        after["representation_results"]["status"] = "UNKNOWN"
        self.assertFalse(compare_revision(before, after, result["revisions"][0]["changes"])[0])
        after = deepcopy(result["iterations"][-1])
        after["representation_results"]["matching"]["clusters"][0]["source_claim_ids"].append("fake_partition_change")
        self.assertFalse(compare_revision(before, after, result["revisions"][0]["changes"])[0])

    def test_complete_ninety_tweet_topic_executes_revision_and_rechecking(self):
        result, _ = self.run_fixture("representation", many=True, per_group=30,
                                     max_revisions=1, disparity_threshold=0.01)
        self.assertEqual(result["revision_count"], 1)
        self.assertEqual(len(result["iterations"]), 2)
        self.assertEqual(result["representation_results"]["source_tweet_counts"], {g: 30 for g in ("AA", "Hisp", "White")})
        self.assertEqual(result["revisions"][0]["status"], "accepted")
        self.assertLess(result["iterations"][1]["metrics"]["disparity"], result["iterations"][0]["metrics"]["disparity"])

    def test_later_rejected_proposal_preserves_the_previous_improved_summary(self):
        calls = []
        def fail_second(data):
            calls.append(1)
            if len(calls) >= 2:  # the proposal and its one format repair
                data["replacements"][0]["source_ids"] = ["fabricated"]
            return data
        result, _ = self.run_fixture("representation", many=True, disparity_threshold=0.01,
                                     client=mock_client(transform_revision=fail_second))
        self.assertEqual(result["revision_count"], 2)
        self.assertEqual([r["status"] for r in result["revisions"]], ["accepted", "rejected"])
        self.assertEqual(result["current_summary"], result["summary_versions"][1])
        self.assertNotEqual(result["current_summary"], result["initial_summary"])
        self.assertEqual(result["best_iteration"], 1)

    def test_retrieval_is_deterministic_bounded_and_never_invents_original_text(self):
        result, _ = self.run_fixture()
        graph, _, _ = fixture()
        initial = result["iterations"][0]
        arguments = (graph, "Obama", initial["summary_claims"], initial["factual_results"],
                     initial["representation_results"], mock_embeddings)
        first = retrieve_revision_evidence(*arguments, top_k=1, max_tweets=3)
        self.assertEqual(first, retrieve_revision_evidence(*arguments, top_k=1, max_tweets=3))
        self.assertLessEqual(len(first["tweets"]), 3)
        originals = {str(text) for _, _, text in graph.triples((None, NS.tweetHasText, None))}
        self.assertTrue(all(row["quote"] in originals for row in first["tweets"]))
        self.assertEqual(retrieve_revision_evidence(*arguments, max_evidence_bytes=1)["status"], "insufficient")
        for vector in ([], [0, 0], [float("nan"), 1]):
            with self.assertRaises(ValueError):
                cosine(vector, [1, 0])

    def test_no_original_sources_and_no_silent_embedding_fallback(self):
        result, _ = self.run_fixture()
        graph, _, _ = fixture()
        initial = result["iterations"][0]
        graph.remove((None, NS.tweetHasText, None))
        retrieved = retrieve_revision_evidence(graph, "Obama", initial["summary_claims"],
                                               initial["factual_results"], initial["representation_results"], mock_embeddings)
        self.assertEqual(retrieved["status"], "insufficient")


if __name__ == "__main__":
    unittest.main()
