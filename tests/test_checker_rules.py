"""The revised fact checker: cited tweets first, verdict last, counting rules applied by code."""

import json
import unittest
from unittest.mock import Mock

from fact_agent import DECISION_ORDER, _loop_schema, run_fact_agent
from fact_check import (MAX_INPUT_BYTES, NUM_PREDICT, VERDICT_SCHEMA, apply_verdict, compact_evidence,
                        counting_rule)
from test_fact_agent import loop_response
from test_fact_check import controlled_fixture

TWEET = "http://example.org/run/tweet/McDonalds_{:03d}"
TEXTS = {1: "McDonalds be stingy af with them sauce packets", 2: "ONE pack of sauce for a 20 piece, really",
         3: "only one sauce again smh", 4: "I hate that breakfast ends at 10:30"}


def assessment(claim):
    evidence = {TWEET.format(n): {"kind": "tweet", "quote_text": text, "text": text} for n, text in TEXTS.items()}
    return {"summary_claim_text": claim, "evidence": evidence, "evidence_complete": True}


def verdict(label, *numbers):
    return {"verdict": label, "explanation": "The quoted tweets decide it.",
            "references": [{"evidence_id": TWEET.format(n), "quote": TEXTS[n]} for n in numbers]}


class CountingRules(unittest.TestCase):
    def check(self, claim, label, *numbers):
        return apply_verdict(assessment(claim), verdict(label, *numbers))

    def test_one_tweet_supports_some_others_and_plain_claims(self):
        for claim in ("Some users complain about sauce packets.", "Others complain about sauce packets.",
                      "Breakfast service has limited hours."):
            with self.subTest(claim=claim):
                result = self.check(claim, "SUPPORTED", 1)
                self.assertEqual((result["verdict"], result["decision_source"]), ("SUPPORTED", "model"))

    def test_many_needs_three_distinct_quoted_tweets(self):
        two = self.check("Many users complain about sauce packets.", "SUPPORTED", 1, 2)
        self.assertEqual((two["verdict"], two["reason"], two["model_verdict"]),
                         ("UNKNOWN", "fewer_than_three_tweets", "SUPPORTED"))
        self.assertIn("Counting rule:", two["explanation"])
        self.assertEqual(self.check("Users frequently complain about sauce.", "SUPPORTED", 1, 2, 3)["verdict"],
                         "SUPPORTED")

    def test_a_share_of_the_collection_is_never_supported_by_cited_tweets(self):
        result = self.check("A significant number of tweets complain about sauce.", "SUPPORTED", 1, 2, 3)
        self.assertEqual((result["verdict"], result["reason"]), ("UNKNOWN", "share_of_collection"))

    def test_other_users_never_refute_what_some_users_say(self):
        result = self.check("Some users say breakfast is served all day.", "CONTRADICTED", 4)
        self.assertEqual((result["verdict"], result["reason"], result["model_verdict"]),
                         ("UNKNOWN", "attributed_claim_not_refuted", "CONTRADICTED"))
        # A plain factual claim can still be contradicted by a cited tweet.
        self.assertEqual(self.check("Breakfast is served until noon.", "CONTRADICTED", 4)["verdict"], "CONTRADICTED")

    def test_unknown_is_never_changed(self):
        self.assertIsNone(counting_rule("Many users complain.", "UNKNOWN", []))


class EvidenceAndFormat(unittest.TestCase):
    def test_every_cited_tweet_comes_before_any_interpretation(self):
        tweet = "urn:run/tweet/McDonalds_{}"
        evidence = {
            tweet.format(1): {"kind": "tweet", "quote_text": "lol that line", "text": "lol that line"},
            tweet.format(2): {"kind": "tweet", "quote_text": "stingy with sauce packets", "text": "stingy with sauce packets"},
            "urn:run/claim/McDonalds_2_C01": {"kind": "source_claim", "quote_text": "stingy with sauce packets",
                                              "text": "The author says McDonald's is stingy with sauce packets",
                                              "source_tweet_uri": tweet.format(2)},
        }
        view = compact_evidence({"summary_claim_text": "Users say McDonald's is stingy with sauce packets",
                                 "evidence": evidence})
        kinds = [entry["kind"] for entry in view["evidence_catalogue"].values()]
        self.assertEqual(kinds, ["tweet", "tweet", "source_claim"])

    def test_the_model_explains_and_quotes_before_the_verdict(self):
        self.assertEqual(VERDICT_SCHEMA["required"], ["explanation", "references", "verdict"])
        schema = _loop_schema([], {"E001": {"kind": "tweet", "quote_text": "x"}})
        self.assertEqual(list(schema["properties"]), list(DECISION_ORDER))
        self.assertEqual(schema["required"][-1], "verdict")

    def test_requests_use_the_raised_limits(self):
        graph, claim = controlled_fixture()
        client = Mock()
        client.chat.side_effect = [loop_response("lookup_fact"), loop_response(
            "finalize", verdict="UNKNOWN", explanation="The cited tweet does not show it.")]
        run_fact_agent(graph, claim, client)
        payload = client.chat.call_args.args[0]
        self.assertEqual(payload["options"]["num_predict"], NUM_PREDICT)
        self.assertLessEqual(len(json.dumps({"messages": payload["messages"], "format": payload["format"]},
                                            ensure_ascii=False).encode()), MAX_INPUT_BYTES)


class RecordedVerdicts(unittest.TestCase):
    def test_verdicts_of_an_older_checker_are_checked_anew(self):
        from pathlib import Path
        from fact_check import CHECKER_VERSION
        from run_final_agent import first_check
        from summary_agent import seed_check_cache
        root, name = Path(__file__).resolve().parents[1] / "results", "McDonalds_prompt1_seed101"
        run = json.loads((root / "agent_rerun" / f"{name}.json").read_text(encoding="utf-8"))
        run = run.get("agent_result", run)
        checked = first_check(json.loads((root / "agent_final" / f"{name}.json").read_text(encoding="utf-8"))["iterations"][0])
        cache = seed_check_cache(run["initial_summary"], checked)
        self.assertEqual(cache["facts"], {})
        self.assertTrue(cache["extraction"])  # claim extraction does not depend on the checker
        self.assertTrue(seed_check_cache(run["initial_summary"], checked, any_checker=True)["facts"])
        stamped = [{**a, "checker_version": CHECKER_VERSION} for a in checked["factual_results"]["assessments"]]
        current = {**checked, "factual_results": {**checked["factual_results"], "assessments": stamped}}
        self.assertEqual(len(seed_check_cache(run["initial_summary"], current)["facts"]), len(stamped))


if __name__ == "__main__":
    unittest.main()
