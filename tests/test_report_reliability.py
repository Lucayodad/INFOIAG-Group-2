"""Offline regressions for the report metrics, the evaluation files and result accounting."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import evaluate
import evaluate_label_free_run
from fact_check import FactCheckError, _validate_verdict
from equal_coverage import calculate_equal_coverage
from model_usage import RecordingClient
from summary_agent import run_summary_agent
from fakes import fixture, mock_client

SENTENCES = ["The author likes Obama's speech.", "The author dislikes Obama's speech.",
             "Millions watched the speech.", "End of briefing.", "Further context follows."]
TWEETS = {g: [{"text": f"{g} tweet {n}"} for n in range(1, 4)] for g in evaluate.GROUPS}


class SentenceCoverageTests(unittest.TestCase):
    """Equal Coverage as the evaluation computes it: one pair per (tweet, summary sentence)."""

    def setUp(self):
        self.scorer = Mock()
        self.scorer.score_pairs.side_effect = lambda pairs: [0.1] * len(pairs)
        evaluate._MNLI, evaluate._CACHE = None, {"mnli": {}, "bart": {}}

    def coverage(self, by_group):
        self.scorer.score_pairs.side_effect = lambda pairs: [by_group[pairs[0][0].split()[0]]] * len(pairs)
        return calculate_equal_coverage(TWEETS, [{"claim": s} for s in SENTENCES], scorer=self.scorer)

    def test_equal_coverage_is_the_mean_deviation_from_overall_coverage(self):
        result = self.coverage({"AA": 0.05, "Hisp": 0.1, "White": 0.1})
        self.assertAlmostEqual(result["overall_coverage"], 0.25 / 3)
        self.assertAlmostEqual(result["equal_coverage"], 0.1 / 4.5)
        for group, expected in {"AA": 0.05, "Hisp": 0.1, "White": 0.1}.items():
            self.assertAlmostEqual(result["coverage_by_group"][group]["coverage"], expected)
        self.assertEqual(result["pair_count"], 45)  # 9 tweets x 5 sentences
        self.assertEqual(result["metric_version"], "mnli_sentence_mean_v1")
        self.assertAlmostEqual(self.coverage({"AA": 0.1, "Hisp": 0.1, "White": 0.1})["equal_coverage"], 0)

    def test_invalid_scores_are_rejected(self):
        for values in ([0.1], [float("nan")] * 15, [1.5] * 15):
            self.scorer.score_pairs.side_effect = lambda pairs, values=values: values
            with self.subTest(values=values[:1]), self.assertRaises(ValueError):
                calculate_equal_coverage(TWEETS, [{"claim": s} for s in SENTENCES], scorer=self.scorer)

    def test_both_evaluations_score_the_same_pairs_and_agree(self):
        summary = {"sentences": [{"text": s} for s in SENTENCES]}
        tweets = [{"text": t["text"], "group": g} for g, rows in TWEETS.items() for t in rows]
        with patch("equal_coverage._default_scorer", return_value=self.scorer):
            offline = evaluate.metric_ec({"sentences": SENTENCES}, tweets)
        pairs = [call.args[0] for call in self.scorer.score_pairs.call_args_list]
        self.scorer.reset_mock()
        final = evaluate_label_free_run.group_coverage(TWEETS, summary, self.scorer)
        self.assertEqual(pairs, [call.args[0] for call in self.scorer.score_pairs.call_args_list])
        self.assertAlmostEqual(final["equal_coverage"], offline["ec"])

    def test_missing_group_is_not_zero_coverage(self):
        with self.assertRaisesRegex(ValueError, "Every group"):
            calculate_equal_coverage({g: [] for g in evaluate.GROUPS}, [{"claim": "x"}], scorer=self.scorer)


class AccountingTests(unittest.TestCase):
    def test_explicit_conflicting_verdict_is_rejected(self):
        with self.assertRaisesRegex(FactCheckError, "conflicts"):
            _validate_verdict({"verdict": "SUPPORTED", "references": [],
                "explanation": "Therefore, the verdict should be CONTRADICTED."}, {})
        # Merely discussing other labels does not declare a different verdict.
        _validate_verdict({"verdict": "UNKNOWN", "references": [],
            "explanation": "Neither SUPPORTED nor CONTRADICTED can be established."}, {})

    def test_all_revision_extraction_and_recheck_calls_count_once(self):
        graph, draft = fixture()
        backend = mock_client()
        original = backend.chat.side_effect
        def chat(payload):
            result = original(payload)
            result["usage"] = {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
            return result
        backend.chat.side_effect = chat
        recorder = RecordingClient(backend)
        result = run_summary_agent(graph, draft, topic="Obama", prompt=draft["summary_prompt"], client=recorder)
        self.assertEqual(result["revision_count"], 1)
        self.assertEqual(result["llm_calls"], backend.chat.call_count)
        self.assertEqual(len(result["model_calls"]), result["call_counts"]["model_calls"])
        self.assertEqual(result["usage"]["input_tokens"], 2 * backend.chat.call_count)
        self.assertTrue(result["usage_complete"])
        json.dumps(result)

    def test_failed_or_missing_usage_never_looks_complete(self):
        backend = Mock()
        backend.chat.side_effect = [{"usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}}, RuntimeError("failed")]
        client = RecordingClient(backend)
        client.chat({})
        with self.assertRaises(RuntimeError):
            client.chat({})
        result = client.accounting()
        self.assertFalse(result["usage_complete"])
        self.assertIsNone(result["usage"])
        self.assertEqual(result["known_usage"]["total_tokens"], 3)
        self.assertEqual(result["llm_calls"], 2)


class EvaluationIOTests(unittest.TestCase):
    def data(self):
        summary = {"summary": "One. Two. Three. Four. Five.",
                   "sentences": [{"text": x, "source_ids": ["Obama_001"]} for x in ("One.", "Two.", "Three.", "Four.", "Five.")]}
        return {"topic": "Obama", "prompt_id": "1", "seed": 101,
                "initial_summary": summary, "current_summary": summary,
                "status": "completed_with_uncertainty", "stop_reason": "uncertainty_only",
                "revision_count": 0, "usage": {"input_tokens": 100}}

    def test_native_state_and_pipeline_output_and_legacy_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Obama.json"
            path.write_text(json.dumps(self.data()), encoding="utf-8")
            rows = evaluate.records_from_file(path, "agent")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["sentences"], ["One.", "Two.", "Three.", "Four.", "Five."])
            self.assertEqual(rows[0]["stop_reason"], "uncertainty_only")
            self.assertFalse(rows[0]["usage_complete"])
            self.assertEqual(rows[0]["input_tokens"], "")

    def test_manifest_excludes_drafts_and_overlapping_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sub in ("drafts", "summaries"):
                (root / sub).mkdir()
                (root / sub / "Obama_prompt1_seed101.json").write_text(json.dumps(self.data()), encoding="utf-8")
            manifest = {"runs": [{"status": "completed", "output_path": "summaries\\Obama_prompt1_seed101.json"}]}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            rows = evaluate.collect_records([root, root / "summaries"], "agent")
            self.assertEqual(len(rows), 2)  # final and explicitly labelled initial draft
            self.assertEqual([r["system"] for r in rows], ["agent", "agent_draft"])

    def test_old_csv_schema_upsert_and_report_partial_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_text("system,topic,prompt_id,seed,file,rouge1\nagent,Obama,1,101,a.json,0.2\n", encoding="utf-8")
            row = dict(system="agent", topic="Obama", prompt_id="1", seed=101, file="a.json", bart_src=-3.1)
            evaluate.append_rows([row], path)
            evaluate.append_rows([row], path)
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["rouge1"], "0.2")
            self.assertEqual(rows[0]["bart_src"], "-3.1")
            with redirect_stdout(io.StringIO()):
                evaluate.report(path)
            self.assertIn("n=1", path.with_name("report.md").read_text(encoding="utf-8"))

    def test_final_agent_output_reports_its_recorded_cost(self):
        data = {**self.data(), "llm_calls": 20, "usage_complete": True,
                "usage": {"input_tokens": 34706, "output_tokens": 3527},
                "usage_scope": "agent_only_excludes_initial_draft_and_source_extraction",
                "provenance": {"seconds": 533}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Obama_prompt1_seed101.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            final, draft = evaluate.records_from_file(path, "agent")
        self.assertEqual((final["execution_seconds"], final["llm_calls"], final["input_tokens"]), (533, 20, 34706))
        self.assertEqual(final["usage_scope"], "agent_only_excludes_initial_draft_and_source_extraction")
        self.assertEqual((draft["system"], draft["llm_calls"]), ("agent_draft", 1))

    def test_score_cache_round_trip(self):
        evaluate._CACHE = {"mnli": {("tweet", "sentence"): 0.25}, "bart": {("source", "target"): -3.5}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.json"
            evaluate.save_cache(path)
            evaluate._CACHE = {"mnli": {}, "bart": {}}
            evaluate.load_cache(path)
            self.assertEqual(evaluate._CACHE, {"mnli": {("tweet", "sentence"): 0.25}, "bart": {("source", "target"): -3.5}})
            self.assertEqual(evaluate.bart_logprob("source", "target"), -3.5)  # served from the cache, no model
            path.write_text(json.dumps([["tweet", "other sentence", 0.5]]), encoding="utf-8")
            evaluate.load_cache(path)  # the plain MNLI pair list of evaluate_label_free_run.py
            self.assertEqual(evaluate._CACHE["mnli"][("tweet", "other sentence")], 0.5)
        evaluate._CACHE = {"mnli": {}, "bart": {}}

    def test_changed_summary_invalidates_old_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            row = dict(system="agent", topic="Obama", file="a.json", summary_sha256="old", ec=0.1)
            evaluate.append_rows([row], path)
            evaluate.append_rows([{**row, "summary_sha256": "new", "ec": ""}], path)
            with path.open(newline="", encoding="utf-8") as f:
                self.assertEqual(next(csv.DictReader(f))["ec"], "")


if __name__ == "__main__":
    unittest.main()
