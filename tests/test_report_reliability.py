"""Offline regressions for report units, missing knowledge and result accounting."""

from contextlib import redirect_stdout
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import evaluate
import prepare_report
from fact_check import FactCheckError, _validate_verdict
from equal_coverage import calculate_equal_coverage
from model_usage import RecordingClient
from representation_check import representation_check
from summary_agent import run_summary_agent
from demo_revision_loop import fixture, mock_client
from test_check_components import complete_check_fixture


class SentenceCoverageTests(unittest.TestCase):
    def setUp(self):
        self.graph, self.claims = complete_check_fixture()
        self.scorer = Mock()
        self.scorer.score_pairs.side_effect = lambda pairs: [0.1] * len(pairs)

    def check(self, claims=None):
        return representation_check(self.graph, self.claims if claims is None else claims,
            Mock(), "Obama", 0.25, expected_group_counts={g: 3 for g in evaluate.GROUPS},
            mnli_scorer=self.scorer)

    def check_coverages(self, coverages, threshold=0.2):
        values = iter(coverages)
        self.scorer.score_pairs.side_effect = lambda pairs: [next(values)] * len(pairs)
        return representation_check(self.graph, self.claims, Mock(), "Obama", threshold,
            expected_group_counts={g: 3 for g in evaluate.GROUPS}, mnli_scorer=self.scorer)

    def test_threshold_applies_the_four_fifths_rule(self):
        # Raw EC 0.022 could never exceed a [0, 1] threshold; AA is at half of White.
        result = self.check_coverages([0.05, 0.1, 0.1])
        self.assertAlmostEqual(result["equal_coverage"], 0.1 / 4.5)
        self.assertAlmostEqual(result["disparity"], 0.5)
        self.assertEqual(result["threshold_metric"], "four_fifths_coverage_ratio")
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["representation_issues"][0]["dialect_label"], "AA")
        self.assertEqual(self.check_coverages([0.125, 0.1, 0.11])["status"], "PASS")  # exactly 4/5
        self.assertEqual(self.check_coverages([0.125, 0.099, 0.11])["status"], "FAIL")

    def test_failed_extraction_cannot_pass(self):
        self.claims[0].update(status="unclear", claims=[])
        result = self.check()
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertFalse(result["data_complete"])
        self.assertIsNone(result["disparity"])
        self.scorer.score_pairs.assert_not_called()

    def test_empty_input_and_scoring_failure_are_unknown(self):
        self.assertEqual(self.check([])["status"], "UNKNOWN")
        self.scorer.score_pairs.side_effect = RuntimeError("model unavailable")
        result = self.check()
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIn("model unavailable", str(result["reasons"]))
        self.assertTrue(result["data_complete"])
        self.assertIn("mnli_scoring_unavailable", result["reasons"])
        self.assertNotIn("incomplete_topic_or_extraction_data", result["reasons"])
        self.assertTrue(all(row["coverage"] is None for row in result["coverage_by_group"].values()))

    def test_invalid_scores_are_unknown(self):
        for values in ([0.1], [float("nan")] * 15):
            self.scorer.score_pairs.side_effect = lambda pairs: values
            self.assertEqual(self.check()["status"], "UNKNOWN")

    def test_atomic_decomposition_does_not_change_sentence_metric(self):
        first = self.check()
        atomic = deepcopy(self.claims[0]["claims"][0])
        atomic.update(claim_id="S01_C02", claim="Another extracted proposition.")
        self.claims[0]["claims"].append(atomic)
        self.assertEqual(first["equal_coverage"], self.check()["equal_coverage"])
        self.assertEqual(first["pair_count"], 45)
        self.assertEqual(first["hypothesis_unit"], "summary_sentence")

    def test_online_and_offline_use_identical_pairs(self):
        online = self.check()
        pairs = [call.args[0] for call in self.scorer.score_pairs.call_args_list]
        self.scorer.reset_mock()
        tweets = [{"text": text, "group": group} for group, batch in zip(evaluate.GROUPS, pairs)
                  for text, _ in batch[::5]]
        with patch("equal_coverage._default_scorer", return_value=self.scorer):
            offline = evaluate.metric_ec({"sentences": [r["sentence"] for r in self.claims]}, tweets)
        self.assertEqual(pairs, [call.args[0] for call in self.scorer.score_pairs.call_args_list])
        self.assertAlmostEqual(online["equal_coverage"], offline["ec"])

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
        graph, draft, counts = fixture()
        backend = mock_client()
        original = backend.chat.side_effect
        def chat(payload):
            result = original(payload)
            result["usage"] = {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
            return result
        backend.chat.side_effect = chat
        recorder = RecordingClient(backend)
        result = run_summary_agent(graph, draft, topic="Obama", prompt=draft["summary_prompt"],
            client=recorder, expected_group_counts=counts, disparity_threshold=0.25, representation_metric="internal")
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
    def test_saved_scores_require_unchanged_inputs_and_exact_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            scores = root / "results/eval"
            scores.mkdir(parents=True)
            source = root / "Obama.json"
            source.write_text("{}", encoding="utf-8")
            header = "system,topic,prompt_id,seed,file,bart_src,bart_ref\n"
            (scores / "metrics.csv").write_text(header, encoding="utf-8")
            (scores / "bart.csv").write_text(header + "baseline,Obama,1,101,Obama.json,-3.0,-2.0\n", encoding="utf-8")
            records = [{"system": "baseline", "topic": "Obama", "prompt_id": "1", "seed": 101,
                        "file": str(source), "summary": "Original.", "sentences": ["Original."]}]
            with patch.object(prepare_report, "ROOT", root), patch.object(evaluate, "DATA", data), \
                 patch("prepare_report.subprocess.run", return_value=SimpleNamespace(returncode=0)):
                cache, origins = prepare_report.saved_metric_cache(records)
                self.assertIn(("bart", "Obama", "Original.", ("Original.",), None), cache)
                self.assertNotIn(("bart", "Obama", "Altered.", ("Altered.",), None), cache)
                self.assertTrue(next(iter(origins.values()))["mode"].startswith("historical"))
            with patch.object(prepare_report, "ROOT", root), patch.object(evaluate, "DATA", data), \
                 patch("prepare_report.subprocess.run", side_effect=[SimpleNamespace(returncode=0), SimpleNamespace(returncode=1)]):
                with self.assertRaisesRegex(ValueError, "unchanged committed"):
                    prepare_report.saved_metric_cache(records)

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
