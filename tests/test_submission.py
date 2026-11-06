"""The submitted inputs reproduce the numerical results claimed in the report."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from export_report_figures import export_figures
from ontology_tools import DEFAULT_ONTOLOGY
from reproduce_report import ROOT, accuracy, checker_tables, citation_probe, experiment_tables
from rerun_agent_trials import saved_drafts


class SubmissionTests(unittest.TestCase):
    def test_checker_validation_reproduces_report(self):
        result = checker_tables()
        self.assertEqual(result['table10_confusion']['SUPPORTED'],
                         {'SUPPORTED': 36, 'CONTRADICTED': 1, 'UNKNOWN': 57})
        self.assertAlmostEqual(result['table12_held_out']['Revised']['accuracy'], 105 / 132)
        self.assertEqual((result['mcnemar']['corrected'], result['mcnemar']['broken']), (21, 7))
        self.assertAlmostEqual(result['mcnemar']['p'], 0.012540951371192932)
        self.assertAlmostEqual(result['annotation_agreement']['Saif vs reference']['kappa'], 0.9)

    def test_reported_agent_and_proportion_outcomes_are_preserved(self):
        result = experiment_tables()
        self.assertEqual(result['table13_factual']['revised_final'],
                         {'SUPPORTED': 83, 'CONTRADICTED': 0, 'UNKNOWN': 127})
        self.assertEqual(result['summaries_changed'], {'original': 3, 'revised': 0, 'proportion': 1})
        self.assertEqual(result['table14_fairness']['draft']['passing'], 5)
        self.assertAlmostEqual(result['table14_fairness']['draft']['ec'], 0.0024, places=4)
        self.assertEqual(result['table16_proportion'],
                         {'already_reported': 9, 'proportional': 5, 'invalid_feedback': 2, 'disproportionate': 2})

    def test_missing_predictions_cannot_inflate_reported_accuracy(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            accuracy({'A': 'SUPPORTED'}, {'A': 'SUPPORTED', 'B': 'UNKNOWN'}, ['A', 'B'])

    def test_curated_drafts_and_all_replay_graphs_are_present(self):
        drafts = saved_drafts()
        self.assertEqual(len(drafts), 18)
        self.assertTrue(all(path.parent == ROOT / 'results/drafts' for path in drafts.values()))
        for path in (ROOT / 'results/agent_rerun').glob('*_prompt*_seed*.json'):
            result = json.loads(path.read_text(encoding='utf-8'))
            result = result.get('agent_result', result)
            graph = (ROOT / result['pipeline']['graph'].replace('\\', '/')).resolve()
            self.assertTrue(graph.is_relative_to(ROOT))
            self.assertTrue(graph.is_file(), path.name)
            self.assertTrue((ROOT / 'results/agent_final' / path.name).is_file())

    def test_figure_sources_are_derived_without_models_or_ontology_edits(self):
        before = hashlib.sha256(DEFAULT_ONTOLOGY.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory, \
             patch('ollama_backend.OllamaClient.chat', side_effect=AssertionError('No model needed')):
            out = Path(directory)
            export_figures(out)
            self.assertIn('route_checks', (out / 'figure1_agent.mmd').read_text())
            self.assertIn('style=dashed', (out / 'figure1_agent.dot').read_text())
            self.assertIn('"Person"', (out / 'figure2_classes.dot').read_text())
            self.assertIn('worksFor', (out / 'figure3_properties.dot').read_text())
        self.assertEqual(hashlib.sha256(DEFAULT_ONTOLOGY.read_bytes()).hexdigest(), before)

    def test_reported_citation_retrieval_is_reproducible_offline(self):
        result = citation_probe()
        self.assertEqual((result['status'], result['claims'], result['hits']), ('computed', 58, 38))


if __name__ == '__main__':
    unittest.main()
