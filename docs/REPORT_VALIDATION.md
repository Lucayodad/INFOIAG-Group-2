# Report validation and saved results

Actual final results and the controlled demonstration are recorded in
[validation results](../results/eval/report_release/validation.md).

A run that finishes with UNKNOWN claims is an uncertainty result, not a verified
summary. A small Equal Coverage deviation does not prove that a summary covers
any group well.

Agent outputs retain two summary views: `current_summary` is the official
result after rollback and stopping rules, while `best_summary` is the
best-retained checked iteration identified by `best_iteration`, with its
corresponding `best_metrics`. The latter is diagnostic evidence of progress
and is not an alternative accepted result.
The `attempted_candidates` output additionally records every generated
revision candidate, including candidates that were later rejected or rolled
back, together with its status, reason and checked metrics. These candidates
are diagnostic only and must not be presented as accepted summaries.

## Fixes after the 36-trial analysis

The saved 18 agent trials never changed a summary. Three causes were fixed:

- **Threshold scale.** Raw MNLI EC is bounded by the small absolute coverage
  (observed 0.002-0.015), so the 0.25 threshold could never be exceeded. The
  online check now routes on the four-fifths rule (`threshold_metric:
  four_fifths_coverage_ratio`): `disparity` is 1 - min/max group coverage and
  the default threshold is 0.2, so a group covered below 80% of the
  best-covered group fails. This is the established disparate-impact standard,
  chosen instead of a value fitted to these results; `equal_coverage` still
  reports raw EC. On the saved trials it would flag 9/18 summaries (min/max
  0.66-0.77: all six McDonalds trials, Obama prompt 2 seeds 101/202, and
  Superbowl prompt 2 seed 202).
- **Quantifier rule.** The factual prompt told Qwen that unestablished 'many' or
  'frequently' is UNKNOWN, which produced most of the 179/207 UNKNOWNs. Vague
  quantifiers are now judged on the cited tweets ('some': one or more; 'several',
  'many', 'frequently': at least three). Explicit majority/proportion claims
  still need corpus-scope evidence.
- **Revision quotations.** The only real revision was rejected because Qwen
  wrapped a correct quotation in quotation marks. Surrounding quotation marks
  are now stripped before the exact-substring check, and an invalid proposal
  gets one format repair with its error. The repair belongs to the same
  revision attempt, so the three-revision bound is unchanged.
- **HTML entities.** 22/270 tweets store entities such as `&amp;`; Qwen quotes
  them as `&`. `fact_check.quote_in()` unescapes both sides (and strips
  wrapping quotation marks) for the factual checker and the reviser alike.
  Any other difference from the original text is still rejected.

Two behaviour changes followed the first live reruns:

- **Retry after rollback.** A rechecked fix that is unsupported or regresses
  is rolled back and retried, within the three-revision bound, with the
  recheck's findings and the rejected text as feedback. A valid fix with no
  measured gain still stops (`no_improvement`).
- **No group labels in summary text.** The reviser is told that dialect labels
  are dataset annotations, and a replacement naming a group (e.g. "users in
  the AA group") is rejected and repaired.

## Changes for consistent execution

- `equal_coverage.calculate_equal_coverage()` is the shared MNLI implementation.
  Online checking and `evaluate.metric_ec()` use original summary sentence
  entries as hypotheses and original tweets as premises. Atomic claims remain
  the factual checker's unit. The version is `mnli_sentence_mean_v1`; do not
  combine legacy atomic-claim EC with new sentence EC.
- `representation_check._mnli_representation_check()` validates extraction
  completeness, source identities, dialect labels and group counts. Missing
  inputs, unavailable models and invalid probabilities yield UNKNOWN, with no
  invented point estimate. Probabilities are estimates, not semantic confidence
  intervals. The internal checker remains available with
  `--representation-metric internal`.
- `fact_check.compact_evidence()` labels extracted interpretations separately
  from original `quote_text`. `_validate_verdict()` rejects a verdict that
  disagrees with an explicitly named verdict in its explanation. This narrow
  consistency check is not a general semantic correctness test.
  `fact_agent.run_fact_agent()` retains one bounded repair and strict original
  substring citation validation, with clearer repair instructions. Format repair
  may retain a verdict or conservatively return UNKNOWN; it cannot flip to the
  opposing definitive verdict while ostensibly repairing formatting.
  `_loop_schema()` requires finalization after cited evidence is available
  (allowing explicit ontology lookups and subsequent tweet retrieval when
  configured). The default cited-tweet path permits only observed
  ID/original-quotation pairs. Optional entity lookups retain a compact reference
  schema and strict runtime quote validation. The 16,000-byte request guard now
  includes the response schema as well as messages. UNKNOWN remains available
  for insufficient or ambiguous evidence; no semantic judgment is forced.
- `model_usage.RecordingClient` records each actual chat call once, including
  extraction, repair, revision and rechecking. `run_summary_agent()` adds its
  agent-only ledger; `experiment_runner._agent_run()` records draft plus agent.
  Source preprocessing is excluded. Missing/failed usage makes total usage
  incomplete; known partial counts are separate. No monetary price is inferred.
- `evaluate.records_from_file()` accepts experiment wrappers and native
  LangGraph/pipeline states, preserving sentence entries, status and stop reason.
  Legacy incomplete agent token totals are blank in complete token columns.
- `evaluate.collect_records()` follows completed manifest outputs, handles
  Windows paths, excludes draft folders, deduplicates overlapping paths and
  rejects ambiguous repeated trials. Agent drafts are a separate category.
- `evaluate.append_rows()` migrates headers and atomically upserts rows.
  Changed summary hashes invalidate old scores. `report()` tolerates absent
  metric columns and reports measured sample counts for each metric.
- `prepare_report.py` verifies the saved 18-baseline/18-agent design, evaluates
  saved text, hashes inputs, audits revisions, and writes `metrics.csv`,
  `report.md`, `findings.md`, and `audit.json` in a dedicated folder. Identical
  texts share computations within the process. It does not generate summaries.
  `--reuse-saved-metrics` explicitly permits historical semantic scores only
  after checking committed inputs are unchanged; exact metric-input equality
  allows reuse across categories. Every row records the original CSV hash,
  source row and legacy model-version limitation. Reused scores are not fresh
  inference. The fresh spot checks are retained separately in `fresh_probe.json`.

The StateGraph, three-revision bound, retrieval, rechecking and rollback rules
are preserved. Original ontology, tweets and checkpoints are preserved. The
0.25 routing threshold remains exploratory and was not tuned to force revisions.

## Commands (PowerShell, repository root)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
Remove-Item Env:HF_HUB_OFFLINE
```

Tests use explicit internal fixtures or injected MNLI scores; they need no model
downloads. Mocked tests verify mechanics, not model accuracy.

`python validate_controlled_agent.py` runs the complete graph with synthetic
evidence and real local Qwen/embedding calls. It uses an explicit internal
coverage metric and 0.5 controlled-fixture threshold, saves its output to a new
file, and is separate from the 90-tweet experiment and its 0.25 threshold.

Evaluate the saved real trials, without Qwen generation:

```powershell
.\.venv\Scripts\python.exe .\prepare_report.py --out .\results\eval\report_release --metrics cost,rouge,gap,ec,bart --reuse-saved-metrics
```

First use downloads MiniLM, RoBERTa MNLI and BART to the normal Hugging Face
cache. `--metrics cost,rouge` works without downloaded weights. DeepEval remains
an optional separate evaluation, outside the default metrics/dependencies.
Remove `--reuse-saved-metrics` for a fully fresh model evaluation; that is much
slower on a CPU. Reuse does not need the Hugging Face models if all selected
scores are available from the verified committed inputs.

One real Obama pass, reusing a saved draft and source checkpoint:

```powershell
.\.venv\Scripts\python.exe .\run_pipeline.py --topic Obama --prompt-id 1 --reuse-draft .\results\agent_pc_mnli\experiments\20261102T123744Z_56b70559\drafts\ontology_agent_Obama_prompt1_seed101.json --disparity-threshold 0.25 --max-revisions 3 --output .\results\evidence\Obama_report_validation.json
```

The pipeline refuses an existing output; choose a new filename for another run.
It skips successful source checkpoints, constructs an isolated RDF graph, and
calls the existing agent. Usage is agent-only because the draft is reused;
this is not a new baseline/agent paired trial.

## Interpretation

The saved experiment has three topics, two prompts and three seeds per system.
All 18 agent finals equal their own initial drafts: 17 stopped on uncertainty
alone, one rejected its revision. These data support a limitations analysis,
not a claim of improvement from revision.

Report absolute group coverages alongside EC: uniformly low coverage can have
small EC. Separate agent-versus-own-draft comparisons from independently
generated baselines. Retain UNKNOWN and rejected revisions. Times from different
computers include loading and are not controlled speed comparisons. Legacy
agent token totals are incomplete. Three seeds provide descriptive variation.

The repository PDF is Intermediate Report 1 and does not establish a numeric
EC threshold. This update aligns the existing online/offline metric; it does
not claim a new validation against the absent Intermediate Report 2 text.
