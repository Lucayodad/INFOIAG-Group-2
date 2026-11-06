# Report validation and saved results

Actual final results and the controlled demonstration are recorded in
[validation results](../results/eval/report_release/validation.md).

**Fact checker revised on 2026-11-05** ([checker evaluation](CHECKER_EVALUATION.md)):
against reference labels the original checker confirmed 38% of supported claims
and all 3 of its CONTRADICTED verdicts in `results/agent_compliant_v2` were false;
the revised checker reaches accuracy 0.80 on held-out topics (was 0.69). Results
made with the original checker (agent_compliant_v2, the poster) are superseded by
`results/agent_checker_v2`.

A run that finishes with UNKNOWN claims is an uncertainty result, not a verified
summary. A small Equal Coverage deviation does not prove that a summary covers
any group well.

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

## Dialect labels are evaluation-only (feedback on Intermediate Report 2)

The graders: the dialect annotations may not be fed to the LLM, and using them
inside the agent uses test data for training. Every earlier agent run did both:
the reviser's prompt carried each evidence tweet's `dialect_label` and the task
"Improve the group's MNLI Equal Coverage", and the online group check that
routed and accepted revisions was the same `calculate_equal_coverage` with the
same labels as the final evaluation. Their revision results are therefore not
reported as agent results.

- **No group check in the agent.** `representation_metric=None` is the default
  of `build_summary_agent`, `summary_agent.py`, `run_pipeline.py` and
  `experiment_runner.py`; the graph is extract -> facts -> route -> retrieve ->
  revise -> recheck. The agent revises only contradicted claims (UNKNOWN alone
  still never triggers a revision). `mnli`/`internal` keep the earlier
  label-based check for reproducing the recorded runs only.
- **No label in the agent's knowledge.** The agent's read-only copy of the RDF
  graph omits every `dialectLabel` triple; revision evidence is read with a
  label-free query (`revision_retrieval.topic_tweets`), and no row or task sent
  to the reviser has a label field. The reviser's instructions name no dialect.
- **Checked, not trusted.** `run_final_agent.py` passes every Qwen and embedding
  request through `LabelFreeClient`, which stops the run if a request contains
  a label (`dialect`, or the string "AA", "Hisp" or "White").
  `tests/test_label_free.py` pins this on the recorded graphs and first checks.
- **Labels only at the end.** `evaluate_label_free_run.py` joins the dataset
  labels after the agent has finished and compares each final summary with its
  own draft: fact verdicts and MNLI Equal Coverage per dialect group.
- **Gate.** The summary may gain no contradiction and no issue in total, so a
  contradiction that becomes UNKNOWN is progress (McDonalds P1 s202 was wrongly
  rejected for this before).
- **Repair evidence.** A factual repair gets all of the sentence's own cited
  tweets (within the 9-tweet and byte limits), not only the 3 most similar to
  the contradicted claim, and is told to keep the uncontested viewpoints with
  their support. In the first label-free run (`results/agent_compliant`) the
  Superbowl P1 s202 repair turned "many" into "some" but lost the tweets behind
  the sentence's SUPPORTED claim, so it was rightly rejected.

Unchanged and compliant: draft/baseline generation and source-claim extraction
send Qwen only tweet IDs and texts, and the fact checker's evidence view
(`compact_evidence`) never included a label. The ontology is populated
automatically from each run's input tweets by this extraction (claim, target,
stance and an exact source quote per tweet), not by hand.

## Label-free viewpoint check (optional, `--viewpoint-check`)

The agent's reflection on its own bias (Lecture 5) without dialect labels.
After the fact check, `viewpoint_check.py` reads the stances of the claims
extracted from each tweet's text: a viewpoint (positive or negative) held by
at least 25% of the topic's tweets must be reported by a summary claim that
is not CONTRADICTED. It is off by default, so the results above reproduce
unchanged; `results/agent_compliant_v2` was run without it.

- **Open world.** If a sentence's claims could not be extracted, an unreported
  viewpoint is UNKNOWN, not missing: uncertainty alone never triggers a revision.
- **Order.** A missing viewpoint is repaired only once no contradiction is left.
- **Repair.** The rewritten sentence is a no-claim sentence if any, else the one
  with the fewest SUPPORTED claims, never the only sentence reporting another
  viewpoint. It keeps its own cited tweets and gains the 3 most similar tweets
  expressing the viewpoint; the replacement must cite one of them.
- **Gate.** No reported viewpoint may be lost, and a missing one counts as added
  only when a SUPPORTED claim reports it. A failed viewpoint repair is retried
  with that finding (within the 3 attempts).
- **Profiling.** With the check on, the reviser's prompt starts with a role
  profile (an impartial editor); the default prompt is unchanged.
- **Evaluation.** `evaluate_label_free_run.py` recomputes the check for draft and
  final of any run.
- **Known flaw, to fix before using it.** The summary side reads the stance of
  each extracted summary claim, but the extractor labels a claim that *reports*
  an opinion as neutral ("Some individuals express genuine admiration",
  "a minority of the tweets offer supportive commentary"). In
  `results/agent_compliant_v2` the check flags 4 of 18 final summaries, but on
  reading them only Superbowl P1 s101 truly reports no negative viewpoint;
  Obama P1 s101, Obama P1 s303 and Superbowl P1 s303 do report the flagged
  viewpoint. The live trial (Obama P1 s101) confirmed it: Qwen kept the
  sentence (it already reports admiration) and the gate rightly accepted no
  rewrite. Fix: classify the viewpoints a sentence reports with a dedicated
  reflection call instead of the extraction's stance field. Do not report the
  check's counts until then.

    python run_final_agent.py --viewpoint-check --out results/agent_viewpoint
    python evaluate_label_free_run.py --run results/agent_viewpoint

`tests/test_viewpoint_check.py` pins the check, the task, the gate and the
reviser guard, and that the default agent is unchanged.

## Proportion feedback layer (optional, `--proportion-check`)

The bias improver asked for at the poster session, done implicitly: no group
is computed, read or named. After the fact check, `proportion_check.py` gives
the summary one feedback layer (reflection, Lecture 5): Qwen reads every tweet
of the topic (ID and text, read label-free from the graph) and the five
sentences, and judges whether each main viewpoint gets roughly the weight of
the tweets expressing it. Its prompt says every tweet counts equally, whatever
its position in the list, length, spelling, slang or grammar. It is off by
default, so the results above reproduce unchanged.

- **Output.** An assessment written first, then `proportional`. It is false
  only when Qwen can name a concrete opinion that many tweets hold and no
  sentence reports; it then states that opinion in at most 15 words and the
  sentence to rewrite. It names no tweets: in the first live trial it listed
  Obama_001 to Obama_020, the first 20 of the list (the position bias this
  layer is meant to remove), and gave a long description that the reviser
  then copied verbatim. All three rewrites were rightly rolled back.
- **Open world.** Feedback naming no opinion, or still malformed after one
  format repair, is UNKNOWN, which never triggers a revision.
- **Verified before acting.** Qwen proposes, the verified evidence decides:
  the agent finds the topic's 3 tweets closest to the stated opinion
  (all-minilm, wherever they are in the list). If a SUPPORTED summary claim
  already quotes one of them, the summary already reports it: UNKNOWN
  (`already_reported`), nothing is rewritten. In the second live trial Qwen
  flagged "Obama is frequently compared to the devil, Satan, or the
  Anti-Christ", which sentence S02 already reports (claim S02_C02, SUPPORTED,
  quoting Obama_008, 044 and 061); all three rewrites were rightly rolled
  back, and this check now stops it before any rewrite.
- **No group named.** Feedback that names a group of authors or a way of
  writing is rejected (the reviser's `GROUP_LABEL` guard), and the repair
  request never echoes it, so `LabelFreeClient` sees no label.
- **Order.** Contradictions are repaired first, as for the viewpoint check.
- **Repair.** The feedback's chosen sentence is rewritten (else the least
  supported one). Evidence is the same 3 closest tweets, then the sentence's
  own cited tweets. The replacement must
  cite one of the 3, is told to describe it in its own words and not copy the
  description, and the reviser gets the role profile.
- **Gate.** As before, plus: the opinion counts as added only when a
  SUPPORTED claim of the rewritten sentence quotes one of those 3 tweets
  (from the fact checker's references), and only when the sentence's text
  changed: new citations alone change nothing a reader sees (the reviser is
  told so and retried). Otherwise the attempt is rolled back and retried with
  that finding (within the 3 attempts). After an accepted rewrite the
  feedback runs again on the new summary.
- **Third live trial (before the text-change rule).** Obama P1 s101: the
  devil finding was `already_reported`, no rewrite, 1.0 min. Superbowl P1
  s101 (the one draft truly lacking a negative viewpoint): Qwen flagged that
  most users are not watching because their teams lost; S02 "Several
  contributors expressed strong opinions about specific teams, ..." became
  "Some users expressed disinterest in the Super Bowl, citing reasons such as
  team losses or a desire for other activities like eating cookies."
  (SUPPORTED claim quoting Superbowl_010, "This superbowl dont even interest
  me"). A second accepted attempt only added citations to the same text,
  which the text-change rule now rejects. 9.9 min. Evaluated afterwards with
  the labels (`evaluate_label_free_run.py`): Equal Coverage 0.0023 -> 0.0004,
  coverage AA 4.11% -> 4.86%, Hisp 3.99% -> 4.85%, White 4.56% -> 4.76%
  (worst group 3.99% -> 4.76%, four-fifths disparity 12.6% -> 2.1%),
  CONTRADICTED 0 -> 0, SUPPORTED 1 -> 2. One trial, so an illustration, not
  a result. The stance-based viewpoint metric still lists "negative" as
  missing: its known flaw (a reported opinion is extracted as neutral).
- **Measurement stays independent.** The agent acts on Qwen's judgment; the
  report measures Equal Coverage with MNLI and the dialect labels, joined
  only by `evaluate_label_free_run.py`. Qwen's judgment is not validated
  against humans; treat it as feedback, not as a measurement.
- **Cost.** One extra Qwen call over all 90 tweets per check, plus one per
  candidate rewrite.

    python run_final_agent.py --proportion-check --out results/agent_proportion
    python evaluate_label_free_run.py --run results/agent_proportion

`tests/test_proportion_check.py` pins the feedback parsing and its open-world
rule, the label-free request (Qwen names no tweets), the verification
(including a replay of the second live trial), the task and its retrieval
order, the gate, the reviser guard, and that the default agent is unchanged.

## Fixes after the 18-trial rerun (`results/agent_rerun`)

Every failed revision attempt of the rerun (36 attempts) was traced to the
piece that broke it: reviser 18, acceptance gate 11, retrieval 4, the
four-fifths ratio 3; three more trials never revised because a failed claim
extraction blanked the group check. `tests/test_real_pieces.py` replays these
recorded states, candidates and graphs against the current code.

- **Acceptance gate** (`compare_revision`). A changed sentence must contain no
  CONTRADICTED claim, keep at least one SUPPORTED claim and raise no more
  issues than the sentence it replaces; one leftover UNKNOWN phrase no longer
  voids a repair (10 attempts, incl. both poster scenarios). The worst-covered
  group's coverage must not fall, and progress means fewer contradictions or a
  higher worst-group coverage. The four-fifths ratio still routes revisions,
  but as an acceptance test it accepted levelling down (every group lost
  coverage, Superbowl P2 s101) and rejected fixes that lifted every group
  (McDonalds P1 s303).
- **Copy guard.** A replacement that copies a cited tweet (verbatim, or 85% of
  its word trigrams) is rejected with a request to summarize in its own words
  (7 attempts pasted a tweet; one was accepted). An invalid proposal is now
  retried with its error while attempts remain, instead of ending the trial.
- **Group retrieval.** Group-coverage evidence is ranked by similarity to the
  sentence being rewritten, not to "Represent viewpoints from the X group",
  which returned tweets on unrelated topics (cosine to the sentence 0.12-0.37
  vs 0.26-0.59).
- **Claim extraction.** A summary sentence's failed extraction is retried once
  with its validation error, and an unclear extraction no longer makes the MNLI
  group check UNKNOWN (it scores sentence text only; 4/6 Superbowl trials).
- **Retries.** At temperature 0 a retry repeated the rolled-back sentence, so
  retries sample at temperature 0.7 with seed 42 + attempt; first attempts are
  unchanged. Rollback feedback for a group regression now includes the group
  coverages.
- **Caching.** Unchanged sentences are not re-extracted or re-checked, and each
  (tweet, sentence) MNLI pair is scored once per run: all 133 recorded rechecks
  of unchanged sentences were identical, and the fact checker took 79% of a
  trial's model time. `run_final_agent.py` also reuses each recorded first
  check of the same draft and graph.

Open: every atomic claim is checked against all tweets its sentence cites, so
the checker judges e.g. "others express disgust" next to the tweet cited for
"some find it delicious" (marked as an expected failure in the tests).
Restricting a claim to its closest cited tweets is not a safe fix: it made a
vague "significant number" claim SUPPORTED from three tweets.

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
