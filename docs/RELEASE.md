# Final engineering release

These are earlier release notes. See [report validation](REPORT_VALIDATION.md)
for the current MNLI default, factual schema fixes, report tables and validation
results. Internal semantic batching described below is an optional metric.

This is a runnable research prototype, not a completed scientific evaluation.
The original OWL ontology, datasets, teammate commits and existing stage CLIs
are preserved. No main-branch merge or 36-run evaluation is part of this release.

## Remaining bottleneck fixed

The saved Obama v2 pilot spent all 64 representation calls on equivalence and
whole-registry reconciliation, leaving coverage unprocessed. The release removes
those registry scans:

1. Keep deterministic exact duplicate grouping.
2. For larger inputs, embed interpreted propositions/targets/stances locally
   with existing `all-minilm`, in chunks of 32, caching vectors across rechecks.
3. Retrieve at most two earlier same-stance candidates with cosine >= 0.8.
4. Ask Qwen to verify only these alternatives. Validate IDs, quotations, stance
   and membership in each candidate's actual shortlist before merging.
5. Unite only validated equivalences, retaining every original claim/supporter.
6. Reserve half the chat budget for independent summary-coverage requests.
   Failed batches get at most one repair; UNKNOWN remains UNKNOWN.
7. Calculate coverage, distinct-tweet weights, shared-unknown disparity bounds
   and threshold decisions with the unchanged Python formulas.

Live embedding retrieval on the existing Obama evidence produced 21 incoming
queries and 26 proposed alternatives for 166 exact candidates, using six small
embedding calls. This replaces expensive repeated whole-registry requests.
Coverage has a protected budget; equivalence cannot consume all 64 calls.

Shortlisting is approximate. A low score or an unselected neighbor is not proof
of semantic difference. Reports record `equivalence_scope=embedding_shortlist_only`
and `global_equivalence_exhaustive=false`. Numeric coverage remains conditional
on the resulting **unreviewed partition**; genuine equivalence uncertainty still
widens bounds. These are engineering retrieval settings, not new research metrics.

## Final stabilization

Small candidate collections that exceed the request byte cap now use the same
bounded embedding shortlist as larger collections. Independent byte-split local
partitions can no longer silently miss cross-request equivalences. Reused drafts
must carry matching topic and prompt provenance; missing or contradictory prompt
metadata is rejected, and the saved summary is rebuilt from validated sentences.
Seven regression tests cover these release fixes. Direct test dependency
`httpx2` is declared, and previously tracked Python bytecode is removed.

## One complete CLI

`run_pipeline.py` lazily loads dependencies, checks the local models, loads the
chosen topic, resumes source extraction, populates RDF, creates a five-sentence
draft and calls the existing `run_summary_agent`. It reuses
`resume_source_extraction`, `populate_ontology`, `generate_summary` and the existing
LangGraph nodes and revision safeguards. It does not duplicate those agents.

The source-extraction CLI now calls the same reusable checkpoint function; its
old arguments/defaults and retry/checkpoint behavior remain. Graph/draft/result
directories are created automatically. Outputs include intermediate artifacts,
model digests, complete final state and structured errors. Successful source
checkpoints are reused; final output files cannot be accidentally overwritten.

Install `requirements.txt` in a normal virtual environment. LangGraph and the
direct `typing_extensions` import are declared. No external dependency folder,
developer directory, VS Code setting or saved pilot is required. Historical
probe scripts explicitly require their diagnostic input paths; they are optional
and are not the complete-run entry point.

## Release verification

- The complete suite passes **210 tests** in the project virtual environment,
  with no added source path or external dependency folder.
- The complete fresh-input synthetic 90-tweet pipeline exercises source claims,
  RDF, initial generation, factual/representation checking, conditional routing,
  justified revision, both rechecks and persisted results; checkpoint resume
  does not repeat previously successful source calls.
- A 166-distinct-viewpoint fixture reaches every coverage batch with 21 Qwen
  calls, below the binding 64-call budget. Other fixtures verify semantic merges,
  distinct supporters, UNKNOWN, malformed replies, phase budgets and rollback.
- Missing dependencies print the requirements installation command; missing
  models print `ollama pull` commands; an unavailable server prints `ollama serve`.
- `run_pipeline.py --help`, existing CLI imports, syntax, dependency consistency
  and Git whitespace checks are verified before publication.
- A clean export of the staged source files, without generated results or a
  virtual environment, passes CLI/import checks and all 210 tests using the
  declared dependencies. `PYTHONPATH` is removed for that verification.
- RDFLib/pyparsing emit existing deprecation warnings; tests still pass.

The README contains fresh-clone instructions, topic commands, outputs, resuming
rules and known limitations. Its 0.25 disparity threshold is explicitly exploratory.

## Real Obama smoke result

One real local Qwen/Ollama run completed from the release CLI, using all 90
Obama tweets, the existing validated 166-claim source checkpoint, a newly
populated RDF graph and a **freshly generated** Prompt 1 summary. No saved draft
was reused. Smoke settings were `--max-revisions 0 --max-model-calls 32
--max-tool-calls 1`, with threshold 0.25. The saved result is local and gitignored:
`results/evidence/Obama_release_smoke.json`, with call progress in the adjacent
`.log` and intermediate evidence in `Obama_release_smoke_artifacts/`.

- Runtime on the local machine: approximately 55 minutes.
- Source completeness: 30 AA, 30 Hisp, 30 White; all 166 source claims retained.
- Factual results: 2 SUPPORTED, 0 CONTRADICTED, 8 UNKNOWN. One finalized response
  remained invalid after format repair; the other unknowns were explicit
  uncertainty. No unsupported definitive verdict was substituted.
- Representation: **22 chat calls**, consisting of 5 equivalence attempts and
  **17 completed, validated coverage batches**, plus 6 embedding calls. This
  reached coverage without exhausting either the 32-call smoke budget or the
  normal 64-call limit, unlike the previous pilot.
- The provisional partition has 163 clusters: 2 MATCHED, 134 NOT_MATCHED and
  27 UNKNOWN. Coverage evaluated all 136 eligible clusters; unresolved clusters
  remained UNKNOWN. All original supporters survive the grouping.
- Two equivalence batches failed both validation attempts: one attempted an
  out-of-shortlist target, and others omitted required quotations. The guards
  rejected these responses. The accepted batch also contained five genuinely
  uncertain equivalence decisions. These are distinguished in the saved trace.
- Consequently all group coverage bounds remain [0, 1], point coverage and
  disparity are UNKNOWN, and final status is `completed_with_uncertainty` with
  stop reason `uncertainty_only`. No revisions were attempted in this check-only
  smoke; justified revision/rechecking/rollback are exercised by synthetic tests.

This verifies completion, bounded model calls, preservation, error handling and
final JSON persistence. It **does not** establish numerical fairness or semantic
accuracy: the live rejected responses and uncertain equivalences remain model
limitations requiring review. Numeric coverage is verified when evidence and
matching are sufficient in the controlled tests; the real unknowns were not
forced into values for release. No 36-run research grid was executed.

## Final focused real-model probes

After final stabilization, two additional probes used the installed local Qwen
model without mocked responses. Results are saved locally and gitignored at
`results/evidence/final_release_focused_probes.json`.

- Complete controlled 90-tweet evidence with a deliberately small 5,000-byte
  request cap reached coverage in six Qwen calls and one embedding call. Three
  replies omitted required quotations and one was truncated; validation rejected
  all four. The result remained UNKNOWN, with unresolved equivalence preserved.
- Three original Obama tweets supplied six source claims, including the separate
  propositions that Obama would not ban all firearms and would ban assault
  rifles. Equivalence and coverage both returned validated responses in two Qwen
  calls. All six claims survived in separate clusters; the gun-policy scopes
  were not merged. Coverage yielded one MATCHED, four NOT_MATCHED and one UNKNOWN.
  This small probe is not a full-topic fairness measurement.
- A fresh temporary virtual environment installed only `requirements.txt`. An
  export of the staged source, excluding generated results, passed imports,
  CLI help, dependency checks and all 210 tests with `PYTHONPATH` removed.

## Scientific limitations

Embeddings can miss paraphrases. Qwen can misinterpret sarcasm, idioms, attribution,
stance, negation, quantifiers and coverage granularity despite valid provenance.
Source-claim errors can propagate into later checks. Matching/partition choices
remain unreviewed model judgments. Local runtimes may be slow or time out.
UNKNOWN cannot be reported as verified support, omission, fairness or improvement.
Only reviewed/evaluated results should be used as research findings.

## Live handoff monitoring

The complete CLI now records and prints a live observation feed. It includes
actual LangGraph stage handoffs, bounded model request/reply previews, validated
factual decisions, tool guard blocks, returned evidence, representation
validation/repairs, routing, candidate revisions and later acceptance/rollback.
Model replies remain explicitly pending until checked. A 30-second heartbeat
shows a call is still active. JSONL is flushed immediately and its path is saved
in the result manifest. Full evidence remains in the ordinary final result.
Twelve tests cover streaming, ordered concurrent events, heartbeats, observer
failures, unmodified transport, real graph handoffs, revision outcomes and
identical final state/model-call counts with observation enabled.
