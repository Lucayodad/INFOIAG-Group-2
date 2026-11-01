# Factual and representation checking

Both interfaces consume the populated asserted RDF graph and the list of five
sentence-extraction records produced by `summary_claims.py`. Each record contains
`sentence_id`, `sentence`, `status` and `claims`. `flatten_summary_claims` checks
identity, extraction status, quotations and citations, and returns claim copies
annotated with their sentence IDs. Input data and the graph are not changed.

```python
from fact_check import fact_check_summary
from representation_check import representation_check

facts = fact_check_summary(graph, extracted_summary_claims, client,
                           allowed_entity_ids=["Obama"], max_tool_calls=3)
representation = representation_check(
    graph, extracted_summary_claims, client, topic="Obama",
    disparity_threshold=chosen_threshold,
)
```

The caller owns the Ollama-compatible client. Results are JSON-serializable.
The [LangGraph controller](SUMMARY_AGENT.md) now connects extraction and these
checks, evidence-guided targeted revision and bounded rechecking. Experimental
evaluation metrics remain pending. The checking agents themselves are reused.

Build checking inputs with `populate_ontology`, which excludes illustrative
base tweets, claims (including named subclasses), assessments and summaries.
Schema and domain assertions remain available. Raw `OntologyTools.asserted`
is intended for ontology inspection and still contains the supplied examples;
it is not a complete per-run source-evidence graph.

New runs use `tweetHasText`, topic `SuperBowl` for dataset key `Superbowl`, and
`claimAboutTopic` links. Readers explicitly accept legacy `tweetText` and
`Topic_Superbowl` values without depending on OWL equivalence or identity
inference. Matching alias values do not add duplicate supporters. Conflicting
text values generate group integrity issues; factual retrieval preserves the
alternatives as `conflicting_text_values` and sets usable `text` to null, so
the existing completeness guard prevents a definitive verdict. Rebuild any
saved graph containing base examples; compatibility reads do not remove them.

## Factual workflow

`fact_check_summary` reuses the existing `run_fact_agent` for each extracted claim.
The bounded agent retains its model-selected tools, allowlist, observations,
validated references, completeness guards and call counts. Shared assessment
preparation/finalization in `fact_check.py` keeps the one-claim and agent paths
consistent, with no unrecorded retrieval.

`compact_evidence` ranks observed evidence by deterministic word overlap with
the current claim, then stable original identifier. Qwen sees at most 16 whole
entries and 8,000 UTF-8 catalogue bytes, with short `E001` identifiers. It does
not receive duplicate tool results or accumulating evidence messages. Original
RDF IDs, quotations and file/line provenance remain in the returned assessment;
`evidence_id_map` connects shown IDs to those originals. IDs remain stable as
observations accumulate. `expand_verdict` validates shown references and exact
quotations before restoring full identifiers. Unshown or invented IDs fail.

Factual message input is capped at 16,000 bytes. Selection records shown and
omitted counts and explicitly does not establish corpus frequency. All cited
evidence is still retrieved and the existing completeness guard remains binding.
An oversized entry is omitted whole. An oversized request stops UNKNOWN.
Each claim allows one extra format-repair call, including for empty explanations;
it cannot execute another tool. Persistent invalid output remains UNKNOWN. Thus
the agent allows at most `max_tool_calls + 2` model calls, with the original tool
limit unchanged. The one-shot `check_claim` uses the same compact view and one
repair. All verdicts require a nonempty explanation. Only an opposing fact
justifies CONTRADICTED: missing support, negative sentiment elsewhere and an
unestablished quantifier require UNKNOWN. This semantic distinction remains an
unreviewed model judgment, not something quote validation can prove.

The deterministic issue policy is:

- `SUPPORTED`: no factual issue.
- `CONTRADICTED`: `factual_contradiction`, severity `serious`.
- `UNKNOWN`: `factual_uncertainty`, severity `unresolved`.
- Unclear sentence extraction: an unresolved issue with null claim ID, because
  the sentence has no reliably extracted claim to assess.

Summary status is FAIL if any claim is contradicted, UNKNOWN if any issue is
unresolved or no claims were assessed, otherwise PASS. These are unreviewed
judgments relative to retrieved evidence, not independently established truth.
Counts and completeness never claim full-topic verification from partial data.

## Representation workflow

1. `get_group_claims` binds topic and literal dialect label in the existing
   `queries/source_evidence.rq` SPARQL query. It preserves tweets without claims,
   source quotations and file/line provenance. Integrity problems are explicit.
2. `match_summary_claims` globally groups only equal claim text, target and stance,
   allowing case/whitespace differences. It sorts stable candidate IDs and packs
   deterministic batches, at most `batch_size` (default 8) new candidates and at
   most 12,000 input bytes per request. Each exact group sends one original
   representative quotation; all supporter records stay in the Python result.
   Qwen proposes semantic-equivalence clusters in an equivalence-only task.
   Every candidate must occur exactly once and every merged candidate requires
   a supplied quotation. No summary is included in this phase.
3. Each batch is then reconciled against the global canonical registry built by
   earlier batches. Qwen must choose EQUIVALENT, DISTINCT or UNKNOWN for every
   incoming cluster in each registry page. EQUIVALENT requires quotations for
   both sides and the exact registry ID. Requests use original representatives,
   not generated summaries. Registry pages split deterministically by the byte
   cap; successful matches stop further comparisons for that incoming cluster.
   Different stances cannot merge and are skipped where possible. Distinct
   claims are never automatically merged because of words, names or targets.
   This compares batches to canonical representatives, not individual original
   claim pairs. The all-distinct worst case still requires more registry pages.
   Once equivalence/reconciliation is done, a separate summary-coverage task
   assesses each canonical representative as MATCHED, NOT_MATCHED or UNKNOWN.
   This task must return singleton candidates; it cannot merge or split clusters.
   Summary IDs and quotations are validated independently. Unresolved equivalence
   keeps the affected candidates UNKNOWN and prevents definitive coverage bounds.
4. Supporting tweet IDs/group labels come from RDF, never from the model. Every
   canonical claim's group weight is its number of distinct supporting tweets.
   Multiple claim records from one tweet do not inflate that weight.
5. `check_summary_fairness` computes weighted coverage and disparity in Python.
   MATCHED contributes its weight, NOT_MATCHED contributes none, UNKNOWN makes
   coverage an interval `[matched_weight / total, (matched + unknown) / total]`.
   Unknown claims are not listed as omissions. Zero denominators are undefined.
6. Disparity bounds preserve the fact that an unknown canonical claim's `m_c`
   is shared across groups. For every ordered group pair, its known coverage
   difference is augmented by each unknown claim's normalized weight difference:
   negative differences give the pair's lower bound, positive differences its
   upper bound. The maximum pair lower bound (or zero) is a conservative lower
   bound on D; the maximum pair upper bound (or zero) is its upper bound. These
   are conditional on the returned unreviewed claim partition. PASS requires
   upper disparity <= threshold; FAIL requires lower disparity > threshold;
   otherwise status is UNKNOWN. A point disparity is returned when the bounds
   coincide, even if individual coverages remain uncertain.

There is **no paper-established threshold hardcoded**. The default is None, which
returns UNKNOWN with `threshold_not_configured`; callers must select and record
their research setting. Confirmed disparity produces serious representation
issues; unresolved validation produces unresolved issues. Omitted claims are
listed separately and do not individually imply unfairness.

Data completeness defaults to the actual dataset's 30 tweets for each of AA,
Hisp and White. Missing groups, incomplete fields, invalid labels and unclear
extractions prevent a definitive full-topic result. Controlled synthetic data
must explicitly set `expected_group_counts`, for example 3 per group in the
included fixture. Sample graphs do not silently redefine a complete topic.

## Bounds and limitations

The public interface and report keys are preserved, with an additional optional
`batch_size=8` keyword. The default model-call budget increases from 1 to 64 so
multi-batch topics work by default. An explicitly supplied budget, including 1
or 0, remains binding. Each request still has a 12,000-byte input limit including
instructions, a 16,384-token context, and 3,000 output tokens. The byte limit is
per request, not per topic. A separate 30,000-byte cap includes the complete JSON
request and its reference-constraining schema. Batches reserve 400 bytes for
repair instructions. Input, quotations and representative claim text are
never silently truncated.

```python
representation = representation_check(
    graph, extracted_summary_claims, client, topic="Obama",
    disparity_threshold=chosen_threshold,
    batch_size=8, max_model_calls=64, max_input_bytes=12000,
)
```

Every attempted call is counted, including malformed/failed replies. The trace
records request evidence, input size, phase, raw response, validation failures
and token usage when available. Request-specific JSON schemas constrain generated
references to the supplied ID/full-quotation pairs; independent Python validation
still rejects wrong pairs, unshown IDs, missing references and malformed results.
Every invalid batch gets one bounded repair; both attempts count against 64.
A failed equivalence, reconciliation or coverage task preserves earlier results
and continues independent batches. The trace distinguishes `semantic_equivalence`,
`cross_batch_reconciliation` and `summary_coverage`, with attempts and errors.
All original members survive; unprocessed/failed claims remain UNKNOWN. Budget exhaustion
or an oversized individual claim/comparison yields explicit UNKNOWN, not an
invented omission. There is no whole-topic 12,000-byte rejection.

`probe_checker_reliability.py` performs two focused live probes against the saved
Obama pilot: original factual claim S01_C01 and the eight propositions from
original representation batch K009-K016. It preserves original source IDs and
quotations. It does not run LangGraph or revise a summary. Use a new output path
for every invocation; existing outputs cannot be overwritten.

`data_complete=false` identifies missing/invalid topic input or unclear extraction.
Complete input with budget exhaustion, uncertain equivalence or matching failure
keeps `data_complete=true` and explains semantic uncertainty separately in
`matching`, `reasons` and the trace. Incomplete input skips model calls and reports
`incomplete_input_data`/`semantic_assessment_not_run`.

Exact duplicates are deterministic; paraphrase equivalence, matching, sarcasm
and extraction quality still require human validation. Model partitions and
matches stay unreviewed. UNKNOWN equivalence prevents definitive coverage even
if a provisional partition's numeric estimates look decisive. Coverage bounds
on incomplete graphs describe observed claims only, not the unseen corpus.
When equivalence itself is unresolved, denominators may change after merging:
defined group bounds are conservatively widened to [0, 1], disparity to [0, 1],
and point coverage/disparity are null. Tighter shared-claim bounds apply when the
canonical partition is established but MATCHED/NOT_MATCHED remains uncertain.
Canonical decisions are not a proof of semantic equivalence: model errors can
still split or merge viewpoints incorrectly. Greedy canonical assignment uses
original representatives and deterministic order, but model quality must be
evaluated independently. More source claims, longer claims or many distinct
viewpoints can exhaust a chosen call budget even in a 90-tweet topic.

This is the report's internal weighted claim-coverage calculation. It is not the
entailment-based Equal Coverage metric used for final experimental evaluation.

The functions are called by `summary_agent.py` LangGraph nodes, with a read-only
RDF wrapper supplied outside state. Integration does not establish model
accuracy or full-dataset experimental readiness.
