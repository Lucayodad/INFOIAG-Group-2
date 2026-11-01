# Checker reliability after the first Obama pilot

The original pilot is preserved at `results/evidence/Obama_prompt1_pilot.json`.
Its initial factual pass had five protocol-invalid UNKNOWN results: one missing
explanation and four responses referencing local claim names instead of the
catalogue's full RDF identifiers. Some explanations also confused insufficient
support with contradiction. S01_C01's evidence dictionary alone was 170,010
characters. Representation stopped at its second batch after mixed-stance merges
and missing candidates, leaving later batches unprocessed.

## Functions and connections

| Function | Change and connection |
| --- | --- |
| `fact_check.compact_evidence` | New pure context selector. It assigns stable short IDs before deterministic lexical ranking, retains whole entries within 16 items / 8,000 UTF-8 bytes, and returns a validated mapping and selection counts. Source-claim entries include their original tweet context and are omitted if their quote is absent from that tweet. Full evidence stays in the assessment. This does not replace revision's embedding retrieval. |
| `fact_check.expand_verdict` | New boundary between Qwen's short references and existing full-ID outputs. It rejects invented/unshown IDs and nonliteral quotes using the existing validator, then restores original identifiers. For existing callers, a full ID is accepted only if it belongs to the shown selection. |
| `fact_check.check_claim` | The standalone factual API uses these same helpers and allows one format repair. Persistent invalid output still raises `FactCheckError`; the bounded agent instead records UNKNOWN. |
| `fact_agent._loop_schema` | Constrains generated references to shown short IDs and requires nonempty string explanations when present. The initial no-evidence schema allows an empty references array without an invalid empty enum. Python still requires nonempty explanations for every finalized verdict. |
| `fact_agent.run_fact_agent` | Reuses the original model-selected tools and completeness guards, replaces accumulated/duplicated observations with a compact current view, caps message input at 16,000 bytes, and permits one format-repair decision. Repairs cannot execute tools. Counts, full observations, short-ID mapping, rejected attempts and input sizes remain auditable. |
| `representation_check._parse_equivalence` | Validates the new equivalence-only shape by adapting it to the existing strict partition, stance and quotation validator. No coverage judgment is accepted in this phase. |
| `representation_check._request_schema` | Builds evidence-constrained schemas for equivalence, reconciliation and coverage. Coverage candidates must be singleton canonical representatives, so coverage responses cannot change the partition. |
| `representation_check._request_matching` | Preserves model settings and budgets, records complete request size, and rejects requests exceeding 30,000 JSON bytes. Transport/malformed-response errors remain distinguishable in the trace. |
| `representation_check._validated_request` | New shared bounded retry: validate, retry once with a short error instruction, then return an unresolved task. Both attempts consume the shared budget; errors never abort independent batches. |
| `representation_check._merge_clusters` | Combines equivalence provenance and distinct supporters only. Coverage is assessed afterwards, avoiding competing coverage decisions made before canonical grouping. |
| `representation_check.match_summary_claims` | Runs deterministic exact grouping, local equivalence, cross-batch reconciliation, then independent summary coverage. Its nested batching helper reserves repair space and preserves stable IDs. Failed/unprocessed claims stay UNKNOWN; uncertain partition denominators cannot produce definite coverage. |
| `demo_summary_agent.mock_qwen` / `demo_revision_loop.mock_client` | Mock replies now follow the separated protocols and compact factual observations. These are fixture adaptations, not changes to revision behavior. |
| Test helpers `split_matching_reply` / `oracle_client` | Adapt existing synthetic gold labels into separate equivalence and canonical-coverage replies. Fault injection happens after constructing a valid response for its phase, exercising the real validators and retry behavior. |
| `probe_checker_reliability.main` | New opt-in script for original S01_C01 and the eight propositions in failed K009-K016 only. It uses real Qwen, preserves source IDs/quotes, checks that the graph and pilot bytes are unchanged, and exclusively creates a separate JSON report. The filtered representation probe renumbers K009-K016 as K001-K008 and records that fact. |

The factual instructions now explicitly distinguish opposing facts from missing
support. Negative sentiment elsewhere cannot refute an existential claim, and
failure to establish "many" or "frequently" requires UNKNOWN. Equivalence prompts
preserve proposition, stance, negation, attribution, modality and scope, with
examples drawn from the failed pilot batch. An all-singleton partition may be a
complete assessment; finding no equivalent pairs does not imply uncertainty.
Coverage instructions distinguish a definite omission (no summary proposition
represents the viewpoint) from genuine ambiguity. UNKNOWN remains available for
ambiguous meaning or coverage; Python never changes UNKNOWN into NOT_MATCHED.

## Unchanged safeguards and limits

`lookup_fact`, `lookup_entity_facts`, `get_group_claims`, exact grouping, factual
completeness, strict quote validation, distinct-tweet weights, fairness bounds,
disparity arithmetic and the LangGraph/revision architecture are unchanged.
Representation retains a binding default 64-call budget and 12,000 message-input
bytes per request, including instructions and any repair. A 400-byte reserve
allows repair instructions without exceeding that input limit. Whole evidence
is omitted when it cannot fit; source text is never silently truncated.

All model judgments remain unreviewed. Literal quotations prove provenance,
not entailment or equivalence. Lexical context selection can omit relevant
evidence. Unresolved equivalence still widens group coverage and disparity to
conservative bounds; UNKNOWN never becomes an omission or a measured improvement.
Large collections of distinct propositions can still exhaust reconciliation's
bounded call budget. No full-topic performance claim follows from small probes.

## Verification

The regression suite covers short-ID restoration, nonexistent and unshown IDs,
empty explanations and bounded repair, compact large tool observations, stable
IDs across observations, incomplete partitions, invalid stance merges, continued
processing, separate semantic tasks, request/call budgets and conservative UNKNOWN.
Existing complete synthetic LangGraph/revision-loop checks still exercise both
checkers, supporter weights, coverage bounds, rollback and rechecking.
The complete suite passes: **181 tests**, including the final coverage-instruction
change. Syntax and Git whitespace checks also pass. Existing RDFLib/pyparsing
deprecation warnings remain; they are not checker failures.

Run the complete suite in the project's configured Python environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Run only the two focused live probes (choose a fresh output path):

```powershell
.\.venv\Scripts\python.exe probe_checker_reliability.py --output results/evidence/checker_probes_new.json
```

The default timeout is the existing Ollama client's 300 seconds per request.
This script does not run LangGraph, generate a draft, revise a summary or run the
90-tweet experiment. Mocked regression results must be distinguished from the
separate live probe reports.

## Saved live verification

`results/evidence/checker_reliability_probes_v3.json` records a successful live
factual S01_C01 assessment and the separated representation tasks. The factual
agent made two model calls and one read-only retrieval, selecting 16 of 250
evidence entries. Its largest factual message input was 11,344 bytes; the final
call used 2,893 input tokens. It returned a nonempty, valid UNKNOWN explanation
because the evidence does not establish "frequently", rather than asserting
CONTRADICTED or failing on local RDF names. No format repair was needed. The
original pilot's catalogue alone was 170,010 characters and its final input
usage was 8,194 tokens. These sizes describe different boundaries: catalogue
versus complete messages; the point is the now-enforced request bound.

The eight propositions from original K009-K016 produced a complete, validated
all-singleton partition and proceeded to a separate coverage call, two calls in
total with no repair or failed quotations. Initial coverage was one MATCHED and
seven UNKNOWN. Several UNKNOWN explanations described definite absence rather
than ambiguity, motivating the explicit coverage-label definitions above.

`results/evidence/checker_reliability_representation_final.json` records the
representation-only verification after that clarification. Both requests were
valid on their first attempts: eight singleton propositions, complete partition,
all original quotations validated, aggregation ready, two model calls, and no
aborted batch. Input sizes were 3,205 and 5,455 bytes; full JSON requests were
6,089 and 11,475 bytes. Qwen returned eight NOT_MATCHED decisions.

This resolves the prior invalid-merge/incomplete-partition failure for this
specific batch. It does not prove all coverage judgments correct. In particular,
the devil-resemblance viewpoint changed from MATCHED to NOT_MATCHED: Qwen first
treated "mockery of appearance" as covering that viewpoint, then required the
specific resemblance to be explicit. This sensitivity in coverage granularity
remains a model/operational-definition limitation requiring human review. No
Python rule converted an UNKNOWN into an omission. The original S05 factual
contradictions were not rechecked by these focused probes, so their general
semantic accuracy is not claimed as repaired. The factual explanation also
imprecisely calls the selected evidence "16 out of 90 cited items"; the recorded
selection is 16 of 250 evidence entries, with 88 original tweets retrieved.

Preliminary probes are retained separately. An initial HTTP 400 exposed an
empty-enum schema error during development; the corrected schema has a dedicated
regression test. A subsequent 90-second timeout was insufficient on this local
runtime. The successful probes use the project's existing 300-second timeout;
the server logged roughly two minutes for the coverage response. Slow runtime
and HTTP/transport failures can still occur and return conservative UNKNOWN.

These are focused live probes, not mocked model performance, not a full-topic
fairness result and not a repeat of the 90-tweet LangGraph experiment. The probe
checks confirmed that the original pilot, saved graph and in-memory graph were
unchanged. The original ontology's SHA-256 remains
`186df030f75160bebc14ebff2d4d6151873928a99c1a2b53f5f5d881736743c4`.
