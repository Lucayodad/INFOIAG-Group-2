# Ontology-assisted summarization agent: sections 1, 2, and 3

Section 1 loads the tweets. Section 2 exposes ontology query tools. Section 3
now defaults to local Ollama with `qwen3.5:4b`, without an API key or paid calls.
OpenAI remains an explicit optional backend. Factual checking now has a bounded
agent and summary wrapper; representation checking provides group retrieval,
auditable matching and weighted coverage. A LangGraph controller now connects
summary extraction, both checkers, local embedding retrieval, targeted revision
and bounded rechecking. It retains the best checked version and attempts at most
three revisions. Use Python 3.10 or newer.

## Local setup for the group

Get the complete agent from `feature/saif-agent-implementation`. For a fresh
Windows checkout, run these commands in PowerShell:

```powershell
git clone --branch feature/saif-agent-implementation https://github.com/Lucayodad/INFOIAG-Group-2.git
cd INFOIAG-Group-2
py -3 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3.5:4b
ollama pull all-minilm
& .\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Install and start [Ollama](https://ollama.com/download) before the model commands.
For an existing clone, switch to the feature branch and pull its latest changes,
then install `requirements.txt` in the project's environment. On macOS/Linux,
create the environment with `python3 -m venv .venv` and use `.venv/bin/python`
instead of `.venv\Scripts\python.exe`. No paid API key is needed for Ollama.

To exercise the whole agent with synthetic evidence and mocked model responses:

```powershell
& .\.venv\Scripts\python.exe .\demo_revision_loop.py
```

For a real Obama run, generate source claims, populate RDF, and generate Prompt 1:

```powershell
& .\.venv\Scripts\python.exe .\run_source_extraction.py --topic Obama --limit 90
& .\.venv\Scripts\python.exe .\populate_ontology.py --topic Obama
& .\.venv\Scripts\python.exe .\main.py --section draft --topic Obama --prompt-id 1
```

Only populate RDF after extraction reports all 90 tweets complete with no
unresolved failures. Extraction resumes from validated checkpoints when repeated.
Use the graph and draft paths printed by those commands in the complete-agent
command below. Generated checkpoints, RDF graphs, drafts and result JSON files
are local artifacts excluded from Git; pulling the code does not download them.
Real runs can take substantial time on local hardware. A passing test suite
checks the implementation; semantic accuracy still requires reviewing real runs.

## Section 3: local generation

Install Ollama from https://ollama.com/download and open its Mac application.
Download the local model in your terminal (a several-GB download):

```bash
ollama pull qwen3.5:4b
```

From this extracted `bias_summary_agent` directory, preview a request:

```bash
python3 main.py --section draft --topic Obama --prompt-id 1 --dry-run
```

Generate one summary, then all six topic/prompt combinations:

```bash
python3 main.py --section draft --topic Obama --prompt-id 1
python3 main.py --section draft --topic all --prompt-id all
```

If the local server is unavailable, open Ollama or run `ollama serve` in a
separate terminal. Local generation uses Python's standard library and needs
no OpenAI package, key, or API billing. The ontology stage still needs the
packages in `requirements.txt`.

Results are JSON and plain text in `results/drafts/<unique-run-id>/`. Dry runs
save request previews only. These are initial unchecked drafts, not verified
bias-free summaries. See [docs/SECTION_3.md](docs/SECTION_3.md) for details,
optional OpenAI usage, model settings, and troubleshooting.

## Run the complete agent on saved data

`summary_agent.py` is the real Qwen/Ollama CLI. Give it a saved five-sentence
draft JSON and a populated RDF Turtle graph:

```powershell
& .\.venv\Scripts\python.exe .\summary_agent.py "path\to\Obama_prompt1.json" --graph "path\to\Obama.ttl" --topic Obama --disparity-threshold 0.25 --max-revisions 3 --output ".\results\evidence\Obama_prompt1_pilot.json"
```

Install the requirements in that interpreter first, and have Ollama with
`qwen3.5:4b` and `all-minilm` available. This CLI reuses the complete agent; it
does not regenerate source claims or the initial summary. Its exploratory
threshold defaults to 0.25 and remains configurable. Complete JSON is saved to
`--output`, or a unique file under `results/evidence/agent_runs/` if omitted.
The console prints concise factual, coverage, revision and final-status results.
Both demo scripts use controlled mocked data; they are not this real-data run.
Use a complete 90-tweet graph for full-topic representation results. A five-tweet
sample retains incomplete-evidence/UNKNOWN behavior. See
[docs/SUMMARY_AGENT.md](docs/SUMMARY_AGENT.md) for details.

## Section 2: setup and run

From the extracted `bias_summary_agent` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 main.py --section ontology
```

The supplied ontology and report are already included. You do not need to copy
them again. To run one query or compare with asserted triples only:

```bash
python3 main.py --section ontology --query Q1
python3 main.py --section ontology --query Q1 --asserted-only
python3 -m unittest discover -s tests -v
```

See [docs/SECTION_2.md](docs/SECTION_2.md) for the tool interfaces, expected
results, reasoning scope, and remaining ontology limitations.

## Run

Extract the ZIP, open a terminal in `bias_summary_agent`, and run:

```bash
python3 main.py
```

To load only one topic:

```bash
python3 main.py --topic Obama
python3 main.py --topic Superbowl
python3 main.py --topic McDonalds
```

Expected output for all topics:

```text
Obama: 90 tweets | AA=30 | Hisp=30 | White=30 | OK
Superbowl: 90 tweets | AA=30 | Hisp=30 | White=30 | OK
McDonalds: 90 tweets | AA=30 | Hisp=30 | White=30 | OK

Section 1 complete: 270 tweets across 3 topic(s).
```

## Files and responsibilities

| File or directory | Purpose |
|---|---|
| `data_loader.py` | Tweet record type, JSONL loader, schema checks, and dataset-count validation |
| `main.py` | Command-line entry point for this section |
| `data/divsumm/ready_to_use/` | UTF-8 JSONL inputs and separately stored reference summaries |
| `data/divsumm/original/` | Original selected source files, unchanged |
| `data/divsumm/utf8/` | Convenient UTF-8 text copies |
| `data/divsumm/manifest.json` | Dataset provenance, encoding notes, and original-file checksums |
| `data/divsumm/README.md` | Dataset documentation and paper citation |
| `ontology/group2_Ontology.owl` | Luca's updated ontology, preserved by the pipeline; includes illustrative evidence |
| `ontology/evidence_schema.ttl` | Tweet/topic classes and source-evidence properties, including dialect labels |
| `populate_ontology.py` | Validate saved extractions and populate a fresh RDF graph per run |
| `fact_check.py` / `fact_agent.py` | Shared factual verification and the bounded agent, including `fact_check_summary` |
| `representation_tools.py` | SPARQL source evidence retrieval by literal dialect label |
| `representation_check.py` | Bounded semantic aggregation/matching and deterministic coverage with uncertainty bounds |
| `docs/CHECK_COMPONENTS.md` | Checking interfaces, severity rules, thresholds, completeness and research limitations |
| `summary_agent.py` | Typed LangGraph revision/rechecking, rollback, history and CLI |
| `revision_retrieval.py` | Group-specific original evidence with local embeddings and cosine ranking |
| `summary_revision.py` | Targeted Qwen proposals, citation guards and exact preservation of other sentences |
| `demo_revision_loop.py` | Complete mocked revision loop and optional separate live proposal probe |
| `tests/test_revision_loop.py` | Revision, stopping, rollback and controlled 90-tweet integration tests |
| `demo_summary_agent.py` | Complete synthetic offline execution with mocked Qwen replies |
| `tests/test_summary_agent.py` | Real-graph orchestration and CLI integration tests |
| `docs/SUMMARY_AGENT.md` | Controller state, node connections, routing, usage and demonstration |
| `queries/source_evidence.rq` | Query evidence by topic/group, including tweets without claims |
| `tests/test_populate_ontology.py` | Provenance, validation, run isolation, SPARQL and CLI checks |
| `ontology_tools.py` | Query tools and limited inference |
| `queries/Q1.rq` through `Q6.rq` | Local SPARQL templates |
| `tests/test_ontology_tools.py` | Eight tests using the ontology and isolated fixtures |
| `docs/SECTION_2.md` | Explanation of section 2 and its limitations |
| `docs/section2_results.json` | Results captured from this version |
| `docs/report_AI.pdf` | Supplied Intermediate Report 1 |
| `prompts.py` | The report's two summary prompts and fixed system instructions |
| `summarizer.py` | Request construction, API call, output validation, and saving |
| `tests/test_summarizer.py` | Seven offline checks including the optional OpenAI SDK |
| `ollama_backend.py` | Local HTTP transport and response normalization |
| `tests/test_ollama_backend.py` | Six local transport and validation checks |
| `docs/SECTION_3.md` | Model choice, setup, and generation-stage explanation |

## How the code works

1. `load_tweets(topic)` opens that topic's JSONL file using UTF-8.
2. Each record is checked for required fields, matching topic, recognized label,
   a valid source line, and unique identifiers/source references.
3. Valid rows become `Tweet` objects in their original order. Tweet text is not
   cleaned, normalized, unescaped, or stripped.
4. `validate_dataset(tweets)` verifies the expected 90 tweets and 30 rows per
   dialect label. It reports damaged/incomplete input instead of changing it.
5. `main.py` prints the counts. It reads no reference summaries.

You can reuse the loader in the next sections:

```python
from data_loader import load_tweets

tweets = load_tweets("Obama")
first_tweet = tweets[0]
print(first_tweet.id)
print(first_tweet.dialect_label)
print(first_tweet.text)
```

Labels `AA`, `Hisp`, and `White` are the source dataset's dialect categories,
not independently verified author demographics. The generated IDs are local
record identifiers, not Twitter IDs. Reference summaries stay separate for
later evaluation; they are not evidence supplied to the agent for rewriting.

Some original source files use a legacy encoding. This project includes the
previously prepared UTF-8 copies and the original bytes. The inferred Mac Roman
conversion and its limitations are documented in `data/divsumm/README.md`.

## Source evidence in RDF

Run these commands from the repository directory with the project environment
active. First extract the complete topic, then populate its evidence graph:

```bash
python run_source_extraction.py --topic Obama --limit 90
python populate_ontology.py --topic Obama
```

Source extraction resumes `results/source_claims/Obama.json` and skips every
successfully saved tweet. Quote validation tolerates valid HTML entities, balanced
outer quotation marks and whitespace while retaining the exact original source
substring. It does not use fuzzy/paraphrase matching or accept changed numbers
and negation.

Invalid model output is retried twice with explicit exact-copy instructions
(three total attempts per tweet). Persistent failures are recorded separately
in `results/source_claims/Obama.failures.json`, never as synthetic `no_claim`
records. Processing continues, then reports unresolved IDs and exits with code 1
if any remain. Rerun the same extraction command to retry unresolved tweets;
already validated records are preserved and skipped. Failure history retains
per-run/cumulative failed-attempt counts, and recovered entries are marked
`resolved`. Checkpoints and failure logs use temporary files and atomic replace.
Population still requires all 90 validated extraction records for a complete run.

Population reads `results/source_claims/Obama.json`; it does not call the LLM.
Use `--claims path/to/extractions.json` to select a different extraction file.
The default command requires all 90 tweets with exactly one extraction result
per tweet. To inspect an explicitly incomplete extraction checkpoint:

```bash
python populate_ontology.py --topic Obama --sample
```

Each invocation writes `results/ontology/<unique-run-id>/Obama.ttl` and a
`manifest.json` with the source paths, run ID, tweet/claim/status counts and
whether this is complete topic evidence or a sample. Generated results are
ignored by Git. Sample graphs are for inspection, not complete coverage analysis.

The graph contains a filtered copy of the supplied OWL ontology, the local
`ontology/evidence_schema.ttl` extension, and the selected source evidence:

- Every tweet retains its original ID, exact text, source file/line, extraction
  status, topic link and dataset dialect label as a literal `dialectLabel` property.
  New graphs use the base ontology's `tweetHasText` property. Readers also accept
  `tweetText` from older graphs without requiring an OWL reasoner.
- `containsClaim` connects tweets to per-tweet `Claim` records. Claims retain
  their original IDs, interpretation, actual source quotation, target and stance.
  Claim text uses the ontology's existing `hasText` property.
  `claimAboutTopic` links each extracted claim to its tweet's topic.
- `unclear` and `no_claim` tweets remain in the graph without claim links.
- Unknown/missing/duplicate tweet results, dialect mismatches, duplicate claim
  IDs, inconsistent statuses and quotations absent from the source are rejected.
- Tweet/claim IRIs are scoped to the run. Each function call loads a fresh graph,
  with no shared mutable evidence graph or edits to the base OWL file.
- Base example tweets, claims (including subclass instances), assessments and
  summaries are excluded before run evidence is added. A 90-tweet run therefore
  contains exactly 90 typed tweets, not 95. Domain assertions and schema remain.
  `SuperBowl` and `DivSummDataset` remain topic/source reference individuals.
- The shared topic mapping uses `Topic_Obama`, `Topic_McDonalds`, and the base
  ontology's `SuperBowl` for dataset key `Superbowl`. Readers also accept the old
  `Topic_Superbowl` identifier. Topics have `hasTopicName` labels.
- Conflicting text under the two spellings is reported as an integrity issue
  by group retrieval; factual retrieval returns no usable single text and keeps
  both values, so factual checking remains UNKNOWN.

Following the team's architecture decision, dialect is tweet metadata:
`<tweet-iri> :dialectLabel "AA"`. There is no `DialectGroup` class, group individual
or `belongsToDialectGroup` object property. Group-specific queries filter this
literal directly, keeping the labels available for later coverage calculations.
The report's architecture description should be updated to match this decision.
See [ontology/README.md](ontology/README.md) for the base filtering policy.
Manifests record it as `base_evidence_policy`. Existing saved graphs are not
rewritten; regenerate any graph populated from the updated base before this fix
to remove its example evidence. Old text/topic spellings remain readable.

The population function can also be called by a future controller:

```python
import json
from pathlib import Path
from rdflib import Literal
from data_loader import load_tweets
from ontology_tools import NS
from populate_ontology import populate_ontology

tweets = load_tweets("Obama")
records = json.loads(Path("results/source_claims/Obama.json").read_text(encoding="utf-8"))
graph = populate_ontology(tweets, records)
query = Path("queries/source_evidence.rq").read_text(encoding="utf-8")
rows = graph.query(query, initBindings={
    "topic": NS.Topic_Obama,
    "dialect_label": Literal("AA"),
})
for row in rows:
    print(row.tweet_id, row.status, row.claim_text, row.quote)
```

The function requires one record per supplied tweet; the CLI additionally
enforces full-topic completeness unless `--sample` is used. The query uses
`OPTIONAL` so tweets without claims are included. Saved Turtle graphs can be
reopened with `rdflib.Graph().parse(path, format="turtle")`; the existing
`main.py --section ontology` command still queries the original RDF/XML input.

`evidence_store.py` remains an optional JSON entity-link report. RDF population
uses the original extraction records and does not depend on that report.
Extracted statements describe what authors express, not verified world facts;
population creates no `ClaimAssessment` or `isValid` labels. Quotation matching
checks provenance, not whether the interpretation is semantically correct.
Factual and representation checking are described in
[docs/CHECK_COMPONENTS.md](docs/CHECK_COMPONENTS.md). Representation matching uses
deterministic batches and a shared canonical registry to reconcile equivalent
claims across batches. The input limit applies per request; the default total
budget is 64 model calls and `batch_size` defaults to 8. Unresolved matching or
budget exhaustion remains UNKNOWN. Equivalence and summary coverage use separate
tasks; each invalid batch gets one repair and independent batches continue.
Factual prompts use at most 16 whole evidence entries / 8,000 catalogue bytes,
validated short IDs mapped back to RDF provenance, and one format repair.
Function-by-function changes are documented in
[docs/CHECKER_RELIABILITY.md](docs/CHECKER_RELIABILITY.md).
`python probe_checker_reliability.py --output results/evidence/new_probes.json`
runs only the two focused live Obama probes, using the saved pilot and graph
paths documented by `--help`; it refuses to overwrite existing results.
The LangGraph controller is available through
`run_summary_agent` and `python summary_agent.py`; see
[docs/SUMMARY_AGENT.md](docs/SUMMARY_AGENT.md). It finalizes with uncertainty when
needed. Confirmed issues trigger bounded evidence-guided revision and both checks
are repeated before accepting a new version. See the workflow documentation for
the same-metric improvement rule and all stopping conditions.

Revision retrieval uses local `all-minilm` embeddings through Ollama, with
in-memory cosine ranking and no vector database or additional Python dependency.
Install the model once with `ollama pull all-minilm`. Missing/invalid embeddings
or insufficient original evidence stop revision safely; no lexical fallback is
substituted. Run `python demo_revision_loop.py` for the controlled example, or
add `--live` for a separate local Qwen proposal probe. These are development
checks, not the 36-run experimental evaluation.
