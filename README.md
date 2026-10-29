# Ontology-assisted summarization agent: sections 1, 2, and 3

Section 1 loads the tweets. Section 2 exposes ontology query tools. Section 3
now defaults to local Ollama with `qwen3.5:4b`, without an API key or paid calls.
OpenAI remains an explicit optional backend. The checking/revision loop is not
implemented yet. Use Python 3.10 or newer.

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
| `ontology/group2_Ontology.owl` | Your original ontology, unchanged |
| `ontology/evidence_schema.ttl` | Tweet/topic classes and source-evidence properties, including dialect labels |
| `populate_ontology.py` | Validate saved extractions and populate a fresh RDF graph per run |
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

The graph contains a fresh copy of the original OWL ontology, the local
`ontology/evidence_schema.ttl` extension, and the selected source evidence:

- Every tweet retains its original ID, exact text, source file/line, extraction
  status, topic link and dataset dialect label as a literal `dialectLabel` property.
- `containsClaim` connects tweets to per-tweet `Claim` records. Claims retain
  their original IDs, interpretation, actual source quotation, target and stance.
  Claim text uses the ontology's existing `hasText` property.
- `unclear` and `no_claim` tweets remain in the graph without claim links.
- Unknown/missing/duplicate tweet results, dialect mismatches, duplicate claim
  IDs, inconsistent statuses and quotations absent from the source are rejected.
- Tweet/claim IRIs are scoped to the run. Each function call loads a fresh graph,
  with no shared mutable evidence graph or edits to the base OWL file.

Following the team's architecture decision, dialect is tweet metadata:
`<tweet-iri> :dialectLabel "AA"`. There is no `DialectGroup` class, group individual
or `belongsToDialectGroup` object property. Group-specific queries filter this
literal directly, keeping the labels available for later coverage calculations.
The report's architecture description should be updated to match this decision.

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
Claim aggregation, factual/representation checking, revision and the LangGraph
controller remain to be implemented.
