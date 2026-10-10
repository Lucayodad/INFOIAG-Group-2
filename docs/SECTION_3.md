# Section 3: first-draft generation with local Ollama

## Scope and model

This update changes the default backend to local Ollama and `qwen3.5:4b`.
There is no API key or per-request provider charge for this local model;
downloading it and running it use your connection, storage, memory, and power.
Model file size is not total runtime memory use. Start with 4B and check speed
and memory pressure before trying a larger model. No claim is made that its
summary quality matches GPT-4.1 or that any model is inherently bias-free.

The agent still supplies all 90 tweets and creates an initial summary only.
Claim extraction, ontology checking, retrieval, revision, and rechecking are
later sections. Reference summaries remain outside all generation prompts.

## Setup and first run

Install https://ollama.com/download and open Ollama on your Mac. In Terminal:

```bash
ollama pull qwen3.5:4b
```

Enter the extracted `bias_summary_agent` folder, then run:

```bash
python3 main.py --section draft --topic Obama --prompt-id 1 --dry-run
python3 main.py --section draft --topic Obama --prompt-id 1
```

The dry run saves the payload without contacting any model. The second command
calls the local server at `http://127.0.0.1:11434/api/chat` and saves a summary.
If unavailable, open Ollama or run `ollama serve` in another terminal. If its
port is already in use, the server may already be running; do not start a
second one. Check model installation with `ollama list`.

Local draft generation needs only Python's standard library. For ontology
queries and all tests, set up the project environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Generate another topic or all six combinations:

```bash
python3 main.py --section draft --topic McDonalds --prompt-id 2
python3 main.py --section draft --topic all --prompt-id all
```

The latter performs six local requests sequentially. Each run has a unique
folder; successful earlier results remain if a later request fails.

Select a downloaded local model with `--model` or `OLLAMA_MODEL`. For example,
if hardware permits, first download `qwen3.5:9b`, then pass
`--model qwen3.5:9b`. Use a model supporting schemas and `think: false`.
This backend is intended for downloaded local models, not Ollama Cloud models.

## Execution and settings

1. The existing loader validates the selected 90-tweet dataset.
2. The selected report prompt and the fixed system instructions are prepared.
3. All original tweet texts and their IDs are sent as JSON source data.
4. Ollama receives the JSON schema through `format`; the schema is also in the
   user message as recommended by its documentation.
5. The backend requires a completed response with `done_reason: stop` and
   normalizes it to the same response representation as the optional OpenAI backend.
6. The shared validator requires five entries, nonempty text, and valid unique
   source IDs. It rejects malformed JSON, repeated entries, and invalid IDs.
7. JSON metadata and the plain-text summary are saved.

Ollama settings are `stream: false`, `think: false`, `temperature: 0`, `seed: 42`,
`num_ctx: 16384`, and `num_predict: 2000`. They are included in the request hash;
generation settings are also recorded in the result. The larger context setting
is intended to hold all 90 tweets and the schema without using a small default
context. Future larger inputs need explicit token-budget handling or chunking.
Each request times out after 300 seconds and has no automatic retry. The local
HTTP client bypasses proxy environment settings and rejects redirects.

Temperature and seed reduce variability; they do not guarantee identical
outputs across hardware, versions, or model changes. Ollama tags can change.
For the final experiment, retain your Ollama version (`ollama --version`),
model identifier (`ollama list`), and downloaded model, and use the same model,
prompt, and generation settings for baseline and ontology-assisted variants.

## Saved outputs and limits

Outputs are `results/drafts/<run-id>/<topic>_prompt<id>.json` and `.txt`.
Metadata includes provider, requested/returned model, source-ID annotations,
input/request hashes, settings, token counts when available, local server
nanosecond durations, and elapsed request time in seconds. Elapsed time includes
model loading when it occurs, which matters for later timing comparisons.
Ollama supplies no Responses API response ID, so that field is null.

Every output is marked `draft_unchecked`. Five JSON entries do not prove five
linguistic sentences. IDs do not prove that the tweets support the claims.
Semantic grounding, sentence boundaries, bias/coverage, and ontology checks
remain explicitly pending. Dialect labels are kept locally for the later
checker; the generator receives text and IDs only.

## Optional paid OpenAI backend

OpenAI is available only when explicitly selected:

```bash
export OPENAI_API_KEY="$(python3 -c 'import getpass; print(getpass.getpass("OpenAI API key: "))')"
python3 main.py --section draft --provider openai --topic Obama --prompt-id 1
```

Its default is `gpt-4.1-mini-2025-04-14`, configurable through `--model` or
`OPENAI_MODEL`. It uses the Responses API with strict structured output,
2,000 maximum output tokens, a 60-second timeout, and no automatic retries.
Its temperature uses the model default. OpenAI generation requires API billing
separate from ChatGPT Plus and sends the selected tweets to OpenAI. The local
backend never falls back to it. Do not compare different providers/models if
your purpose is to isolate the effect of the ontology workflow.

## Verification and troubleshooting

Twenty-one automated tests passed: eight ontology checks, seven existing
OpenAI/output/dry-run checks, and six added Ollama checks. The latter use a real
local HTTP fixture server with synthetic output, exercising serialization,
normalization, token metadata, missing-model errors, connection errors,
truncation rejection, and invalid source IDs. They are not model evaluation.

No real Ollama summary was generated during implementation because Ollama was
not installed/available in this session. Live model schema acceptance, speed,
and quality still require your first run. No synthetic summaries are included
as project results.

Run local tests without installing the OpenAI SDK:

```bash
python3 -m unittest discover -s tests -p test_ollama_backend.py -v
```

Run all tests after installing requirements:

```bash
python3 -m unittest discover -s tests -v
```

- Connection/timeout: open Ollama or use `ollama serve`; check memory pressure
  if it is already running. Start with one topic/prompt.
- HTTP 404: download the exact model tag with `ollama pull`.
- Request-setting errors: update Ollama and check schema/thinking support.
- Truncated output: no draft is saved. Inspect the model/output budget.
- Invalid JSON or source IDs: the validator rejects the response; share the
  error for review rather than accepting an invalid summary.

Official references:
- https://docs.ollama.com/api/chat
- https://docs.ollama.com/capabilities/structured-outputs
- https://ollama.com/library/qwen3.5

## Pause point

Stop after this updated section for user review. Next: extract claims from the
source tweets and draft, and link them to ontology entities. Retrieval and the
check/rewrite/recheck loop follow later.
