"""

Metrics (choose with --metrics, default: all):
  cost      execution time, tokens and LLM calls, read from the result JSON
  rouge     ROUGE-1/2/L F1 vs. both human abstractive references (mean)
  gap       group similarity gap (sentence-transformers, all-MiniLM-L6-v2)
  ec        Equal Coverage (Li et al. 2025) with roberta-large-mnli
  deepeval  DeepEval summarization: alignment + coverage, local Ollama judge
  bart      BARTScore (Yuan et al. 2021) with facebook/bart-large-cnn:
            bart_src = faithfulness to the tweets, bart_ref = F-score vs references

Typical use, from the repository root:
  python evaluate.py --system baseline --inputs results/drafts/<run-id>
  python evaluate.py --system agent    --inputs path/to/agent/outputs
  python evaluate.py --references          # the two human summaries per topic
  python evaluate.py --report              # mean/std tables from metrics.csv

Every evaluated summary becomes one row in results/eval/metrics.csv.
"""

import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "divsumm" / "ready_to_use"
OUT_DEFAULT = ROOT / "results" / "eval" / "metrics.csv"
TOPICS = ("Obama", "Superbowl", "McDonalds")
GROUPS = ("AA", "Hisp", "White")
ALL_METRICS = ("cost", "rouge", "gap", "ec", "deepeval", "bart")


# ---------------------------------------------------------------- data

def load_tweets(topic):
    path = DATA / f"{topic}.jsonl"
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [{"id": r["id"], "group": r["dialect_label"], "text": r["text"]} for r in rows]


def load_references():
    with (DATA / "references.json").open(encoding="utf-8") as f:
        refs = json.load(f)
    return {t: [r["text"].strip() for r in refs[t]["abstractive"]] for t in refs}


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def guess_topic_prompt(path, data):
    topic = data.get("topic")
    prompt = data.get("prompt_id")
    m = re.search(r"(Obama|Superbowl|McDonalds)(?:_prompt(\d))?", path.name)
    if not topic and m:
        topic = m.group(1)
    if not prompt and m and m.group(2):
        prompt = m.group(2)
    return topic, (str(prompt) if prompt else "")


def records_from_file(path, system):
    """Turn one result JSON into one or two summary records.

    Accepts the draft format of summarizer.py ("summary", "sentences") and
    tries common agent keys ("final_summary", "draft_summary"). If the
    agent output uses other keys, add them here.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    topic, prompt = guess_topic_prompt(path, data)
    if topic not in TOPICS:
        print(f"  skipped {path.name}: topic unknown")
        return []
    final = data.get("final_summary") or data.get("summary")
    if not final:
        print(f"  skipped {path.name}: no 'summary' or 'final_summary' key")
        return []
    if isinstance(data.get("sentences"), list) and data.get("summary") == final:
        sentences = [s["text"] if isinstance(s, dict) else str(s) for s in data["sentences"]]
    else:
        sentences = split_sentences(final)
    usage = data.get("usage") or {}
    agent = data.get("agent_result") or {}
    base = {
        "topic": topic, "prompt_id": prompt, "seed": data.get("seed", ""), "file": str(path),
        "execution_seconds": data.get("execution_seconds", ""),
        "input_tokens": usage.get("input_tokens", ""),
        "output_tokens": usage.get("output_tokens", ""),
        "llm_calls": data.get("llm_calls", 1 if system == "baseline" else ""),
        "revision_count": agent.get("revision_count", ""),
        "agent_status": agent.get("status", ""),
    }
    out = [dict(base, system=system, summary=final, sentences=sentences)]
    # The agent's first draft: "initial_summary" in the experiment runner's output.
    initial = data.get("initial_summary") if isinstance(data.get("initial_summary"), dict) else {}
    draft = data.get("draft_summary") or initial.get("summary")
    if draft:
        if isinstance(initial.get("sentences"), list):
            draft_sents = [s["text"] if isinstance(s, dict) else str(s) for s in initial["sentences"]]
        else:
            draft_sents = split_sentences(draft)
        d_usage = initial.get("usage") or {}
        out.append(dict(base, system=f"{system}_draft", summary=draft, sentences=draft_sents,
                        execution_seconds=initial.get("execution_seconds", ""),
                        input_tokens=d_usage.get("input_tokens", ""),
                        output_tokens=d_usage.get("output_tokens", ""),
                        llm_calls=1, revision_count="", agent_status=""))
    return out


def collect_records(inputs, system):
    records = []
    for item in inputs:
        p = Path(item)
        files = sorted(p.rglob("*.json")) if p.is_dir() else [p]
        for f in files:
            if f.name.endswith(".request.json") or f.name == "manifest.json":
                continue
            records.extend(records_from_file(f, system))
    return records


def reference_records():
    out = []
    for topic, texts in load_references().items():
        for i, text in enumerate(texts, 1):
            out.append({"system": "human_reference", "topic": topic, "prompt_id": "",
                        "file": f"references.json:{topic}:Annotator{i}",
                        "summary": text, "sentences": split_sentences(text),
                        "execution_seconds": "", "input_tokens": "",
                        "output_tokens": "", "llm_calls": "", "_ref_index": i - 1})
    return out


# ---------------------------------------------------------------- metrics

def metric_rouge(rec, refs):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    targets = refs[rec["topic"]]
    if "_ref_index" in rec:  # a human reference is compared with the other annotator only
        targets = [t for i, t in enumerate(targets) if i != rec["_ref_index"]]
    scores = [scorer.score(t, rec["summary"]) for t in targets]
    return {k: round(statistics.mean(s[k].fmeasure for s in scores), 4)
            for k in ("rouge1", "rouge2", "rougeL")}


def coverage_by_group(matrix, groups):
    """matrix[i][j]: score of tweet i vs summary sentence j. Returns per-group
    means over all (tweet, sentence) pairs, and the overall mean."""
    def mean_rows(rows):
        vals = [v for r in rows for v in r]
        return sum(vals) / len(vals)
    per_group = {g: mean_rows([row for row, gg in zip(matrix, groups) if gg == g]) for g in GROUPS}
    return per_group, mean_rows(matrix)


def equal_coverage(matrix, groups):
    per_group, overall = coverage_by_group(matrix, groups)
    ec = sum(abs(per_group[g] - overall) for g in GROUPS) / len(GROUPS)
    return ec, per_group, overall


def similarity_gap(matrix, groups):
    sims = {}
    for g in GROUPS:
        rows = [max(row) for row, gg in zip(matrix, groups) if gg == g]
        sims[g] = sum(rows) / len(rows)
    return max(sims.values()) - min(sims.values()), sims


_EMB = None
def metric_gap(rec, tweets):
    global _EMB
    from sentence_transformers import SentenceTransformer, util
    if _EMB is None:
        _EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    t = _EMB.encode([x["text"] for x in tweets], convert_to_tensor=True)
    s = _EMB.encode(rec["sentences"], convert_to_tensor=True)
    matrix = util.cos_sim(t, s).tolist()
    gap, sims = similarity_gap(matrix, [x["group"] for x in tweets])
    out = {"sim_gap": round(gap, 4)}
    out.update({f"sim_{g}": round(v, 4) for g, v in sims.items()})
    return out


_NLI = None
def metric_ec(rec, tweets, batch_size=32):
    """p(d_i, s_j) = probability that tweet d_i (premise) entails summary
    sentence s_j (hypothesis), as in Li et al. (2025), eq. 2-5."""
    global _NLI
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if _NLI is None:
        name = "FacebookAI/roberta-large-mnli"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        ent = {v.upper(): k for k, v in model.config.id2label.items()}["ENTAILMENT"]
        _NLI = (tok, model, ent)
    tok, model, ent = _NLI
    pairs = [(tw["text"], s) for tw in tweets for s in rec["sentences"]]
    probs = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            enc = tok([p for p, _ in batch], [h for _, h in batch], return_tensors="pt",
                      padding=True, truncation=True, max_length=256)
            probs.extend(torch.softmax(model(**enc).logits, dim=-1)[:, ent].tolist())
    n = len(rec["sentences"])
    matrix = [probs[i * n:(i + 1) * n] for i in range(len(tweets))]
    ec, per_group, overall = equal_coverage(matrix, [x["group"] for x in tweets])
    out = {"ec": round(ec, 4), "cov_all": round(overall, 4)}
    out.update({f"cov_{g}": round(v, 4) for g, v in per_group.items()})
    return out


def metric_deepeval(rec, tweets, judge_model):
    from deepeval.metrics import SummarizationMetric
    from deepeval.models import OllamaModel
    from deepeval.test_case import LLMTestCase
    judge = OllamaModel(model=judge_model, base_url="http://localhost:11434", temperature=0)
    metric = SummarizationMetric(model=judge, async_mode=False)
    source = "\n".join(x["text"] for x in tweets)
    metric.measure(LLMTestCase(input=source, actual_output=rec["summary"]))
    breakdown = getattr(metric, "score_breakdown", None) or {}
    return {"deepeval_score": round(metric.score, 4),
            "deepeval_alignment": round(breakdown.get("Alignment", getattr(metric, "alignment_score", float("nan"))), 4),
            "deepeval_coverage": round(breakdown.get("Coverage", getattr(metric, "coverage_score", float("nan"))), 4)}


_BART = None
def _bart():
    global _BART
    if _BART is None:
        from transformers import BartForConditionalGeneration, BartTokenizer
        name = "facebook/bart-large-cnn"
        _BART = (BartTokenizer.from_pretrained(name),
                 BartForConditionalGeneration.from_pretrained(name).eval())
    return _BART


def bart_logprob(src, tgt):
    """Average log-likelihood per token of tgt given src (the BARTScore of src -> tgt)."""
    import torch
    tok, model = _bart()
    enc = tok(src, max_length=1024, truncation=True, return_tensors="pt")
    dec = tok(tgt, max_length=1024, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    labels=dec["input_ids"])
    return -out.loss.item()


def tweet_chunks(tweets, max_tokens=900):
    """BART reads at most 1024 tokens, but 90 tweets are ~4,000, so the tweets
    are split into consecutive chunks that each fit."""
    tok, _ = _bart()
    chunks, current, size = [], [], 0
    for t in tweets:
        n = len(tok(t["text"], add_special_tokens=False)["input_ids"]) + 1
        if current and size + n > max_tokens:
            chunks.append("\n".join(current)); current, size = [], 0
        current.append(t["text"]); size += n
    if current:
        chunks.append("\n".join(current))
    return chunks


_CHUNKS = {}
def metric_bart(rec, tweets, refs):
    # Faithfulness: each summary sentence scored against the tweet chunk that
    # supports it best (max over chunks), averaged over the sentences.
    chunks = _CHUNKS.setdefault(rec["topic"], tweet_chunks(tweets))
    src = statistics.mean(max(bart_logprob(c, s) for c in chunks) for s in rec["sentences"])
    # Reference-based F-score: mean of P(summary|ref) and P(ref|summary), over both references.
    targets = refs[rec["topic"]]
    if "_ref_index" in rec:
        targets = [t for i, t in enumerate(targets) if i != rec["_ref_index"]]
    ref = statistics.mean((bart_logprob(t, rec["summary"]) + bart_logprob(rec["summary"], t)) / 2
                          for t in targets)
    return {"bart_src": round(src, 4), "bart_ref": round(ref, 4)}


# ---------------------------------------------------------------- output

FIELDS = ["system", "topic", "prompt_id", "seed", "file",
          "ec", "cov_AA", "cov_Hisp", "cov_White", "cov_all",
          "sim_gap", "sim_AA", "sim_Hisp", "sim_White",
          "deepeval_score", "deepeval_alignment", "deepeval_coverage",
          "bart_src", "bart_ref",
          "rouge1", "rouge2", "rougeL",
          "execution_seconds", "input_tokens", "output_tokens", "llm_calls",
          "revision_count", "agent_status", "eval_seconds"]


def append_rows(rows, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def report(out):
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = ["ec", "sim_gap", "deepeval_alignment", "deepeval_coverage", "bart_src", "bart_ref",
            "rouge1", "rouge2", "rougeL", "execution_seconds"]

    def fmt(vals):
        nums = [float(v) for v in vals if v not in ("", None)]
        if not nums:
            return "-"
        if len(nums) == 1:
            return f"{nums[0]:.3f}"
        return f"{statistics.mean(nums):.3f} ± {statistics.stdev(nums):.3f}"

    def table(key_fn, title):
        groups = {}
        for r in rows:
            groups.setdefault(key_fn(r), []).append(r)
        lines = [f"\n## {title}\n", "| group | n | " + " | ".join(cols) + " |",
                 "|" + "---|" * (len(cols) + 2)]
        for k in sorted(groups):
            g = groups[k]
            lines.append(f"| {k} | {len(g)} | " + " | ".join(fmt([r[c] for r in g]) for c in cols) + " |")
        return "\n".join(lines)

    text = (table(lambda r: r["system"], "Per system (mean ± std)")
            + table(lambda r: f"{r['system']} / {r['topic']} / prompt {r['prompt_id'] or '-'}",
                    "Per system, topic and prompt"))
    print(text)
    path = out.with_name("report.md")
    path.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved: {path}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", help="Label for these summaries, e.g. baseline or agent.")
    ap.add_argument("--inputs", nargs="*", default=[], help="Result JSON files or folders.")
    ap.add_argument("--references", action="store_true", help="Evaluate the human reference summaries.")
    ap.add_argument("--metrics", default=",".join(ALL_METRICS),
                    help="Comma-separated subset of: " + ",".join(ALL_METRICS))
    ap.add_argument("--judge-model", default="llama3.1:8b", help="Ollama model used as DeepEval judge.")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--report", action="store_true", help="Print mean/std tables from the CSV.")
    args = ap.parse_args()

    if args.report:
        report(args.out)
        return
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = set(metrics) - set(ALL_METRICS)
    if unknown:
        ap.error(f"unknown metrics: {', '.join(sorted(unknown))}")

    records = reference_records() if args.references else []
    if args.inputs:
        if not args.system:
            ap.error("--system is required with --inputs")
        records += collect_records(args.inputs, args.system)
    if not records:
        ap.error("nothing to evaluate: give --inputs or --references")

    refs = load_references() if ("rouge" in metrics or "bart" in metrics) else None
    tweets_cache = {}
    rows = []
    for n, rec in enumerate(records, 1):
        print(f"[{n}/{len(records)}] {rec['system']} {rec['topic']} prompt {rec['prompt_id'] or '-'}")
        tweets = tweets_cache.setdefault(rec["topic"], load_tweets(rec["topic"]))
        row = {k: v for k, v in rec.items() if k in FIELDS}
        start = time.perf_counter()
        if "rouge" in metrics:
            row.update(metric_rouge(rec, refs))
        if "gap" in metrics:
            row.update(metric_gap(rec, tweets))
        if "ec" in metrics:
            row.update(metric_ec(rec, tweets))
        if "deepeval" in metrics:
            row.update(metric_deepeval(rec, tweets, args.judge_model))
        if "bart" in metrics:
            row.update(metric_bart(rec, tweets, refs))
        row["eval_seconds"] = round(time.perf_counter() - start, 2)
        print("   ", {k: row[k] for k in FIELDS[5:] if row.get(k) not in (None, "")})
        append_rows([row], args.out)  # saved immediately, so Ctrl+C loses nothing finished
        rows.append(row)
    print(f"\nAppended {len(rows)} row(s) to {args.out}")


if __name__ == "__main__":
    main()