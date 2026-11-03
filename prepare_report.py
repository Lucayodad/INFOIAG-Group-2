"""Evaluate saved completed trials and audit revision outcomes without generating summaries.

Example: python prepare_report.py --out results/eval/report_release --metrics cost,rouge
Optional model metrics: gap,ec,bart. The default needs no downloaded model weights.
"""

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import evaluate


ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_agent(record):
    path = Path(record["file"])
    data = json.loads(path.read_text(encoding="utf-8"))
    agent = data.get("agent_result") or data
    initial, final = agent["initial_summary"], agent["current_summary"]
    rep, facts = agent["representation_results"], agent["factual_results"]
    return {"file": str(path.relative_to(ROOT)), "file_sha256": sha(path),
        "topic": record["topic"], "prompt_id": record["prompt_id"], "seed": record["seed"],
        "summary_unchanged": initial["sentences"] == final["sentences"],
        "status": agent["status"], "stop_reason": agent["stop_reason"],
        "revision_attempts": agent["revision_count"],
        "revision_outcomes": dict(Counter(r["status"] for r in agent["revisions"])),
        "factual_verdicts": facts["verdict_counts"], "representation_status": rep["status"],
        "online_metric": rep["metric"], "online_metric_version": rep.get("metric_version", "legacy_atomic_claim_hypotheses"),
        "online_coverage": {g: rep["coverage_by_group"][g]["coverage"] for g in evaluate.GROUPS},
        "online_equal_coverage": rep.get("equal_coverage"),
        "usage_complete": record["usage_complete"]}


def saved_metric_cache(records):
    """Reuse committed scores only for unchanged inputs and identical metric inputs.

    Historical CSVs predate hashes. Require their original summaries, datasets,
    references and CSVs to match HEAD before constructing content-based keys.
    This does not certify the old model judgments or recover missing versions.
    """
    csvs = [ROOT / "results/eval/metrics.csv", ROOT / "results/eval/bart.csv"]
    paths = list(evaluate.DATA.glob("*.json*")) + csvs
    paths += list({Path(r["file"]) for r in records if Path(r["file"]).is_file()})
    relative = [p.relative_to(ROOT).as_posix() for p in paths]
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
    subprocess.run([*git, "ls-files", "--error-unmatch", "--", *relative], cwd=ROOT,
                   check=True, capture_output=True)
    checked = subprocess.run([*git, "diff", "--quiet", "HEAD", "--", *relative], cwd=ROOT)
    if checked.returncode:
        raise ValueError("Saved-score reuse requires unchanged committed source summaries, data and CSVs.")
    columns = {"gap": ("sim_gap", "sim_AA", "sim_Hisp", "sim_White"),
               "ec": ("ec", "cov_AA", "cov_Hisp", "cov_White", "cov_all"),
               "bart": ("bart_src", "bart_ref")}
    identity = lambda r: tuple(str(r.get(k, "")) for k in ("system", "topic", "prompt_id", "seed"))
    by_identity = {identity(r): r for r in records}
    # Human references use their file suffix to distinguish annotators.
    cache, origins = {}, {}
    for path in csvs:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for number, row in enumerate(rows, 2):
            if row["system"] == "human_reference":
                rec = next((r for r in records if r["system"] == "human_reference"
                            and r["file"] == row["file"]), None)
            else:
                rec = by_identity.get(identity(row))
                if rec is not None and Path(row["file"].replace("\\", "/")).name != Path(rec["file"]).name:
                    raise ValueError(f"Historical score points to a different trial file: {row['file']}")
            if rec is None:
                continue
            for metric, fields in columns.items():
                if not all(row.get(field) not in (None, "") for field in fields):
                    continue
                value = {field: float(row[field]) for field in fields}
                key = (metric, rec["topic"], rec["summary"], tuple(rec["sentences"]), rec.get("_ref_index"))
                if key in cache and cache[key] != value:
                    raise ValueError("Conflicting saved scores for identical metric inputs; rescore explicitly.")
                cache[key] = value
                origins[key] = {"mode": "historical_committed_score_unchanged_inputs",
                    "csv": str(path.relative_to(ROOT)), "csv_sha256": sha(path), "row": number,
                    "original_trial_file": row["file"], "model_versions": "not_recorded_in_legacy_results"}
    return cache, origins


def build_report(out, metrics, reuse_saved=False):
    if out.resolve() == (ROOT / "results/eval").resolve():
        raise ValueError("Use a new dedicated folder; original evaluation files must be preserved.")
    if (out / "metrics.csv").exists() and not (out / "audit.json").exists():
        raise ValueError("Output contains an unrelated CSV; choose a dedicated report folder.")
    out.mkdir(parents=True, exist_ok=True)
    baseline_dirs = sorted((ROOT / "results/baseline_pc/experiments").glob("*/manifest.json"))
    agent_dirs = sorted((ROOT / "results").glob("agent*mnli/experiments/*/manifest.json"))
    records = evaluate.collect_records(baseline_dirs, "baseline") + evaluate.collect_records(agent_dirs, "agent")
    finals = [r for r in records if r["system"] == "agent"]
    baselines = [r for r in records if r["system"] == "baseline"]
    expected = {(t, str(p), s) for t in evaluate.TOPICS for p in (1, 2) for s in (101, 202, 303)}
    for label, rows in (("baseline", baselines), ("agent", finals)):
        keys = {(r["topic"], r["prompt_id"], r["seed"]) for r in rows}
        if keys != expected or len(rows) != len(expected):
            raise ValueError(f"Incomplete or duplicated {label} design: {len(rows)} rows, missing {expected - keys}")
    records += evaluate.reference_records()
    audit = [audit_agent(r) for r in finals]
    baseline_by_trial = {(r["topic"], r["prompt_id"], r["seed"]): r for r in baselines}
    for rec, observation in zip(finals, audit):
        data = json.loads(Path(rec["file"]).read_text(encoding="utf-8"))
        initial = (data.get("agent_result") or data)["initial_summary"]["summary"]
        observation["initial_matches_paired_baseline"] = initial == baseline_by_trial[
            (rec["topic"], rec["prompt_id"], rec["seed"])]["summary"]
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in baseline_dirs + agent_dirs}
    inputs.update({str(p.relative_to(ROOT)): sha(p) for p in evaluate.DATA.glob("*.json*")})
    inputs.update({str(p.relative_to(ROOT)): sha(p) for p in ROOT.glob("*.py")})
    for rec in records:
        p = Path(rec["file"])
        if p.is_file():
            inputs[str(p.relative_to(ROOT))] = sha(p)
    revision = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    metadata = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(), "base_commit": revision,
        "working_tree_changes": "See git diff; this validation precedes any commit.",
        "metrics_requested": metrics, "summary_generation": "none; existing real outputs only",
        "saved_metric_reuse": reuse_saved,
        "unit": "one completed trial per system/topic/prompt/seed; drafts labelled separately",
        "ec_definition": "mean absolute group deviation from pooled tweet-sentence entailment mean",
        "ec_version": "mnli_sentence_mean_v1", "threshold": "0.25 exploratory; not tuned here",
        "cost_scope": "Legacy agent token totals are incomplete and excluded from complete token columns.",
        "input_sha256": inputs, "agent_audit": audit}
    (out / "audit.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = out / "metrics.csv"
    refs, tweets = evaluate.load_references(), {}
    cache, origins = saved_metric_cache(records) if reuse_saved else ({}, {})
    functions = {"rouge": lambda r, t: evaluate.metric_rouge(r, refs),
                 "gap": evaluate.metric_gap, "ec": evaluate.metric_ec,
                 "bart": lambda r, t: evaluate.metric_bart(r, t, refs)}
    for i, rec in enumerate(records, 1):
        print(f"[{i}/{len(records)}] {rec['system']} {rec['topic']} prompt {rec['prompt_id']} seed {rec.get('seed', '-')}", flush=True)
        source = tweets.setdefault(rec["topic"], evaluate.load_tweets(rec["topic"]))
        row = {key: value for key, value in rec.items() if key in evaluate.FIELDS}
        provenance = {}
        for name in metrics:
            if name == "cost":
                continue
            key = (name, rec["topic"], rec["summary"], tuple(rec["sentences"]), rec.get("_ref_index"))
            if key not in cache:
                cache[key] = functions[name](rec, source)
                origins[key] = {"mode": "fresh_local_evaluation"}
            row.update(cache[key])
            provenance[name] = origins[key]
            if name == "ec" and origins[key]["mode"].startswith("historical"):
                row["ec_version"] = "legacy_offline_sentence_mean"
        row["metric_provenance"] = json.dumps(provenance, sort_keys=True)
        evaluate.append_rows([row], csv_path)
    evaluate.report(csv_path)
    metadata["status"] = "completed"
    metadata["package_versions"] = {}
    for name in ("torch", "transformers", "sentence-transformers", "rouge-score", "langgraph"):
        try:
            metadata["package_versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            metadata["package_versions"][name] = None
    # Cache inspection is read-only and records immutable revisions actually
    # downloaded, without treating a mutable model tag as a pinned revision.
    try:
        from huggingface_hub import scan_cache_dir
        metadata["model_cache_revisions"] = {
            repo.repo_id: sorted(rev.commit_hash for rev in repo.revisions)
            for repo in scan_cache_dir().repos}
    except (ImportError, OSError):
        metadata["model_cache_revisions"] = {}
    (out / "audit.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    unchanged = sum(a["summary_unchanged"] for a in audit)
    baseline_matches = sum(a["initial_matches_paired_baseline"] for a in audit)
    statuses = dict(Counter(a["status"] for a in audit))
    stops = dict(Counter(a["stop_reason"] for a in audit))
    outcomes = Counter()
    for a in audit:
        outcomes.update(a["revision_outcomes"])
    findings = f"""# Saved experiment: auditable report results

This evaluates saved real summaries, not a rerun of generation.
Reuse of committed historical semantic scores: {reuse_saved}. Every metric's
origin is recorded in the CSV. Reused scores require unchanged committed source
files and identical text/corpus/reference inputs; they are not fresh inference.
There are {len(baselines)} baseline trials and {len(finals)} agent trials:
3 topics x 2 prompts x 3 seeds per system. Initial agent drafts and the six human
references are separate comparison categories, never additional agent trials.

Agent outcomes: {statuses}. Stop reasons: {stops}.
Final summaries unchanged from their own initial drafts: {unchanged}/{len(finals)}.
Revision outcomes: {dict(outcomes)}.
Initial drafts identical to their paired baseline text: {baseline_matches}/{len(finals)}.

These runs do not demonstrate an improvement caused by revision. Comparison
against each agent's own draft isolates the checking/revision effect; comparison
against separately generated baselines also includes generation variation.
Completed-with-uncertainty is an unresolved result, not factual verification.

Metrics in this folder: {', '.join(metrics)}. Missing metrics are unmeasured.
The original results/eval CSV files and experiment JSON files are preserved.
Every source file is hashed in audit.json; CSV metric cells show their measured
sample counts. Standard deviations describe these seeds, not significance.

MNLI EC is a disparity proxy: equally low coverage can have EC close to zero.
Report cov_AA, cov_Hisp, cov_White and cov_all alongside EC. MNLI probabilities
are model estimates, not proof of fairness or support. Fresh online/offline EC
uses the same original summary sentences; legacy online EC used atomic claims
and must not be compared directly with the sentence scores in the tables.

Legacy agent token totals omit extraction/recheck usage. Their input/output
token columns are blank here. Recorded elapsed time is hardware-dependent,
includes model loading, and is not a controlled speed comparison across machines.
New runs save an actual-call ledger and explicit usage scope/completeness.

The 0.25 routing threshold is exploratory and was not tuned to obtain revisions.
No new 36-run generation experiment was performed. See report.md for tables.
"""
    (out / "findings.md").write_text(findings, encoding="utf-8")
    print(findings, flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New dedicated report output folder")
    parser.add_argument("--metrics", default="cost,rouge", help="cost,rouge,gap,ec,bart")
    parser.add_argument("--reuse-saved-metrics", action="store_true",
                        help="Reuse committed semantic scores after checking unchanged inputs; keep provenance")
    args = parser.parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics or set(metrics) - {"cost", "rouge", "gap", "ec", "bart"}:
        parser.error("Choose cost,rouge,gap,ec,bart; DeepEval is a separate optional evaluation.")
    build_report(args.out, metrics, args.reuse_saved_metrics)


if __name__ == "__main__":
    main()
