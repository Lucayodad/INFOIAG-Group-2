"""Rebuild the report's numerical tables from saved evidence, without model calls.

    python reproduce_report.py

Writes results/report/tables.md, numbers.json, ontology_queries.json, figure
sources and an input-hash manifest. Fresh Qwen runs and neural metric scoring
remain separate commands documented in README.md.
"""

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

from scipy.stats import binomtest, friedmanchisquare, wilcoxon

from data_loader import DIALECT_LABELS, TOPICS
from ontology_tools import OntologyTools, QUERY_DEFAULTS
from run_queries import with_test_assertions

ROOT = Path(__file__).resolve().parent
CHECKS = ROOT / "results/checker_evaluation"
VERDICTS = ("SUPPORTED", "CONTRADICTED", "UNKNOWN")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def accuracy(predictions, labels, keys):
    """Strict paired scoring: missing predictions are an error, not exclusions."""
    keys = list(keys)
    if not keys or any(key not in predictions or key not in labels for key in keys):
        raise ValueError("Incomplete checker predictions or reference labels.")
    tp = sum(labels[k] == predictions[k] == "SUPPORTED" for k in keys)
    supported = sum(predictions[k] == "SUPPORTED" for k in keys)
    reference_supported = sum(labels[k] == "SUPPORTED" for k in keys)
    return {"n": len(keys), "accuracy": sum(predictions[k] == labels[k] for k in keys) / len(keys),
            "precision": tp / supported if supported else None,
            "recall": tp / reference_supported if reference_supported else None,
            "false_unknown": sum(labels[k] == "SUPPORTED" and predictions[k] == "UNKNOWN" for k in keys),
            "false_supported": sum(labels[k] != "SUPPORTED" and predictions[k] == "SUPPORTED" for k in keys),
            "false_contradicted": sum(labels[k] != "CONTRADICTED" and predictions[k] == "CONTRADICTED" for k in keys)}


def load_guard():
    path = CHECKS / "scripts/guard.py"
    spec = importlib.util.spec_from_file_location("report_checker_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.final_guard


def checker_tables():
    claims = {c["key"]: c for c in read_json(CHECKS / "scripts/claims.json")}
    reference = {r["key"]: r for r in csv_rows(CHECKS / "reference_labels.csv")}
    if set(claims) != set(reference) or len(claims) != 210:
        raise ValueError("The report requires the same 210 claims and reference labels.")
    labels = {key: row["reference_verdict"] for key, row in reference.items()}
    original = {key: claim["baseline"] for key, claim in claims.items()}
    dev = [key for key in claims if claims[key]["topic"] == "McDonalds"]
    held_out = [key for key in claims if key not in dev]
    guard = load_guard()

    def recorded(name, guarded=False):
        rows = [json.loads(line) for line in (CHECKS / f"runs/{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len({row["key"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate claims in {name}.")
        return {row["key"]: guard(claims[row["key"]]["claim"], row["verdict"], row["references"])[0]
                if guarded else row["verdict"] for row in rows}

    development = {"Original": accuracy(original, labels, dev)}
    for name in ("A", "B", "D", "C"):
        development[name] = accuracy(recorded("dev_" + name), labels, dev)
    development["Final (C + rules)"] = accuracy(recorded("dev_C", True), labels, dev)
    final = recorded("test_C", True)
    comparison = {"Original": accuracy(original, labels, held_out),
                  "Revised": accuracy(final, labels, held_out)}
    high = [key for key in held_out if reference[key]["confidence"] == "high"]
    comparison["Original, high confidence"] = accuracy(original, labels, high)
    comparison["Revised, high confidence"] = accuracy(final, labels, high)
    by_topic = {topic: {"original": accuracy(original, labels, [k for k in held_out if claims[k]["topic"] == topic]),
                        "revised": accuracy(final, labels, [k for k in held_out if claims[k]["topic"] == topic])}
                for topic in ("Obama", "Superbowl")}
    corrected = sum(original[k] != labels[k] and final[k] == labels[k] for k in held_out)
    broken = sum(original[k] == labels[k] and final[k] != labels[k] for k in held_out)
    sample = [k for k in claims if reference[k]["in_validation_sample"] == "yes"]
    sheets = {name: {r["key"]: r["your_verdict"] for r in csv_rows(CHECKS / filename)}
              for name, filename in (("Saif", "annotation_sheet_60_Saif.csv"),
                                     ("Muhire", "annotation_sheet_60_Muhire.csv"))}
    agreement = {}
    for first, second in (("Saif", "Muhire"), ("Saif", "reference"), ("Muhire", "reference")):
        a, b = sheets[first], labels if second == "reference" else sheets[second]
        if any(k not in a or k not in b or a[k] not in VERDICTS or b[k] not in VERDICTS for k in sample):
            raise ValueError("The annotation sample must contain 60 complete labels per annotator.")
        observed = sum(a[k] == b[k] for k in sample) / len(sample)
        expected = sum(sum(a[k] == v for k in sample) * sum(b[k] == v for k in sample)
                       for v in VERDICTS) / len(sample) ** 2
        agreement[first + " vs " + second] = {"n": len(sample), "agreement": observed,
                                             "kappa": (observed - expected) / (1 - expected)}
    matrix = {gold: {pred: sum(labels[k] == gold and original[k] == pred for k in claims)
                     for pred in VERDICTS} for gold in VERDICTS}
    return {"table10_confusion": matrix, "table10_metrics": accuracy(original, labels, claims),
            "table11_development": development, "table12_held_out": comparison,
            "held_out_by_topic": by_topic,
            "mcnemar": {"corrected": corrected, "broken": broken,
                        "p": float(binomtest(min(corrected, broken), corrected + broken).pvalue)},
            "annotation_agreement": agreement,
            "annotation_provenance": "Saif and Muhire are sheet identifiers; their supplied labels originated from Claude Sonnet and Claude Opus, respectively.",
            "unsupported_claims_with_uncited_support": sum(r["reference_verdict"] == "UNKNOWN" and
                                                         bool(r["supporting_uncited_tweets"].strip()) for r in reference.values())}


def trial_summary(folder):
    rows = {}
    for path in sorted((ROOT / "results" / folder).glob("*_prompt*_seed*.json")):
        value = read_json(path)
        value = value.get("agent_result", value)
        first, best = value["iterations"][0], value["iterations"][value["best_iteration"]]
        def verdicts(iteration):
            counts = Counter(a["verdict"] for a in iteration["factual_results"]["assessments"])
            return {verdict: counts[verdict] for verdict in VERDICTS}
        refs = [r["evidence_id"] for a in best["factual_results"]["assessments"] if a["verdict"] == "SUPPORTED" for r in a["references"]]
        rows[path.stem] = {"topic": value["topic"], "draft": verdicts(first), "final": verdicts(best),
            "changed": value["initial_summary"]["sentences"] != value["current_summary"]["sentences"],
            "proportion_reason": first.get("proportion_results", {}).get("reason"),
            "revisions": dict(Counter(r["status"] for r in value["revisions"])),
            "seconds": value.get("provenance", {}).get("seconds"), "calls": value.get("llm_calls"),
            "tokens": (value.get("usage") or {}).get("total_tokens"),
            "tool_calls": value["call_counts"]["tool_calls"],
            "supported_tweet_references": sum("/tweet/" in ref for ref in refs),
            "supported_claim_references": sum("/claim/" in ref for ref in refs),
            "label_checks": sum(int(n) for n in re.findall(r"\((\d+) checked", value.get("provenance", {}).get("dialect_labels", "")))}
    if len(rows) != 18:
        raise ValueError(f"Expected 18 completed trials in {folder}, found {len(rows)}.")
    return rows


def fairness(rows, side):
    values = [row[side] for row in rows]
    return {"n": len(values), "ec": statistics.mean(v["equal_coverage"] for v in values),
            "worst_group": statistics.mean(v["worst_group"] for v in values),
            "four_fifths_disparity": statistics.mean(v["four_fifths_disparity"] for v in values),
            "passing": sum(v["four_fifths_disparity"] <= 0.20 for v in values),
            **{g: statistics.mean(v["coverage"][g] for v in values) for g in DIALECT_LABELS}}


def experiment_tables():
    folders = {"original": "agent_compliant_v2", "revised": "agent_checker_v2", "proportion": "agent_checker_v2_proportion"}
    trials = {name: trial_summary(folder) for name, folder in folders.items()}
    evaluations = {name: read_json(ROOT / "results" / folder / "evaluation.json")["rows"] for name, folder in folders.items()}
    expected = set(trials["revised"])
    if any(set(evaluation) != expected for evaluation in evaluations.values()):
        raise ValueError("Fairness evaluations and saved trials must cover the same 18 trials.")
    facts = {f"{name}_{side}": dict(sum((Counter(r[side]) for r in rows.values()), Counter()))
             for name, rows in trials.items() for side in ("draft", "final")}
    facts = {name: {verdict: counts.get(verdict, 0) for verdict in VERDICTS}
             for name, counts in facts.items()}
    coverage = {"draft": fairness(evaluations["revised"].values(), "draft")}
    coverage.update({name: fairness(rows.values(), "final") for name, rows in evaluations.items()})
    by_topic = {topic: fairness([row for key, row in evaluations["revised"].items() if key.startswith(topic + "_")], "draft")
                for topic in TOPICS}
    ordered = sorted(evaluations["revised"])
    groups = {g: [evaluations["revised"][key]["draft"]["coverage"][g] for key in ordered] for g in DIALECT_LABELS}
    tests = {"friedman_p": float(friedmanchisquare(*groups.values()).pvalue), "wilcoxon_p": {
        a + " vs " + b: float(wilcoxon(groups[a], groups[b]).pvalue)
        for a, b in (("AA", "Hisp"), ("AA", "White"), ("Hisp", "White"))}}
    tests["worst_group_counts"] = dict(Counter(min(groups, key=lambda g: groups[g][i]) for i in range(18)))
    quality = {}
    for system in ("baseline", "agent_draft", "agent", "human_reference", "agent_proportion"):
        rows = [r for r in csv_rows(ROOT / "results/eval/metrics.csv") if r["system"] == system]
        if len(rows) != (6 if system == "human_reference" else 18):
            raise ValueError(f"Unexpected number of metric rows for {system}.")
        quality[system] = {field: {"mean": statistics.mean(float(r[field]) for r in rows),
                                   "sd": statistics.stdev(float(r[field]) for r in rows)}
                           for field in ("rouge1", "rouge2", "rougeL", "bart_src", "bart_ref")}
    costs = {}
    for name, rows in trials.items():
        costs[name] = {field: {"mean": statistics.mean(r[field] for r in rows.values()),
                               "sd": statistics.stdev(r[field] for r in rows.values()),
                               "min": min(r[field] for r in rows.values()), "max": max(r[field] for r in rows.values())}
                       for field in ("seconds", "calls", "tool_calls", "tokens") if all(r[field] is not None for r in rows.values())}
    revised = trials["revised"]
    return {"table13_factual": facts, "table14_fairness": coverage, "table15_topics": by_topic,
            "group_tests": tests, "table16_proportion": dict(Counter(r["proportion_reason"] for r in trials["proportion"].values())),
            "table17_quality": quality, "table17_agent_cost": costs,
            "summaries_changed": {name: sum(r["changed"] for r in rows.values()) for name, rows in trials.items()},
            "revised_run_provenance": {"label_checked_requests": sum(r["label_checks"] for r in revised.values()),
                "supported_tweet_references": sum(r["supported_tweet_references"] for r in revised.values()),
                "supported_claim_references": sum(r["supported_claim_references"] for r in revised.values())}}


def markdown_table(title, rows, fields):
    lines = [f"## {title}\n", "| Condition | " + " | ".join(fields) + " |", "|---|" + "---|" * len(fields)]
    for name, row in rows.items():
        format_value = lambda value: f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append("| " + name + " | " + " | ".join(format_value(row.get(field, "-")) for field in fields) + " |")
    return "\n".join(lines) + "\n\n"


def citation_probe():
    """Evaluate the reported retrieval diagnostic from its cached embeddings."""
    cache = CHECKS / "scripts/embeddings.json"
    if not cache.exists():
        return {"status": "missing_embedding_cache", "command":
                "python results/checker_evaluation/scripts/lab_recite.py --cache-embeddings"}
    import html
    from data_loader import load_tweets
    sys.path.insert(0, str(CHECKS / "scripts"))
    try:
        from lab_recite import candidates
    finally:
        sys.path.pop(0)
    vectors = read_json(cache)
    claims = {c["key"]: c for c in read_json(CHECKS / "scripts/claims.json")}
    tweets = {topic: [(t.id, html.unescape(t.text)) for t in load_tweets(topic)] for topic in TOPICS}
    rows = {}
    for reference in csv_rows(CHECKS / "reference_labels.csv"):
        support = set(reference["supporting_uncited_tweets"].split())
        if reference["reference_verdict"] != "UNKNOWN" or not support:
            continue
        key = reference["key"]
        claim = claims[key]
        closest = candidates(claim["claim"], set(claim["cited"]), claim["topic"],
                             tweets[claim["topic"]], vectors, vectors[key], 5)
        rows[key] = {"top5": closest, "supporting_ids": sorted(support), "hit": bool(support & set(closest))}
    return {"status": "computed", "claims": len(rows), "hits": sum(r["hit"] for r in rows.values()),
            "cache_provenance": read_json(cache.with_suffix(".metadata.json")), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/report")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    numbers = {"ontology": OntologyTools().statistics(), "checker": checker_tables(),
               "experiment": experiment_tables(), "citation_retrieval_probe": citation_probe()}
    (args.out / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    checker, experiment = numbers["checker"], numbers["experiment"]
    text = "# Reproduced report numbers\n\nComputed from saved source evidence, not new model calls. See numbers.json for full precision and all statistics.\n\n"
    text += markdown_table("Table 10: original checker", checker["table10_confusion"], VERDICTS)
    fields = ("n", "accuracy", "precision", "recall", "false_unknown", "false_supported", "false_contradicted")
    text += markdown_table("Table 11: development", checker["table11_development"], fields)
    text += markdown_table("Table 12: held-out", checker["table12_held_out"], fields)
    text += f"Exact McNemar: {checker['mcnemar']}.\n\n"
    text += markdown_table("Annotator agreement (LLM sheets)", checker["annotation_agreement"], ("n", "agreement", "kappa"))
    text += markdown_table("Table 13: factual verdicts", experiment["table13_factual"], VERDICTS)
    text += markdown_table("Table 14: fairness", experiment["table14_fairness"], ("n", "ec", "worst_group", "four_fifths_disparity", "passing"))
    text += markdown_table("Table 15: topics", experiment["table15_topics"], ("ec", "worst_group", "four_fifths_disparity", "AA", "Hisp", "White"))
    text += f"Paired group tests: {experiment['group_tests']}.\n\nTable 16: {experiment['table16_proportion']}.\n\n"
    quality = {system: {field: value["mean"] for field, value in row.items()} for system, row in experiment["table17_quality"].items()}
    text += markdown_table("Table 17: quality", quality, ("rouge1", "rouge2", "rougeL", "bart_src", "bart_ref"))
    cost = {system: {field: value["mean"] for field, value in row.items()} for system, row in experiment["table17_agent_cost"].items()}
    text += markdown_table("Table 17: checking-phase cost", cost, ("seconds", "calls", "tool_calls", "tokens"))
    text += f"Summaries changed: {experiment['summaries_changed']}.\n\nProvenance: {experiment['revised_run_provenance']}.\n"
    probe = numbers["citation_retrieval_probe"]
    text += f"\nCitation retrieval diagnostic: {probe['status']}; top-5 hits {probe.get('hits', '-')}/{probe.get('claims', '-')}.\n"
    (args.out / "tables.md").write_text(text, encoding="utf-8")
    plain, example = OntologyTools(), with_test_assertions()
    queries = {query: {"asserted": (example if query in ("Q7", "Q8") else plain).run_query(query, use_inference=False),
                       "inferred": (example if query in ("Q7", "Q8") else plain).run_query(query, use_inference=True)}
               for query in QUERY_DEFAULTS}
    (args.out / "ontology_queries.json").write_text(json.dumps(queries, indent=2), encoding="utf-8")
    from export_report_figures import export_figures
    export_figures(args.out / "figures")
    source_paths = [p for p in ROOT.rglob('*') if p.is_file() and not any(part in ('.git', '.venv', '__pycache__') for part in p.relative_to(ROOT).parts)
                    and not p.is_relative_to(args.out.resolve()) and p.suffix in ('.py', '.csv', '.jsonl', '.json', '.owl', '.ttl', '.rq')]
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths)}
    (args.out / "inputs.sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Reproduced report tables and figure sources in {args.out}")
    print(f"Held-out accuracy: {checker['table12_held_out']['Revised']['accuracy']:.2f}; McNemar p={checker['mcnemar']['p']:.4f}")
    print(f"Final-run factual counts: {experiment['table13_factual']['revised_final']}")


if __name__ == "__main__":
    main()
