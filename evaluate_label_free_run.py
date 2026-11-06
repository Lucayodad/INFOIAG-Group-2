"""Final evaluation of the label-free run: the only step that reads dialect labels.

For every trial of a run_final_agent.py output folder, the agent's final
summary is compared with its own draft: fact verdicts (from the agent's own
checks) and MNLI Equal Coverage per dialect group. The labels are joined from
the dataset here, after the agent has finished; the agent never saw them.
Writes <run>/evaluation.json and prints one row per trial.

    python evaluate_label_free_run.py [--run results/agent_checker_v2]
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from data_loader import DIALECT_LABELS, load_tweets
from equal_coverage import CachedScorer, calculate_equal_coverage

ROOT = Path(__file__).resolve().parent


def group_coverage(tweets_by_group, summary, scorer):
    result = calculate_equal_coverage(tweets_by_group, [{"claim": s["text"]} for s in summary["sentences"]],
                                      scorer=scorer)
    coverage = {g: result["coverage_by_group"][g]["coverage"] for g in DIALECT_LABELS}
    return {"equal_coverage": result["equal_coverage"], "coverage": coverage, "worst_group": min(coverage.values()),
            "four_fifths_disparity": 1 - min(coverage.values()) / max(coverage.values()),
            "metric_version": result["metric_version"]}


def verdicts(iteration):
    return dict(Counter(a["verdict"] for a in iteration["factual_results"]["assessments"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "results" / "agent_checker_v2")
    parser.add_argument("--cache", type=Path, help="Optional JSON file of MNLI pair scores, reused and updated")
    args = parser.parse_args()
    scorer, tweets, rows = CachedScorer(), {}, {}
    if args.cache and args.cache.exists():
        scorer.cache.update({(premise, hypothesis): score for premise, hypothesis, score
                             in json.loads(args.cache.read_text(encoding="utf-8"))})
    for path in sorted(args.run.glob("*_prompt*_seed*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        result = result.get("agent_result", result)
        topic = result["topic"]
        if topic not in tweets:
            loaded = load_tweets(topic)
            tweets[topic] = {g: [{"text": t.text} for t in loaded if t.dialect_label == g] for g in DIALECT_LABELS}
        draft, final = result["initial_summary"], result["current_summary"]
        first, best = result["iterations"][0], result["iterations"][result["best_iteration"]]
        rows[path.stem] = {
            "changed": draft["sentences"] != final["sentences"],
            "revisions": dict(Counter(r["status"] for r in result["revisions"])),
            "stop_reason": result["stop_reason"],
            "verdicts_draft": verdicts(first),
            "verdicts_final": verdicts(best),
            "draft": group_coverage(tweets[topic], draft, scorer),
            "final": group_coverage(tweets[topic], final, scorer),
        }
        row = rows[path.stem]
        print(f"{path.stem:28} changed={row['changed']!s:5} contradicted "
              f"{row['verdicts_draft'].get('CONTRADICTED', 0)}->{row['verdicts_final'].get('CONTRADICTED', 0)} "
              f"EC {row['draft']['equal_coverage']:.4f}->{row['final']['equal_coverage']:.4f} "
              f"worst {row['draft']['worst_group']:.4f}->{row['final']['worst_group']:.4f}", flush=True)
        if args.cache:
            args.cache.write_text(json.dumps([[*pair, score] for pair, score in scorer.cache.items()]),
                                  encoding="utf-8")
    totals = {side: dict(sum((Counter(row[f"verdicts_{side}"]) for row in rows.values()), Counter()))
              for side in ("draft", "final")}
    summary = {"trials": len(rows), "changed": sum(row["changed"] for row in rows.values()), "verdicts": totals,
               "mean_equal_coverage": {side: sum(row[side]["equal_coverage"] for row in rows.values()) / len(rows)
                                       for side in ("draft", "final")},
               "labels_used_by": "this evaluation only (dataset labels joined after the agent finished)",
               "rows": rows}
    (args.run / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n{summary['changed']}/{summary['trials']} changed | verdicts draft {totals['draft']} -> final {totals['final']}")
    print("Mean Equal Coverage (lower is fairer): draft {draft:.4f} -> final {final:.4f}".format(**summary["mean_equal_coverage"]))


if __name__ == "__main__":
    main()
