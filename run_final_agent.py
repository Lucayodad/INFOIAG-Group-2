"""Rerun the 18 agent trials with the current agent, reusing each recorded first check.

Every recorded rerun in results/agent_rerun holds its draft, its RDF graph and
its first check. The draft and graph are reused unchanged. The first check's
claim extraction and fact verdicts seed the agent's cache: they run at
temperature 0 and every recorded recheck repeated them identically (133/133),
so only the representation check, revisions and their rechecks are computed.
Sentences whose recorded extraction failed are extracted and checked anew.
Finished trials are skipped, so rerunning the same command resumes.

    python run_final_agent.py --out results/agent_final [--trial Obama_prompt1]
"""

import argparse
import json
from pathlib import Path
import time
import traceback

from rdflib import Graph

from ollama_backend import OllamaClient
from rerun_agent_trials import summarize
from summary_agent import print_run_summary, run_summary_agent, seed_check_cache

ROOT = Path(__file__).resolve().parent
RECORDED = ROOT / "results" / "agent_rerun"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "agent_final")
    parser.add_argument("--trial", nargs="*", default=[], help="Only trials whose name contains one of these")
    parser.add_argument("--disparity-threshold", type=float, default=0.2)
    parser.add_argument("--max-revisions", type=int, default=3)
    args = parser.parse_args()

    names = sorted(path.stem for path in RECORDED.glob("*_prompt*_seed*.json"))
    if len(names) != 18:
        parser.error(f"Expected 18 recorded trials in {RECORDED}, found {len(names)}.")
    if args.trial:
        names = [name for name in names if any(part in name for part in args.trial)]
    args.out.mkdir(parents=True, exist_ok=True)
    client = OllamaClient()
    failures = []
    for name in names:
        output = args.out / f"{name}.json"
        if output.exists():
            print(f"Skipping {name}: already finished", flush=True)
            continue
        recorded = json.loads((RECORDED / f"{name}.json").read_text(encoding="utf-8"))
        recorded = recorded.get("agent_result", recorded)
        graph_path = Path(recorded["pipeline"]["graph"].replace("\\", "/"))  # recorded on Windows
        graph_path = graph_path if graph_path.is_absolute() else ROOT / graph_path
        print(f"Running {name} at {time.strftime('%H:%M')}", flush=True)
        started = time.time()
        try:
            result = run_summary_agent(
                Graph().parse(graph_path, format="turtle"), recorded["initial_summary"],
                topic=recorded["topic"], prompt=recorded["prompt"], client=client,
                disparity_threshold=args.disparity_threshold, max_revisions=args.max_revisions,
                check_cache=seed_check_cache(recorded["initial_summary"], recorded["iterations"][0]))
        except Exception:  # noqa: BLE001 - record and continue with the next trial
            traceback.print_exc()
            failures.append(name)
            print(f"FAILED {name}; rerun the same command to retry it.", flush=True)
            continue
        result["provenance"] = {
            "recorded_run": str((RECORDED / f"{name}.json").relative_to(ROOT)),
            "graph": str(graph_path.relative_to(ROOT)),
            "first_check": "claim extraction and fact verdicts reused from the recorded run "
                           "(unclear extractions redone); representation and revisions computed in this run",
            "seconds": round(time.time() - started),
        }
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print_run_summary(result, output)
        print(f"  {name} took {result['provenance']['seconds'] / 60:.1f} min", flush=True)

    rows = {path.stem: summarize(path) for path in sorted(args.out.glob("*_prompt*_seed*.json"))}
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    changed = sum(row["changed"] for row in rows.values())
    print(f"\n{len(rows)}/{len(names) if args.trial else 18} finished; {changed} summaries changed by an accepted revision.")
    if failures:
        print("Failed: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
