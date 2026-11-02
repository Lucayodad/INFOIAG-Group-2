"""Rerun the 18 saved agent trials from their own committed drafts.

Each trial reuses its saved initial draft, so differences from the saved
results come from checking and revision only, not from generation. The first
run of a topic creates and checkpoints its source extractions (90 Qwen calls).
Finished trials are skipped, so rerunning the same command resumes.

    python rerun_agent_trials.py --out results/agent_rerun
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PATTERN = re.compile(r"ontology_agent_(Obama|McDonalds|Superbowl)_prompt([12])_seed(\d+)\.json$")
TOPIC_ORDER = ("Obama", "McDonalds", "Superbowl")  # Obama first: its extractions usually exist


def saved_drafts():
    drafts = {}
    for path in sorted(ROOT.glob("results/agent_*/experiments/*/drafts/ontology_agent_*.json")):
        match = PATTERN.search(path.name)
        if match:
            key = match.groups()
            if key in drafts and drafts[key].read_bytes() != path.read_bytes():
                raise ValueError(f"Conflicting saved drafts for {key}: {drafts[key]} and {path}")
            drafts.setdefault(key, path)
    return dict(sorted(drafts.items(), key=lambda item: (TOPIC_ORDER.index(item[0][0]), item[0][1:])))


def summarize(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    result = data.get("agent_result", data)
    facts, rep = result["factual_results"], result["representation_results"]
    return {"status": result["status"], "stop_reason": result["stop_reason"],
            "revisions": Counter(r["status"] for r in result["revisions"]),
            "verdicts": facts.get("verdict_counts"), "representation": rep["status"],
            "disparity": rep.get("disparity"),
            "changed": result["current_summary"]["sentences"] != result["initial_summary"]["sentences"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "agent_rerun")
    parser.add_argument("--topic", choices=("all", *TOPIC_ORDER), default="all")
    parser.add_argument("--disparity-threshold", type=float, default=0.2)
    parser.add_argument("--max-revisions", type=int, default=3)
    args = parser.parse_args()

    drafts = saved_drafts()
    if len(drafts) != 18:
        parser.error(f"Expected 18 saved agent drafts, found {len(drafts)}.")
    args.out.mkdir(parents=True, exist_ok=True)
    failures = []
    for (topic, prompt_id, seed), draft in drafts.items():
        if args.topic not in ("all", topic):
            continue
        name = f"{topic}_prompt{prompt_id}_seed{seed}"
        output = args.out / f"{name}.json"
        if output.exists():
            print(f"Skipping {name}: already finished", flush=True)
            continue
        print(f"Running {name} from {draft.relative_to(ROOT)}", flush=True)
        command = [sys.executable, str(ROOT / "run_pipeline.py"), "--topic", topic, "--prompt-id", prompt_id,
                   "--reuse-draft", str(draft), "--disparity-threshold", str(args.disparity_threshold),
                   "--max-revisions", str(args.max_revisions), "--output", str(output)]
        with open(args.out / f"{name}.log", "w", encoding="utf-8") as log:
            code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        if code or not output.exists():
            failures.append(name)
            print(f"FAILED {name} (exit {code}); see {name}.log. Rerun to retry.", flush=True)

    rows = {path.stem: summarize(path) for path in sorted(args.out.glob("*_prompt*_seed*.json"))}
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for name, row in rows.items():
        print(f"{name:28} {row['status']:28} {row['stop_reason']:28} changed={row['changed']} "
              f"revisions={dict(row['revisions'])} verdicts={row['verdicts']} rep={row['representation']}")
    print(f"\n{len(rows)}/18 finished; {sum(r['changed'] for r in rows.values())} summaries changed by revision.")
    if failures:
        print("Failed: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
