"""Rerun the 18 agent trials label-free, reusing each recorded first check.

Dialect labels are evaluation data, so the agent here never sees one: every
dialectLabel triple is removed from its in-memory copy of the run graph (the
file is unchanged), it has no group check (it revises only contradicted
claims), and LabelFreeClient refuses any Qwen or embedding request that
carries a label. Labels are joined back only by the final evaluation.

The draft and graph of each recorded run in results/agent_rerun are reused.
Its first check seeds the agent's cache: claim extraction and fact verdicts
run at temperature 0, never saw a label (the fact checker's evidence view
drops it) and repeated identically in every recorded recheck (133/133).
results/agent_final holds the same first check with failed extractions
redone; it is preferred when present. Fact verdicts are reused only when
the current checker made them (fact_check.CHECKER_VERSION): after the
2026-11-05 checker revision every claim is checked anew, while the claim
extraction is still reused. Finished trials are skipped, so rerunning the
same command resumes.

    python run_final_agent.py --out results/agent_compliant [--trial Obama_prompt1]

With --viewpoint-check the agent also repairs a viewpoint of the tweets that
the summary omits (stances of the extracted tweet claims, no label):

    python run_final_agent.py --viewpoint-check --out results/agent_viewpoint

With --proportion-check one feedback layer is added on top: Qwen reads every
tweet of the topic and the summary, judges whether the summary is in
proportion to all the tweets, and an under-represented viewpoint it names is
repaired (no label is read or named):

    python run_final_agent.py --proportion-check --out results/agent_proportion
"""

import argparse
import json
from pathlib import Path
import re
import time
import traceback

from rdflib import Graph

from fact_check import CHECKER_VERSION
from ollama_backend import OllamaClient
from ontology_tools import NS
from rerun_agent_trials import summarize
from summary_agent import print_run_summary, run_summary_agent, seed_check_cache

ROOT = Path(__file__).resolve().parent
RECORDED = ROOT / "results" / "agent_rerun"
FIRST_CHECKS = ROOT / "results" / "agent_final"
LABEL_LEAK = re.compile(r'(?i:dialect)|"(?:AA|Hisp|White)"')


class LabelFreeClient:
    """Refuses every request carrying a dialect label: compliance by check, not trust."""

    def __init__(self, client):
        self._client = client
        self.checked_requests = 0

    def _refuse_labels(self, request):
        self.checked_requests += 1
        if LABEL_LEAK.search(json.dumps(request, ensure_ascii=False)):
            raise RuntimeError("A dialect label reached a model request; the run is stopped.")

    def chat(self, payload):
        self._refuse_labels(payload)
        return self._client.chat(payload)

    def embed(self, texts, **options):
        self._refuse_labels(texts)
        return self._client.embed(texts, **options)

    def __getattr__(self, name):
        return getattr(self._client, name)


def first_check(iteration):
    """The claims and fact verdicts of a recorded check, without stored dialect_label fields.

    The recorded group check is dropped: only extraction and facts are reused.
    """
    return without_labels({key: iteration[key] for key in ("summary_claims", "factual_results")})


def without_labels(value):
    """A recorded value without its stored dialect_label fields (never shown to Qwen)."""
    if isinstance(value, dict):
        return {key: without_labels(item) for key, item in value.items() if key != "dialect_label"}
    if isinstance(value, list):
        return [without_labels(item) for item in value]
    return value


def label_free_graph(path):
    """The run graph without its dialectLabel triples (the file is not modified)."""
    graph = Graph().parse(path, format="turtle")
    graph.remove((None, NS.dialectLabel, None))
    if any(True for _ in graph.triples((None, NS.dialectLabel, None))):
        raise RuntimeError("dialectLabel triples remain in the agent's graph.")
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "agent_compliant")
    parser.add_argument("--trial", nargs="*", default=[], help="Only trials whose name contains one of these")
    parser.add_argument("--max-revisions", type=int, default=3)
    parser.add_argument("--first-checks", type=Path, default=FIRST_CHECKS,
                        help="Finished runs whose first check of each draft seeds the cache "
                             "(fact verdicts only from the current checker)")
    parser.add_argument("--viewpoint-check", action="store_true",
                        help="Also repair an omitted viewpoint (label-free reflection on bias)")
    parser.add_argument("--proportion-check", action="store_true",
                        help="Add the feedback layer checking the summary is in proportion to all tweets")
    args = parser.parse_args()

    names = sorted(path.stem for path in RECORDED.glob("*_prompt*_seed*.json"))
    if len(names) != 18:
        parser.error(f"Expected 18 recorded trials in {RECORDED}, found {len(names)}.")
    if args.trial:
        names = [name for name in names if any(part in name for part in args.trial)]
    args.out.mkdir(parents=True, exist_ok=True)
    backend = OllamaClient()
    failures = []
    for name in names:
        output = args.out / f"{name}.json"
        if output.exists():
            print(f"Skipping {name}: already finished", flush=True)
            continue
        recorded = json.loads((RECORDED / f"{name}.json").read_text(encoding="utf-8"))
        recorded = recorded.get("agent_result", recorded)
        first = args.first_checks / f"{name}.json"
        checked = first_check(json.loads(first.read_text(encoding="utf-8"))["iterations"][0] if first.exists()
                              else recorded["iterations"][0])
        graph_path = Path(recorded["pipeline"]["graph"].replace("\\", "/"))  # recorded on Windows
        graph_path = graph_path if graph_path.is_absolute() else ROOT / graph_path
        print(f"Running {name} at {time.strftime('%H:%M')}", flush=True)
        started = time.time()
        client = LabelFreeClient(backend)
        try:
            result = run_summary_agent(
                label_free_graph(graph_path), recorded["initial_summary"],
                topic=recorded["topic"], prompt=recorded["prompt"], client=client,
                max_revisions=args.max_revisions, representation_metric=None,
                viewpoint_check=args.viewpoint_check, proportion_check=args.proportion_check,
                check_cache=seed_check_cache(recorded["initial_summary"], checked))
        except Exception:  # noqa: BLE001 - record and continue with the next trial
            traceback.print_exc()
            failures.append(name)
            print(f"FAILED {name}; rerun the same command to retry it.", flush=True)
            continue
        # evaluate.py identifies a trial by topic, prompt and seed.
        result["prompt_id"], result["seed"] = name.split("_prompt")[1].split("_seed")[0], int(name.rsplit("_seed", 1)[1])
        result["provenance"] = {
            "recorded_run": str((RECORDED / f"{name}.json").relative_to(ROOT)),
            "graph": str(graph_path.relative_to(ROOT)),
            "first_check": str((first if first.exists() else RECORDED / f"{name}.json").relative_to(ROOT)),
            "first_check_note": "claim extraction reused (label-free prompts), unclear extractions redone; "
                                "fact verdicts reused only from the same checker version, otherwise checked "
                                "anew; revisions and rechecks computed in this run",
            "checker_version": CHECKER_VERSION,
            "dialect_labels": "removed from the agent's graph; no group check; every model request "
                              f"checked for labels ({client.checked_requests} checked, none found)",
            "viewpoint_check": args.viewpoint_check,
            "proportion_check": args.proportion_check,
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
