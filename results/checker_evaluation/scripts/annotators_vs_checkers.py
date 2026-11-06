"""The 60-claim validation sample: each annotator against the reference labels and both fact checkers.

    python scripts/annotators_vs_checkers.py annotation_sheet_60_A.csv annotation_sheet_60_B.csv

Prints agreement and Cohen's kappa for every pair of raters, the verdict counts, and the accuracy of
the original and the revised fact checker on the claims where both annotators agree. The revised
checker's verdicts are those of the final run (results/agent_checker_v2, first check of each draft).
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from kappa import VERDICTS, cohen_kappa, read

HERE = Path(__file__).resolve().parent.parent
RUN = HERE.parent / "agent_checker_v2"


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sample = {row["key"]: row["reference_verdict"]
              for row in csv.DictReader((HERE / "reference_labels.csv").open(encoding="utf-8-sig"))
              if row["in_validation_sample"] == "yes"}
    original = {c["key"]: c["baseline"] for c in json.loads((HERE / "scripts" / "claims.json").read_text(encoding="utf-8"))}
    revised = {}
    for path in RUN.glob("*_seed*.json"):
        first = json.loads(path.read_text(encoding="utf-8"))["iterations"][0]
        revised.update({f"{path.stem}|{a['summary_claim_id']}": a["verdict"] for a in first["factual_results"]["assessments"]})
    raters = {"annotator 1": read(sys.argv[1], "your_verdict"), "annotator 2": read(sys.argv[2], "your_verdict"),
              "reference": sample,
              "original checker": {k: original[k] for k in sample}, "revised checker": {k: revised[k] for k in sample}}
    for name, labels in raters.items():
        print(f"{name:17} n = {len(labels)}  " + "  ".join(f"{v} {Counter(labels.values())[v]}" for v in VERDICTS))
    print()
    names = list(raters)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            keys = sorted(set(raters[first]) & set(raters[second]))
            observed, kappa = cohen_kappa([raters[first][k] for k in keys], [raters[second][k] for k in keys])
            print(f"{first:17} vs {second:17} n = {len(keys)}  agreement = {observed:.2f}  kappa = {kappa:.2f}")
    agreed = {k: v for k, v in raters["annotator 1"].items() if raters["annotator 2"].get(k) == v}
    # McDonalds was the checker's development topic; Obama and Superbowl were held out.
    for label, keys in (("all topics", set(agreed)), ("held-out topics (Obama, Superbowl)",
                                                     {k for k in agreed if not k.startswith("McDonalds")})):
        print(f"\n{label}: both annotators agree on {len(keys)} claims")
        for name in ("reference", "original checker", "revised checker"):
            correct = sum(raters[name][k] == agreed[k] for k in keys)
            print(f"  {name:17} matches their label on {correct} of {len(keys)} ({correct / len(keys):.2f})")


if __name__ == "__main__":
    main()
