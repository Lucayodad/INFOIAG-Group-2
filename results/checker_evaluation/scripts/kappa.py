"""Agreement of the two human annotators with each other and with the reference labels.

    python scripts/kappa.py annotation_sheet_60_NAME1.csv annotation_sheet_60_NAME2.csv

Each file is a filled copy of annotation_sheet_60.csv (column your_verdict: SUPPORTED,
CONTRADICTED or UNKNOWN). Prints raw agreement and Cohen's kappa for every pair.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
VERDICTS = ("SUPPORTED", "CONTRADICTED", "UNKNOWN")


def read(path, column):
    labels = {}
    for row in csv.DictReader(Path(path).open(encoding="utf-8-sig")):
        value = row[column].strip().upper()
        if value:
            if value not in VERDICTS:
                raise SystemExit(f"{path}: '{row[column]}' for {row['key']} is not one of {VERDICTS}")
            labels[row["key"]] = value
    return labels


def cohen_kappa(a, b):
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[v] * count_b[v] for v in VERDICTS) / len(a) ** 2
    return observed, (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sample = {row["key"]: row["reference_verdict"]
              for row in csv.DictReader((HERE / "reference_labels.csv").open(encoding="utf-8-sig"))
              if row["in_validation_sample"] == "yes"}
    raters = {"reference": sample,
              "annotator 1": read(sys.argv[1], "your_verdict"),
              "annotator 2": read(sys.argv[2], "your_verdict")}
    for name in ("annotator 1", "annotator 2"):
        missing = set(sample) - set(raters[name])
        if missing:
            print(f"{name}: {len(missing)} of {len(sample)} claims not labeled; they are left out")
    for first, second in (("annotator 1", "annotator 2"), ("annotator 1", "reference"), ("annotator 2", "reference")):
        keys = sorted(set(raters[first]) & set(raters[second]))
        observed, kappa = cohen_kappa([raters[first][k] for k in keys], [raters[second][k] for k in keys])
        print(f"{first} vs {second}: n = {len(keys)}, agreement = {observed:.2f}, Cohen's kappa = {kappa:.2f}")


if __name__ == "__main__":
    main()
