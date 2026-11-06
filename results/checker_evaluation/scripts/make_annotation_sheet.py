"""Write the blind sheet for the two human annotators: the 60 sampled claims and their cited tweets.

The reference verdict, its confidence and the uncited supporting tweets are left out, so the
annotators label without seeing the reference. Run from results/checker_evaluation:

    python scripts/make_annotation_sheet.py
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
rows = [row for row in csv.DictReader((HERE / "reference_labels.csv").open(encoding="utf-8-sig"))
        if row["in_validation_sample"] == "yes"]
out = HERE / "annotation_sheet_60.csv"
with out.open("w", encoding="utf-8-sig", newline="") as handle:  # BOM so Excel reads UTF-8
    writer = csv.writer(handle)
    writer.writerow(["nr", "key", "claim", "cited_tweets", "your_verdict", "comment"])
    for number, row in enumerate(rows, 1):
        writer.writerow([number, row["key"], row["claim"], row["cited_tweets"], "", ""])
print(f"{len(rows)} claims -> {out}")
