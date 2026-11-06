"""Write the blind sheet for the two annotators: the 60 sampled claims and the full text of their cited tweets.

The reference verdict, its confidence and the uncited supporting tweets are left out, so the
annotators label without seeing the reference. Every cited tweet is written out as
"<tweet id>: <text>" (HTML entities decoded), one per line, also for claims citing dozens of tweets (reference_labels.csv
abbreviates those to their IDs, which cannot be judged). Run from results/checker_evaluation:

    python scripts/make_annotation_sheet.py
"""
import csv
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE.parent.parent / "data" / "divsumm" / "ready_to_use"

texts = {}
for topic in ("McDonalds", "Obama", "Superbowl"):
    for line in (DATA / f"{topic}.jsonl").open(encoding="utf-8"):
        if line.strip():
            tweet = json.loads(line)
            texts[tweet["id"]] = html.unescape(tweet["text"])  # readable: "&amp;" shown as "&"
cited = {c["key"]: c["cited"] for c in json.loads((HERE / "scripts" / "claims.json").read_text(encoding="utf-8"))}
rows = [row for row in csv.DictReader((HERE / "reference_labels.csv").open(encoding="utf-8-sig"))
        if row["in_validation_sample"] == "yes"]
out = HERE / "annotation_sheet_60.csv"
with out.open("w", encoding="utf-8-sig", newline="") as handle:  # BOM so Excel reads UTF-8
    writer = csv.writer(handle)
    writer.writerow(["nr", "key", "claim", "cited_tweets", "your_verdict", "comment"])
    for number, row in enumerate(rows, 1):
        tweets = "\n".join(f"{key}: {texts[key]}" for key in dict.fromkeys(cited[row["key"]]))
        writer.writerow([number, row["key"], row["claim"], tweets, "", ""])
print(f"{len(rows)} claims -> {out}")
