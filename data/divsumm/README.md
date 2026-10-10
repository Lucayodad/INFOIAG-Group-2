# DivSumm: Obama, Super Bowl, and McDonald's

This package contains all input tweets and human reference summaries for the three selected topics in your uploaded DivSumm-main.zip.

| Topic/file name | Input tweets | AA | Hisp | White | Abstractive references | Extractive references |
|---|---:|---:|---:|---:|---:|---:|
| Obama | 90 | 30 | 30 | 30 | 2 | 2 |
| Superbowl | 90 | 30 | 30 | 30 | 2 | 2 |
| McDonalds | 90 | 30 | 30 | 30 | 2 | 2 |
| Total | 270 | 90 | 90 | 90 | 6 | 6 |

## What to use

- `ready_to_use/Obama.jsonl`, `Superbowl.jsonl`, and `McDonalds.jsonl`: recommended inputs for your Python agent. One JSON object per tweet, in original order, with dialect labels preserved.
- `ready_to_use/all_topics.jsonl`: the same 270 records combined for convenience. Group by `topic` to run the three datasets separately; this is not an additional dataset.
- `ready_to_use/references.json`: both annotators' abstractive and extractive summaries, organized by topic. Keep these separate from agent inputs and use them for later evaluation.
- `utf8/input_docs/`: UTF-8 text copies in the original label-tab-tweet layout.
- `utf8/abstractive/` and `utf8/extractive/`: UTF-8 reference copies.
- `original/`: the 15 selected data/reference files and original repository README, preserved byte-for-byte.
- `manifest.json`: original-file checksums, counts, provenance, and decoding details.

## Labels and fields

The source README describes three dialect categories: African-American (`AA`), Hispanic (`Hisp`), and White (`White`). These are dataset-provided dialect labels, not independently verified race or ethnicity of tweet authors. The labels are already attached to individual tweets; there is no need to infer them from the text. No separate label file is included in the supplied archive.

JSONL fields: `id`, `topic`, `dialect_label`, `text`, `source_file`, and `source_line`. The `id` field is generated from topic and original line number; it is not a Twitter ID. No author identifiers, tweet timestamps, train/validation/test splits, or original Twitter IDs were added.

## Loading in Python

Run from this package's root folder:

```python
import json
from pathlib import Path

topic = "Obama"  # or "Superbowl" or "McDonalds"
path = Path("ready_to_use") / f"{topic}.jsonl"
with path.open(encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

texts = [row["text"] for row in records]
groups = [row["dialect_label"] for row in records]

# Reference summaries are for evaluation, not generation:
with Path("ready_to_use/references.json").open(encoding="utf-8") as f:
    reference_summaries = json.load(f)[topic]
```

## Text preservation and encoding

No tweets were removed or rewritten. Spelling, dialect, offensive wording, quoted content, HTML entities such as `&amp;`, and literal backslash-n sequences are preserved. Actual file line endings are normalized only in the UTF-8 convenience copies.

Some files in the uploaded archive are not valid UTF-8. For convenience copies, valid UTF-8 is decoded directly; other files are decoded as Mac Roman, inferred from consistent smart-quote, apostrophe, ellipsis, and accented-character byte patterns. This encoding is not declared in the repository README. The original bytes and checksums are included so this conversion can be audited or changed. In Obama.txt, the legacy byte 0xF0 becomes the Mac Roman private-use Apple glyph U+F8FF; it is retained rather than silently removed.

## Project use

Use each topic's 90 tweets as one input dataset and apply your two prompts to it. This gives six topic/prompt combinations. Preserve the dialect labels for fair-representation analysis. The reference files are human-authored summaries, not independently verified factual ground truth or guaranteed unbiased targets. These historical tweets contain opinions and claims; do not treat them as verified current facts for the ontology.

Your ontology export and the two prompts from Intermediate Report 1 are project-specific and are not part of this repository package.

## Source and citation

Repository: https://github.com/PortNLP/DivSumm

Paper: https://aclanthology.org/2022.coling-1.542/

Olabisi, Olubusayo; Hudson, Aaron; Jetter, Antonie; Agrawal, Ameeta. "Analyzing the Dialect Diversity in Multi-document Summaries." COLING 2022.

```bibtex
@inproceedings{olabisi-etal-2022-analyzing,
    title = "Analyzing the Dialect Diversity in Multi-document Summaries",
    author = "Olabisi, Olubusayo and Hudson, Aaron and Jetter, Antonie and Agrawal, Ameeta",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022"
}
```

The original repository README is included at `original/README.md`. No standalone license file was present in the uploaded archive, and this package does not add a license. The supplied ZIP does not establish a Git commit identifier; its SHA-256 is recorded instead.
