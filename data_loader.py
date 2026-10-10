"""Section 1: load the selected DivSumm inputs without modifying tweet text."""

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path


TOPICS = ("Obama", "Superbowl", "McDonalds")
DIALECT_LABELS = ("AA", "Hisp", "White")
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "divsumm" / "ready_to_use"


class DatasetError(ValueError):
    """An input file does not satisfy the expected dataset format."""


@dataclass(frozen=True)
class Tweet:
    """One source tweet; dialect_label is the dataset's label, not verified identity."""

    id: str
    topic: str
    dialect_label: str
    text: str
    source_file: str
    source_line: int


def load_tweets(topic: str, data_dir: Path = DEFAULT_DATA_DIR) -> list[Tweet]:
    """Read one topic's JSONL file, preserving its order, text, and source labels.

    Reference summaries are deliberately not read by this function.
    Structural validation happens here; expected dataset counts are checked
    separately by validate_dataset().
    """
    if topic not in TOPICS:
        raise DatasetError(f"Unknown topic {topic!r}. Choose one of {TOPICS}.")

    path = Path(data_dir) / f"{topic}.jsonl"
    tweets = []
    seen_ids = set()
    seen_sources = set()

    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                location = f"{path.name}, line {line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"{location}: invalid JSON: {exc.msg}") from exc

                if not isinstance(row, dict):
                    raise DatasetError(f"{location}: expected a JSON object.")
                for field in ("id", "topic", "dialect_label", "text", "source_file"):
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        raise DatasetError(f"{location}: {field} must be a nonempty string.")
                if type(row.get("source_line")) is not int or row["source_line"] < 1:
                    raise DatasetError(f"{location}: source_line must be a positive integer.")
                if row["topic"] != topic:
                    raise DatasetError(f"{location}: expected topic {topic!r}, got {row['topic']!r}.")
                if row["dialect_label"] not in DIALECT_LABELS:
                    raise DatasetError(f"{location}: unrecognized dialect label {row['dialect_label']!r}.")
                if row["id"] in seen_ids:
                    raise DatasetError(f"{location}: duplicate tweet ID {row['id']!r}.")
                source = (row["source_file"], row["source_line"])
                if source in seen_sources:
                    raise DatasetError(f"{location}: duplicate source reference {source!r}.")

                seen_ids.add(row["id"])
                seen_sources.add(source)
                tweets.append(Tweet(**{name: row[name] for name in Tweet.__dataclass_fields__}))
    except (OSError, UnicodeError) as exc:
        raise DatasetError(f"Cannot read UTF-8 dataset {path}: {exc}") from exc

    if not tweets:
        raise DatasetError(f"{path}: no tweets were found.")
    return tweets


def validate_dataset(tweets: list[Tweet]) -> dict[str, int]:
    """Check the expected 90 tweets, 30 per dialect, for a selected DivSumm topic.

    This checks file integrity, not whether a summary is fair or accurate.
    """
    if len({tweet.topic for tweet in tweets}) != 1:
        raise DatasetError("Expected exactly one topic.")
    if len({tweet.id for tweet in tweets}) != len(tweets):
        raise DatasetError("Duplicate tweet IDs.")
    counts = Counter(tweet.dialect_label for tweet in tweets)
    expected = {label: 30 for label in DIALECT_LABELS}
    if len(tweets) != 90 or counts != expected:
        raise DatasetError(
            f"Expected 90 tweets and {expected}; found {len(tweets)} tweets and {dict(counts)}. "
            "Check the input files; do not silently discard or rebalance rows."
        )
    return {label: counts[label] for label in DIALECT_LABELS}
