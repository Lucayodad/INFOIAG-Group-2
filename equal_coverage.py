"""MNLI-based Equal Coverage for the planned dialect-fairness evaluation."""

from __future__ import annotations

from typing import Protocol, Sequence

from data_loader import DIALECT_LABELS


MODEL_NAME = "roberta-large-mnli"


class EntailmentScorer(Protocol):
    """Score premise/hypothesis pairs with MNLI entailment probabilities."""

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        ...


class MnliDependencyError(RuntimeError):
    """The local MNLI implementation cannot be loaded."""


class TransformersEntailmentScorer:
    """Lazy local RoBERTa-large-MNLI scorer.

    The model is loaded once per process and reused for all summaries. Model
    downloads are handled by Hugging Face's normal local cache.
    """

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 16):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise MnliDependencyError(
                "MNLI Equal Coverage requires torch and transformers. "
                "Install them with: pip install -r requirements.txt"
            ) from exc
        self._torch = torch
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        self._batch_size = batch_size
        self._entailment_index = self._find_entailment_index()

    def _find_entailment_index(self) -> int:
        labels = getattr(self._model.config, "id2label", {})
        for index, label in labels.items():
            if str(label).casefold() == "entailment":
                return int(index)
        raise MnliDependencyError(
            f"Model does not expose an entailment label: {labels!r}"
        )

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores = []
        with self._torch.inference_mode():
            for start in range(0, len(pairs), self._batch_size):
                batch = pairs[start:start + self._batch_size]
                premises, hypotheses = zip(*batch)
                encoded = self._tokenizer(
                    list(premises),
                    list(hypotheses),
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                probabilities = self._torch.softmax(
                    self._model(**encoded).logits, dim=-1
                )[:, self._entailment_index]
                scores.extend(float(value) for value in probabilities)
        return scores


_DEFAULT_SCORER: EntailmentScorer | None = None


def _default_scorer() -> EntailmentScorer:
    global _DEFAULT_SCORER
    if _DEFAULT_SCORER is None:
        _DEFAULT_SCORER = TransformersEntailmentScorer()
    return _DEFAULT_SCORER


class CachedScorer:
    """Score each (tweet, sentence) pair once per agent run.

    A revision changes one sentence, so a recheck needs 90 new pairs, not 450.
    Reusing the first scores also keeps unchanged pairs bit-identical across
    the checks whose coverages the acceptance gate compares.
    """

    def __init__(self, scorer: EntailmentScorer | None = None):
        self._scorer = scorer
        self.cache: dict[tuple[str, str], float] = {}

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        missing = [pair for pair in dict.fromkeys(pairs) if pair not in self.cache]
        if missing:
            scorer = self._scorer or _default_scorer()
            self.cache.update(zip(missing, _validate_scores(scorer.score_pairs(missing), len(missing))))
        return [self.cache[pair] for pair in pairs]


def _validate_scores(scores: Sequence[float], expected: int) -> list[float]:
    if len(scores) != expected:
        raise ValueError(f"MNLI scorer returned {len(scores)} scores; expected {expected}.")
    values = [float(score) for score in scores]
    if any(not 0.0 <= score <= 1.0 for score in values):
        raise ValueError("MNLI entailment probabilities must be in [0, 1].")
    return values


def calculate_equal_coverage(
    tweets_by_group: dict[str, list[dict]],
    summary_claims: list[dict],
    *,
    scorer: EntailmentScorer | None = None,
) -> dict:
    """Calculate group coverage and Equal Coverage from MNLI probabilities.

    Each source tweet is paired with every supplied summary sentence. Callers
    pass these as {"claim": sentence_text}; atomic claims remain for fact checking.
    Group coverage is the mean entailment probability over that group's tweets
    and claims. EC is the mean absolute deviation of group coverage from the
    overall tweet-level coverage, so it is zero when groups are equally covered.
    """
    if set(tweets_by_group) != set(DIALECT_LABELS):
        raise ValueError("tweets_by_group must contain AA, Hisp and White.")
    hypotheses = [
        claim.get("claim") for claim in summary_claims
        if isinstance(claim, dict) and isinstance(claim.get("claim"), str)
        and claim["claim"].strip()
    ]
    if not hypotheses:
        raise ValueError("At least one summary sentence is required.")
    if len(hypotheses) != len(summary_claims):
        raise ValueError("Every summary hypothesis must contain nonempty text.")
    if any(not tweets_by_group[label] for label in DIALECT_LABELS):
        raise ValueError("Every group requires source tweets; missing data is not zero coverage.")
    if any(not isinstance(tweet.get("text"), str) or not tweet["text"].strip()
           for rows in tweets_by_group.values() for tweet in rows):
        raise ValueError("Every source tweet requires nonempty text.")

    scorer = scorer or _default_scorer()
    group_scores = {}
    all_scores = []
    pair_counts = {}
    for label in DIALECT_LABELS:
        premises = [
            tweet["text"] for tweet in tweets_by_group[label]
            if isinstance(tweet.get("text"), str) and tweet["text"].strip()
        ]
        pairs = [(premise, hypothesis) for premise in premises for hypothesis in hypotheses]
        scores = _validate_scores(scorer.score_pairs(pairs), len(pairs))
        group_scores[label] = sum(scores) / len(scores) if scores else 0.0
        all_scores.extend(scores)
        pair_counts[label] = len(pairs)

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    ec = sum(abs(group_scores[label] - overall) for label in DIALECT_LABELS) / len(DIALECT_LABELS)
    gap = max(group_scores.values()) - min(group_scores.values())
    return {
        "metric": "mnli_equal_coverage",
        "model": MODEL_NAME,
        "coverage_by_group": {
            label: {
                "coverage": group_scores[label],
                "lower": group_scores[label],
                "upper": group_scores[label],
                "tweet_count": len(tweets_by_group[label]),
                "pair_count": pair_counts[label],
                "total_weight": len(tweets_by_group[label]),
            }
            for label in DIALECT_LABELS
        },
        "overall_coverage": overall,
        "equal_coverage": ec,
        "entailment_coverage_gap": gap,
        "hypothesis_unit": "summary_sentence",
        "metric_version": "mnli_sentence_mean_v1",
        "summary_claim_count": len(hypotheses),
        "summary_sentence_count": len(hypotheses),
        "pair_count": len(all_scores),
        "status": "measured",
    }
