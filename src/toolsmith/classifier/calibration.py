"""Judge calibration against hand-labelled traces.

The optimizer inherits whatever the judge gets wrong, so the judge is measured
before it is trusted. Cohen's kappa is the right statistic here rather than raw
agreement: the class distribution is skewed, and two labellers who both guess
the majority class would look accurate by agreement alone.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .taxonomy import KEYS


@dataclass
class Calibration:
    n: int
    agreement: float
    kappa: float
    per_class: dict[str, dict[str, float]]
    confusion: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "agreement": round(self.agreement, 4),
            "cohens_kappa": round(self.kappa, 4),
            "per_class": self.per_class,
            "confusion": self.confusion,
        }


def cohens_kappa(human: list[str], judge: list[str], labels: tuple[str, ...] = KEYS) -> float:
    if len(human) != len(judge):
        raise ValueError("label lists must be the same length")
    n = len(human)
    if n == 0:
        return 0.0
    observed = sum(1 for a, b in zip(human, judge, strict=True) if a == b) / n
    human_counts = Counter(human)
    judge_counts = Counter(judge)
    expected = sum(
        (human_counts[label] / n) * (judge_counts[label] / n) for label in labels
    )
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def _per_class(human: list[str], judge: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for label in KEYS:
        tp = sum(1 for a, b in zip(human, judge, strict=True) if a == label and b == label)
        fp = sum(1 for a, b in zip(human, judge, strict=True) if a != label and b == label)
        fn = sum(1 for a, b in zip(human, judge, strict=True) if a == label and b != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        stats[label] = {
            "support": tp + fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return stats


def _confusion(human: list[str], judge: list[str]) -> dict[str, dict[str, int]]:
    matrix = {row: {column: 0 for column in KEYS} for row in KEYS}
    for actual, predicted in zip(human, judge, strict=True):
        matrix[actual][predicted] += 1
    return matrix


def evaluate(pairs: list[tuple[str, str]]) -> Calibration:
    human = [a for a, _ in pairs]
    judge = [b for _, b in pairs]
    n = len(pairs)
    agreement = sum(1 for a, b in pairs if a == b) / n if n else 0.0
    return Calibration(
        n=n,
        agreement=agreement,
        kappa=cohens_kappa(human, judge),
        per_class=_per_class(human, judge),
        confusion=_confusion(human, judge),
    )


def load_labelled(path: str | Path) -> list[tuple[str, str]]:
    """Read the hand-labelled calibration set."""
    pairs = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        pairs.append((record["human_label"], record["judge_label"]))
    return pairs
