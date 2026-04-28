from __future__ import annotations

from typing import Iterable


def detect_severe(prediction: object, threshold: float = 0.7) -> bool:
    """Detect severe cases using the combined major+destroyed probability."""
    probabilities = getattr(prediction, "probabilities", None)
    if not probabilities or len(probabilities) < 4:
        return False
    severe_score = probabilities[2] + probabilities[3]
    return severe_score >= threshold


def count_severe(predictions: Iterable[object], threshold: float = 0.7) -> int:
    return sum(1 for prediction in predictions if detect_severe(prediction, threshold))
