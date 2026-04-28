from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def top_k_regions(predictions: Iterable[object], k: int = 10) -> List[Tuple[str, int]]:
    """Count severe predictions and return top K regions."""
    counts: Dict[str, int] = {}
    for prediction in predictions:
        severity_score = getattr(prediction, "predicted_class", None)
        if severity_score is None:
            severity_score = getattr(prediction, "severity_score", None)
        if severity_score is None:
            continue
        if severity_score >= 2:
            counts[prediction.region] = counts.get(prediction.region, 0) + 1
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[:k]
