from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class Prediction:
    region: str
    disaster_type: str
    predicted_class: int
    probabilities: List[float]

    @property
    def severity_score(self) -> int:
        return self.predicted_class


def map_phase(
    predictions: Iterable[Prediction],
) -> Iterable[Tuple[Tuple[str, str], int]]:
    """Map phase: emit one key-value pair per prediction."""
    for prediction in predictions:
        key = (prediction.region, prediction.disaster_type)
        value = prediction.severity_score
        yield key, value


def shuffle_sort(
    mapped_data: Iterable[Tuple[Tuple[str, str], int]],
) -> Dict[Tuple[str, str], List[int]]:
    """Shuffle/sort phase: group severity scores by key."""
    grouped: Dict[Tuple[str, str], List[int]] = {}
    for key, value in mapped_data:
        grouped.setdefault(key, []).append(value)
    return grouped


def reduce_phase(
    grouped_data: Dict[Tuple[str, str], List[int]],
) -> Dict[Tuple[str, str], float]:
    """Reduce phase: compute average severity per region/disaster type."""
    result: Dict[Tuple[str, str], float] = {}
    for key, values in grouped_data.items():
        result[key] = sum(values) / len(values)
    return result


def compute_complexity() -> Dict[str, str]:
    return {
        "map": "O(N)",
        "shuffle": "O(N log N)",
        "reduce": "O(N)",
        "overall": "O(N log N)",
    }
