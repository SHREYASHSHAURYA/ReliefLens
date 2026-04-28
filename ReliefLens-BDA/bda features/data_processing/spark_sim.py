from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List


def error_analysis(logs: Iterable[Dict[str, List[str]]]) -> Dict[str, int]:
    """Simulate Spark flatMap, map, reduceByKey, and filter."""
    # flatMap: extract all errors from every log entry
    errors: List[str] = []
    for log in logs:
        errors.extend(log.get("errors", []))

    # map: convert each error into a count of 1
    mapped = [(error, 1) for error in errors]

    # reduceByKey: count occurrences per error code
    counts: Counter[str] = Counter()
    for error, value in mapped:
        counts[error] += value

    # filter: keep only errors appearing more than 10 times
    return {error: count for error, count in counts.items() if count > 10}
