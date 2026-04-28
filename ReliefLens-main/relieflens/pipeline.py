from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List

from relieflens.analytics.threshold import count_severe
from relieflens.analytics.topk import top_k_regions
from relieflens.data_processing.mapreduce import map_phase, reduce_phase, shuffle_sort
from relieflens.data_processing.spark_sim import error_analysis
from relieflens.exports.csv_io import export_csv
from relieflens.storage.cassandra_sim import CassandraSim


@dataclass
class Prediction:
    region: str
    disaster_type: str
    predicted_class: int
    probabilities: List[float]

    @property
    def severity_score(self) -> int:
        return self.predicted_class


def run_analytics(
    predictions: Iterable[Prediction], logs: Iterable[Dict[str, List[str]]]
) -> Dict[str, Any]:
    """Run the ReliefLens analytics pipeline locally."""
    mapped = list(map_phase(predictions))
    grouped = shuffle_sort(mapped)
    avg_severity = reduce_phase(grouped)

    hotspots = top_k_regions(predictions)
    severe_cases = count_severe(predictions)
    error_patterns = error_analysis(logs)

    cassandra = CassandraSim()
    records = [
        {
            "region": key[0],
            "disaster_type": key[1],
            "avg_severity": value,
            "timestamp": datetime.utcnow().isoformat(),
        }
        for key, value in avg_severity.items()
    ]
    cassandra.insert_batch(records)

    export_csv(records, "relieflens_avg_severity.csv")

    return {
        "avg_severity": avg_severity,
        "top_regions": hotspots,
        "severe_cases": severe_cases,
        "error_patterns": error_patterns,
        "cassandra_snapshot": cassandra.query_all(),
        "export_file": "relieflens_avg_severity.csv",
    }
