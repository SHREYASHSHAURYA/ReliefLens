from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

CQL_SCHEMA = """
CREATE KEYSPACE ReliefLens
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

CREATE TABLE DamageMonitoring (
region TEXT,
disaster_type TEXT,
avg_severity FLOAT,
timestamp TIMESTAMP,
PRIMARY KEY (region, disaster_type)
);
"""


@dataclass
class CassandraSim:
    rows: Dict[tuple[str, str], Dict[str, Any]] = field(default_factory=dict)

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            key
            for key, row in self.rows.items()
            if row.get("expires_at") and row["expires_at"] <= now
        ]
        for key in expired:
            del self.rows[key]

    def insert_batch(self, records: Iterable[Dict[str, Any]]) -> None:
        self._cleanup_expired()
        for record in records:
            key = (record["region"], record["disaster_type"])
            self.rows[key] = deepcopy(record)
            self.rows[key].pop("expires_at", None)

    def update(self, region: str, disaster_type: str, new_value: float) -> None:
        self._cleanup_expired()
        key = (region, disaster_type)
        if key in self.rows:
            self.rows[key]["avg_severity"] = new_value

    def add_column(self, column_name: str) -> None:
        self._cleanup_expired()
        for row in self.rows.values():
            if column_name not in row:
                row[column_name] = None

    def insert_with_ttl(self, record: Dict[str, Any], ttl_seconds: int) -> None:
        self._cleanup_expired()
        key = (record["region"], record["disaster_type"])
        stored = deepcopy(record)
        stored["expires_at"] = time.time() + ttl_seconds
        self.rows[key] = stored

    def query_all(self) -> List[Dict[str, Any]]:
        self._cleanup_expired()
        return [deepcopy(row) for row in self.rows.values()]
