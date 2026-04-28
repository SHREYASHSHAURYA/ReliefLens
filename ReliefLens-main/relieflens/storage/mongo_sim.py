from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


class MongoSim:
    def __init__(self) -> None:
        self.collections: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def insert(self, collection: str, document: Dict[str, Any]) -> None:
        self.collections[collection].append(document.copy())

    def increment_counter(self, collection: str, key: str, amount: int = 1) -> None:
        self.counters[collection][key] += amount

    def count_total(self, collection: str) -> int:
        return len(self.collections[collection])

    def count_by_field(self, collection: str, field: str) -> Dict[Any, int]:
        counts: Dict[Any, int] = {}
        for document in self.collections[collection]:
            value = document.get(field)
            counts[value] = counts.get(value, 0) + 1
        return counts

    def find_all(self, collection: str) -> List[Dict[str, Any]]:
        return [doc.copy() for doc in self.collections[collection]]
