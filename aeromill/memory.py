"""Observation-only, versioned last successful absolute pair; cold by default."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryEntry:
    pair: tuple[float, float]
    verification_id: str
    time_s: float


class Memory:
    def __init__(self):
        self.entries: dict[tuple[str, str, str, str], MemoryEntry] = {}

    def get(self, key):
        return self.entries.get(key)

    def remember(self, key, pair, verification_id, time_s):
        self.entries[key] = MemoryEntry(tuple(pair), verification_id, time_s)

    def forget(self, key):
        self.entries.pop(key, None)
