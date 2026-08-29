from __future__ import annotations

import hashlib
import heapq
from collections.abc import Iterable

from .filtering import RecordFilter
from .records import SFTRecord


def _score(record: SFTRecord, seed: int) -> int:
    value = f"{seed}:{record.source}:{record.source_id}:{record.content_key}"
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def select_records(
    records: Iterable[SFTRecord],
    *,
    quota: int,
    record_filter: RecordFilter,
    seed: int,
) -> list[SFTRecord]:
    if quota <= 0:
        return []
    heap: list[tuple[int, int, SFTRecord]] = []
    counter = 0
    local_seen: set[str] = set()
    for raw_record in records:
        record = record_filter.apply(raw_record)
        if record is None or record.content_key in local_seen:
            continue
        local_seen.add(record.content_key)
        score = _score(record, seed)
        entry = (-score, counter, record)
        counter += 1
        if len(heap) < quota:
            heapq.heappush(heap, entry)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    return [entry[2] for entry in sorted(heap, key=lambda item: (-item[0], item[1]))]
