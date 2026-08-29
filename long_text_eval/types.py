from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LongDocument:
    document_id: str
    category: str
    source: str
    target: str
    sentence_count: int
    layout: str
    length_tier: str
    must_keep: tuple[str, ...] = field(default_factory=tuple)
    tail_keep: tuple[str, ...] = field(default_factory=tuple)
    provenance: str = ""
