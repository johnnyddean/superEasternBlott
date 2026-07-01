from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Variant:
    variant_id: str
    sequence: str
    parent_id: str = "sfGFP"
    source: str = "generated"
    mutations: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "sequence": self.sequence,
            "parent_id": self.parent_id,
            "source": self.source,
            "mutations": list(self.mutations),
            **self.metadata,
        }
