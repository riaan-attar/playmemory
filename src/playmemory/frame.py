from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Frame:
    """One memory unit on the timeline."""

    id: int
    t: float
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Hit:
    """A search result: the matched frame plus its temporal neighbours."""

    frame: Frame
    score: float
    before: tuple[Frame, ...] = ()
    after: tuple[Frame, ...] = ()

    def context(self) -> list[Frame]:
        return [*self.before, self.frame, *self.after]
