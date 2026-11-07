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


@dataclass(frozen=True)
class Span:
    """A contiguous stretch of the timeline: "watch from start to end".

    start_id/end_id are inclusive frame ids (pointers an agent can re-read with
    Memory.read). `frames` holds only the frames visible under the query's
    as_of/forget rules. episode_len is the size of the scene it was cut from,
    so a caller can see when the span is partial.
    """

    start_id: int
    end_id: int
    t0: float
    t1: float
    frames: tuple[Frame, ...]
    score: float
    episode_id: int
    episode_len: int
    summary: str = ""

    @property
    def chars(self) -> int:
        return sum(len(f.text) for f in self.frames)

    def text(self) -> str:
        return "\n".join(f.text for f in self.frames)
