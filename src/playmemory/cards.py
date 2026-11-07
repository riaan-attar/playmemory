"""Structured memory cards on top of the frame timeline.

A card is an atomic fact/preference/event... stored as an ordinary frame with
meta {"kind": "card", ...}. Exclusive slots (e.g. lives_in) supersede the old
value automatically, so current search sees only the latest while
`as_of=` still answers "what did we believe then?".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol

from .frame import Frame
from .memory import Memory

KINDS = ("fact", "preference", "event", "profile", "relationship", "goal", "other")
_EXCLUSIVE = {"fact", "profile", "relationship"}


@dataclass(frozen=True)
class Card:
    kind: str
    subject: str
    predicate: str
    value: str
    exclusive: bool | None = None  # default: by kind

    @property
    def is_exclusive(self) -> bool:
        return self.kind in _EXCLUSIVE if self.exclusive is None else self.exclusive

    @property
    def key(self) -> str:
        base = f"{self.subject.lower()}|{self.kind}|{self.predicate}"
        return base if self.is_exclusive else f"{base}|{self.value.lower()}"

    @property
    def text(self) -> str:
        return f"{self.subject} {self.predicate.replace('_', ' ')} {self.value}"


class Extractor(Protocol):
    def extract(self, text: str) -> list[Card]: ...


_RULES: list[tuple[re.Pattern, Callable[[re.Match], Card]]] = [
    (re.compile(r"\bmy name is ([A-Za-z][\w'-]*)", re.I),
     lambda m: Card("profile", "user", "name", m[1])),
    (re.compile(r"\b(?i:i (?:live|am living|reside)s? in) ([A-Z][\w'-]*(?: [A-Z][\w'-]*)*)"),
     lambda m: Card("fact", "user", "lives_in", m[1])),
    (re.compile(r"\b(?i:i (?:moved|am moving|relocated) to) ([A-Z][\w'-]*(?: [A-Z][\w'-]*)*)"),
     lambda m: Card("fact", "user", "lives_in", m[1])),
    (re.compile(r"\bi (?:work|am working|worked) (?:at|for) ([\w&-]+(?: [\w&-]+)?)", re.I),
     lambda m: Card("fact", "user", "works_at", m[1].strip().rstrip(".,;"))),
    (re.compile(r"\bi(?:'m| am) (a vegetarian|vegetarian|vegan|allergic to [\w ]+?)(?=[.,;!?]|$| and | but )", re.I),
     lambda m: Card("profile", "user", "diet", m[1].replace("a ", "", 1).strip())),
    (re.compile(r"\bi (?:like|love|enjoy|prefer) ([^.,;!?]+)", re.I),
     lambda m: Card("preference", "user", "likes", m[1].strip())),
    (re.compile(r"\bi (?:dislike|hate|can't stand) ([^.,;!?]+)", re.I),
     lambda m: Card("preference", "user", "dislikes", m[1].strip())),
    (re.compile(r"\bi (?:want|plan|hope) to ([^.,;!?]+)", re.I),
     lambda m: Card("goal", "user", "wants_to", m[1].strip())),
    (re.compile(r"\bmy (manager|boss|wife|husband|partner|brother|sister|friend) is ([A-Z][\w'-]*)"),
     lambda m: Card("relationship", "user", f"{m[1]}_is", m[2])),
    (re.compile(r"\b([A-Z][a-z]+) (?:lives|is living) in ([A-Z][\w'-]*(?: [A-Z][\w'-]*)*)"),
     lambda m: Card("fact", m[1], "lives_in", m[2])),
]


class RuleExtractor:
    """Dependency-free English baseline. Use LLMExtractor for real coverage."""

    def extract(self, text: str) -> list[Card]:
        cards, seen = [], set()
        for pat, build in _RULES:
            for m in pat.finditer(text):
                c = build(m)
                if (c.key, c.value) not in seen and c.value:
                    seen.add((c.key, c.value))
                    cards.append(c)
        return cards


class LLMExtractor:
    """Wrap any `fn(text) -> list[dict]` (e.g. an LLM call returning JSON rows
    with kind/subject/predicate/value[/exclusive])."""

    def __init__(self, fn: Callable[[str], list[dict]]):
        self.fn = fn

    def extract(self, text: str) -> list[Card]:
        out = []
        for row in self.fn(text):
            kind = row.get("kind", "other")
            out.append(Card(kind if kind in KINDS else "other", str(row["subject"]),
                            str(row["predicate"]), str(row["value"]), row.get("exclusive")))
        return out


def remember(mem: Memory, text: str, extractor: Extractor | None = None,
             t: float | None = None, meta: dict | None = None, schema=None) -> tuple[Frame, list[Frame]]:
    """Store `text` as a raw frame, then extract cards from it. Returns
    (raw frame, card frames added). With `schema` (playmemory.graph.Schema) cards are validated
    and cardinality comes from it. Exclusive slots supersede older values;
    repeating an identical card is a no-op."""
    extractor = extractor or RuleExtractor()
    raw = mem.add(text, {**(meta or {}), "kind": "raw"}, t)
    added: list[Frame] = []
    for card in extractor.extract(text):
        if schema is not None:
            card = schema.validate(card)  # raises SchemaError; sets one-vs-many exclusivity
        current = mem.find({"kind": "card", "card_key": card.key})
        cmeta = {"kind": "card", "card_key": card.key, "card_kind": card.kind,
                 "subject": card.subject, "predicate": card.predicate,
                 "value": card.value, "source": raw.id}
        if current and current[-1].meta["value"] == card.value:
            continue  # same belief already stored
        if current and card.is_exclusive:
            added.append(mem.supersede(current[-1].id, card.text, cmeta, t=raw.t))
        else:
            added.append(mem.add(card.text, cmeta, raw.t))
    return raw, added


def profile(mem: Memory, subject: str = "user", as_of: float | None = None) -> dict[str, list[str]]:
    """Current beliefs about `subject`, grouped by predicate (time-travel via as_of)."""
    out: dict[str, list[str]] = {}
    for f in mem.find({"kind": "card", "subject": subject}, as_of=as_of):
        out.setdefault(f.meta["predicate"], []).append(f.meta["value"])
    return out
