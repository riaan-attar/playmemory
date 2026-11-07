"""Entity graph over memory cards, with a predicate schema.

Nothing new is stored: the graph is a time-aware *view* over frames whose meta has
kind="card" (see cards.remember). That keeps it consistent with supersede/forget and
lets every query take `as_of=`. Nodes are normalised entity names (subjects and the
values of entity-valued cards); edges are (subject, predicate, object) triples that
remember the frame they came from.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field, replace

from .cards import Card
from .memory import Memory


class SchemaError(ValueError):
    pass


_TYPES = {
    "string": lambda v: True,
    "int": lambda v: re.fullmatch(r"[+-]?\d+", v.strip()) is not None,
    "float": lambda v: re.fullmatch(r"[+-]?\d+(\.\d+)?", v.strip()) is not None,
    "date": lambda v: re.fullmatch(r"\d{4}-\d{2}-\d{2}", v.strip()) is not None,
    "bool": lambda v: v.strip().lower() in ("true", "false", "yes", "no"),
    "entity": lambda v: bool(v.strip()),
}


@dataclass(frozen=True)
class Predicate:
    name: str
    cardinality: str = "one"  # "one": a new value supersedes the old; "many": values accumulate
    value_type: str = "string"  # string | int | float | date | bool | entity
    domain: frozenset = frozenset()  # allowed card kinds for the subject's card; empty = any
    description: str = ""


@dataclass
class Schema:
    """Predicate declarations: cardinality (one vs many), value type, allowed card kinds."""

    predicates: dict[str, Predicate] = field(default_factory=dict)
    strict: bool = False  # reject predicates that were never declared

    def declare(self, name: str, cardinality: str = "one", value_type: str = "string",
                domain=(), description: str = "") -> "Schema":
        if cardinality not in ("one", "many"):
            raise ValueError("cardinality must be 'one' or 'many'")
        if value_type not in _TYPES:
            raise ValueError(f"value_type must be one of {sorted(_TYPES)}")
        self.predicates[name] = Predicate(name, cardinality, value_type, frozenset(domain), description)
        return self

    def validate(self, card: Card) -> Card:
        """Return the card with exclusivity set from the schema, or raise SchemaError."""
        spec = self.predicates.get(card.predicate)
        if spec is None:
            if self.strict:
                raise SchemaError(f"predicate {card.predicate!r} is not declared")
            return card
        if not _TYPES[spec.value_type](card.value):
            raise SchemaError(f"{card.predicate}: {card.value!r} is not a valid {spec.value_type}")
        if spec.domain and card.kind not in spec.domain:
            raise SchemaError(f"{card.predicate}: card kind {card.kind!r} not in {sorted(spec.domain)}")
        return replace(card, exclusive=(spec.cardinality == "one"))


def norm(name: str) -> str:
    return " ".join(name.lower().split())


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str
    frame_id: int
    t: float


class Graph:
    def __init__(self, mem: Memory, schema: Schema | None = None):
        self.mem = mem
        self.schema = schema

    # ---- data ------------------------------------------------------------------------
    def triples(self, subject: str | None = None, predicate: str | None = None, obj: str | None = None,
                as_of: float | None = None) -> list[Triple]:
        """Live triples at `as_of` (default: now), optionally filtered."""
        out = []
        for f in self.mem.find({"kind": "card"}, as_of=as_of):
            m = f.meta
            tr = Triple(m["subject"], m["predicate"], m["value"], f.id, f.t)
            if subject is not None and norm(tr.subject) != norm(subject):
                continue
            if predicate is not None and tr.predicate != predicate:
                continue
            if obj is not None and norm(tr.object) != norm(obj):
                continue
            out.append(tr)
        return out

    def entities(self, as_of: float | None = None) -> set[str]:
        """Entity names: every subject, plus values that are themselves subjects or are
        declared as entity-valued by the schema."""
        ts = self.triples(as_of=as_of)
        subjects = {norm(t.subject) for t in ts}
        nodes = set(subjects)
        for t in ts:
            spec = self.schema.predicates.get(t.predicate) if self.schema else None
            if norm(t.object) in subjects or (spec and spec.value_type == "entity"):
                nodes.add(norm(t.object))
        return nodes

    # ---- traversal -------------------------------------------------------------------
    def neighbors(self, entity: str, as_of: float | None = None) -> list[dict]:
        """Direct edges of an entity: outgoing facts and incoming references."""
        e = norm(entity)
        out = []
        for t in self.triples(as_of=as_of):
            if norm(t.subject) == e:
                out.append({"direction": "out", "predicate": t.predicate, "entity": t.object, "frame_id": t.frame_id})
            if norm(t.object) == e:
                out.append({"direction": "in", "predicate": t.predicate, "entity": t.subject, "frame_id": t.frame_id})
        return out

    def related(self, entity: str, hops: int = 2, as_of: float | None = None) -> dict[str, list[tuple[str, str, str]]]:
        """Entities reachable within `hops` edges, each with one shortest path
        [(from, predicate, to), ...] (edges followed in either direction)."""
        nodes = self.entities(as_of)
        adj: dict[str, list[tuple[str, str, str]]] = {}
        for t in self.triples(as_of=as_of):
            a, b = norm(t.subject), norm(t.object)
            if b in nodes:
                adj.setdefault(a, []).append((t.predicate, b, "→"))
                adj.setdefault(b, []).append((t.predicate, a, "←"))
        start = norm(entity)
        seen = {start}
        paths: dict[str, list] = {}
        q = deque([(start, [])])
        while q:
            cur, path = q.popleft()
            if len(path) >= hops:
                continue
            for pred, nxt, arrow in adj.get(cur, ()):
                if nxt in seen:
                    continue
                seen.add(nxt)
                step = path + [(cur, pred if arrow == "→" else f"~{pred}", nxt)]
                paths[nxt] = step
                q.append((nxt, step))
        return paths

    def facts_about(self, entity: str, as_of: float | None = None) -> list[Triple]:
        return self.triples(subject=entity, as_of=as_of)

    # ---- retrieval -------------------------------------------------------------------
    def mentioned(self, query: str, as_of: float | None = None) -> list[str]:
        """Known entities named in the query (word-boundary match, longest first)."""
        q = re.sub(r"['\u2019]s\b", "", query)  # possessives: "Alice's" -> "Alice"
        q = " " + norm(re.sub(r"[^\w\s'-]", " ", q)) + " "
        names = sorted((n for n in self.entities(as_of) if len(n) >= 3), key=len, reverse=True)
        found = []
        for n in names:
            if f" {n} " in q and not any(n in f for f in found):
                found.append(n)
        return found

    def hits(self, query: str, as_of: float | None = None, limit: int = 30) -> list[tuple[int, float]]:
        """Frame ids of the cards (and their source frames) about entities in the query, plus
        cards of neighbouring entities at lower weight. Used as a third list in RRF fusion."""
        ents = self.mentioned(query, as_of)
        scored: dict[int, float] = {}
        for rank, e in enumerate(ents):
            w = 1.0 / (1 + rank)
            for t in self.triples(as_of=as_of):
                if norm(t.subject) == e or norm(t.object) == e:
                    scored[t.frame_id] = max(scored.get(t.frame_id, 0), w)
                    src = self.mem.get(t.frame_id).meta.get("source")
                    if src is not None and self.mem.store.is_live(src, as_of):
                        scored[src] = max(scored.get(src, 0), 0.9 * w)
            for other in self.related(e, hops=1, as_of=as_of):
                for t in self.triples(subject=other, as_of=as_of):
                    scored[t.frame_id] = max(scored.get(t.frame_id, 0), 0.4 * w)
        return sorted(scored.items(), key=lambda x: -x[1])[:limit]


class NerExtractor:
    """spaCy NER -> co-occurrence cards (extra: `ner`, plus `python -m spacy download en_core_web_sm`).
    Experimental: emits non-exclusive relationship cards between entities sharing a sentence."""

    LABELS = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT"}

    def __init__(self, model: str = "en_core_web_sm"):
        try:
            import spacy
        except ImportError as e:
            raise ImportError("NER needs the 'ner' extra: pip install 'playmemory[ner]'") from e
        self.nlp = spacy.load(model)

    def extract(self, text: str) -> list[Card]:
        cards, seen = [], set()
        for sent in self.nlp(text).sents:
            ents = [e.text.strip() for e in sent.ents if e.label_ in self.LABELS]
            for i, a in enumerate(ents):
                for b in ents[i + 1 :]:
                    if a.lower() != b.lower() and (a.lower(), b.lower()) not in seen:
                        seen.add((a.lower(), b.lower()))
                        cards.append(Card("relationship", a, "co_occurs_with", b, exclusive=False))
        return cards
