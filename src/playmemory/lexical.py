from __future__ import annotations

import json
import math
import re

import numpy as np
from collections import defaultdict

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Incremental in-memory BM25 over frame ids (rebuilt from the log on open)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        self._total_len = 0

    def __len__(self) -> int:
        return len(self.doc_len)

    def add(self, text: str) -> None:
        fid = len(self.doc_len)
        toks = tokenize(text)
        counts: dict[str, int] = defaultdict(int)
        for t in toks:
            counts[t] += 1
        for t, c in counts.items():
            self.postings[t].append((fid, c))
        self.doc_len.append(len(toks))
        self._total_len += len(toks)

    def search(self, query: str, valid, k: int) -> list[tuple[int, float]]:
        """Top-k (id, score). `valid(fid)->bool` filters candidates."""
        n = len(self.doc_len)
        if n == 0:
            return []
        avg = self._total_len / n or 1.0
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for fid, tf in plist:
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[fid] / avg)
                scores[fid] += idf * tf * (self.k1 + 1) / norm
        ranked = sorted(((f, s) for f, s in scores.items() if valid(f)),
                        key=lambda x: -x[1])
        return ranked[:k]

    # persistence (checkpoint) ------------------------------------------------
    def to_arrays(self) -> dict[str, np.ndarray]:
        terms = list(self.postings)
        lens = [len(self.postings[t]) for t in terms]
        flat = [p for t in terms for p in self.postings[t]]
        ids = np.fromiter((f for f, _ in flat), dtype=np.int32, count=len(flat))
        tfs = np.fromiter((c for _, c in flat), dtype=np.int32, count=len(flat))
        return {
            "bm_terms": np.frombuffer(json.dumps(terms).encode(), np.uint8),
            "bm_ptr": np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
            "bm_ids": ids,
            "bm_tf": tfs,
            "bm_len": np.asarray(self.doc_len, dtype=np.int32),
        }

    @classmethod
    def from_arrays(cls, a: dict[str, np.ndarray], **kw) -> "BM25Index":
        self = cls(**kw)
        terms = json.loads(a["bm_terms"].tobytes())
        ptr, ids, tfs = a["bm_ptr"], a["bm_ids"].tolist(), a["bm_tf"].tolist()
        for i, t in enumerate(terms):
            lo, hi = int(ptr[i]), int(ptr[i + 1])
            self.postings[t] = list(zip(ids[lo:hi], tfs[lo:hi]))
        self.doc_len = a["bm_len"].tolist()
        self._total_len = int(sum(self.doc_len))
        return self
