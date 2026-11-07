from __future__ import annotations

import json
import math
import re
import unicodedata
from array import array

import numpy as np

TOKENIZER_VERSION = 2  # bump when tokenize() changes; checkpoints with another version are rebuilt

_WORD = re.compile(r"\w+")
# Scripts written without spaces: index overlapping character bigrams instead of words.
_CJK_RUN = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]+")


def _fold_accents(tok: str) -> str:
    """cafe == café, but only for Latin-script words (marks are meaningful elsewhere)."""
    if tok.isascii() or not unicodedata.name(tok[0], "").startswith("LATIN"):
        return tok
    d = unicodedata.normalize("NFD", tok)
    return unicodedata.normalize("NFC", "".join(c for c in d if unicodedata.category(c) != "Mn"))


def tokenize(text: str) -> list[str]:
    """Unicode-aware tokens: NFKC + casefold, Latin accents folded, CJK/Hangul as bigrams."""
    out: list[str] = []
    for word in _WORD.findall(unicodedata.normalize("NFKC", text).casefold()):
        pos = 0
        for m in _CJK_RUN.finditer(word):
            if m.start() > pos:
                out.append(_fold_accents(word[pos : m.start()]))
            run = m.group()
            out.extend([run] if len(run) == 1 else (run[i : i + 2] for i in range(len(run) - 1)))
            pos = m.end()
        if pos < len(word):
            out.append(_fold_accents(word[pos:]))
    return out


class BM25Index:
    """Incremental BM25 over frame ids. Postings are compact int32 arrays that
    are scored with numpy, so a query touches only its terms' posting lists."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.post_ids: dict[str, array] = {}
        self.post_tf: dict[str, array] = {}
        self.doc_len = array("i")
        self._total_len = 0

    def __len__(self) -> int:
        return len(self.doc_len)

    def add(self, text: str) -> None:
        fid = len(self.doc_len)
        toks = tokenize(text)
        counts: dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            ids = self.post_ids.get(t)
            if ids is None:
                ids = self.post_ids[t] = array("i")
                self.post_tf[t] = array("i")
            ids.append(fid)
            self.post_tf[t].append(c)
        self.doc_len.append(len(toks))
        self._total_len += len(toks)

    def search(self, query: str, mask: np.ndarray | None, k: int) -> list[tuple[int, float]]:
        """Top-k (id, score). `mask` (bool, len == n) restricts candidates."""
        n = len(self.doc_len)
        if n == 0:
            return []
        avg = self._total_len / n or 1.0
        dl = np.frombuffer(self.doc_len, dtype=np.int32)
        scores = np.zeros(n, dtype=np.float32)
        hit_any = False
        views = []
        try:
            for term in set(tokenize(query)):
                if term not in self.post_ids:
                    continue
                ids = np.frombuffer(self.post_ids[term], dtype=np.int32)
                tf = np.frombuffer(self.post_tf[term], dtype=np.int32).astype(np.float32)
                views += [ids, tf]
                df = len(ids)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                norm = tf + self.k1 * (1 - self.b + self.b * dl[ids] / avg)
                scores[ids] += idf * tf * (self.k1 + 1) / norm  # ids are unique per term
                hit_any = True
            if not hit_any:
                return []
            valid = scores > 0
            if mask is not None:
                valid &= mask
            cand = np.flatnonzero(valid)
            if len(cand) == 0:
                return []
            k = min(k, len(cand))
            sel = cand[np.argpartition(-scores[cand], k - 1)[:k]]
            sel = sel[np.argsort(-scores[sel], kind="stable")]
            return [(int(i), float(scores[i])) for i in sel]
        finally:
            del dl, views  # release buffer exports so the arrays can grow again

    # persistence (checkpoint) ------------------------------------------------
    def to_arrays(self) -> dict[str, np.ndarray]:
        terms = list(self.post_ids)
        lens = np.fromiter((len(self.post_ids[t]) for t in terms), dtype=np.int64, count=len(terms))
        ids = np.concatenate([np.frombuffer(self.post_ids[t], np.int32) for t in terms]) if terms else np.zeros(0, np.int32)
        tfs = np.concatenate([np.frombuffer(self.post_tf[t], np.int32) for t in terms]) if terms else np.zeros(0, np.int32)
        return {
            "bm_terms": np.frombuffer(json.dumps(terms).encode(), np.uint8),
            "bm_ptr": np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
            "bm_ids": ids.copy(),
            "bm_tf": tfs.copy(),
            "bm_len": np.frombuffer(self.doc_len, np.int32).copy(),
            "bm_tokver": np.asarray([TOKENIZER_VERSION], dtype=np.int32),
        }

    @classmethod
    def from_arrays(cls, a: dict[str, np.ndarray], **kw) -> "BM25Index":
        self = cls(**kw)
        terms = json.loads(a["bm_terms"].tobytes())
        ptr, ids, tfs = a["bm_ptr"], a["bm_ids"].astype(np.int32), a["bm_tf"].astype(np.int32)
        for i, t in enumerate(terms):
            lo, hi = int(ptr[i]), int(ptr[i + 1])
            x, y = array("i"), array("i")
            x.frombytes(ids[lo:hi].tobytes())
            y.frombytes(tfs[lo:hi].tobytes())
            self.post_ids[t], self.post_tf[t] = x, y
        self.doc_len.frombytes(a["bm_len"].astype(np.int32).tobytes())
        self._total_len = int(a["bm_len"].sum())
        return self
