"""Rerankers rescore the top candidates with a stronger (slower) model."""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]:
        """Higher = more relevant; one score per passage."""
        ...


class OnnxCrossEncoder:
    """Local cross-encoder (extra `onnx`, no torch). Default: ms-marco MiniLM-L6,
    downloaded once to ~/.cache/playmemory/."""

    def __init__(self, repo: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_len: int = 512,
                 model_path: str | None = None, tokenizer_path: str | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        from .embed import OnnxEmbedder

        model_path = model_path or OnnxEmbedder._fetch(repo, "onnx/model.onnx")
        tokenizer_path = tokenizer_path or OnnxEmbedder._fetch(repo, "tokenizer.json")
        from .embed import _session_options

        self._sess = ort.InferenceSession(model_path, _session_options(ort), providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self._tok = Tokenizer.from_file(tokenizer_path)
        self._tok.enable_truncation(max_len)
        self._tok.enable_padding()
        self.name = f"onnx-ce:{repo}"

    def score(self, query: str, passages: list[str], batch: int = 16) -> list[float]:
        out: list[float] = []
        for i in range(0, len(passages), batch):
            enc = self._tok.encode_batch([(query, p) for p in passages[i : i + batch]])
            ids = np.asarray([e.ids for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": np.asarray([e.attention_mask for e in enc], dtype=np.int64)}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.asarray([e.type_ids for e in enc], dtype=np.int64)
            logits = self._sess.run(None, feed)[0]
            out += np.asarray(logits, dtype=np.float32).reshape(len(ids), -1)[:, 0].tolist()
        return out


class LLMReranker:
    """Wrap any `fn(query, passages) -> list[float]` (e.g. an LLM scoring call)."""

    def __init__(self, fn: Callable[[str, list[str]], list[float]]):
        self.fn = fn

    def score(self, query: str, passages: list[str]) -> list[float]:
        scores = list(self.fn(query, passages))
        if len(scores) != len(passages):
            raise ValueError("reranker returned the wrong number of scores")
        return scores
