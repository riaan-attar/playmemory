from __future__ import annotations

import re
import zlib
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int
    name: str  # identity used to bind a file to its embedder
    spec: str  # string from_spec() can rebuild this embedder from (never contains secrets)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""
        ...


class HashEmbedder:
    """Dependency-free bag-of-words feature-hashing embedder.

    Lexical only (no semantics) but deterministic and fast; good for tests and
    for benchmarking the memory structure independently of model quality.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = self.spec = f"hash-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _TOKEN.findall(text.lower()):
                h = zlib.crc32(tok.encode())
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    """Wrapper around sentence-transformers (install the `st` extra)."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self.name = model
        self.spec = f"st:{model}"
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True), dtype=np.float32
        )


DebugEmbedder = HashEmbedder  # clearer name: lexical only, for tests/benchmarks


class HTTPEmbedder:
    """OpenAI-compatible `/embeddings` endpoint (OpenAI, Ollama, vLLM, llama.cpp...).

    Stdlib only. `url` is the full endpoint, e.g. https://api.openai.com/v1/embeddings.
    """

    def __init__(self, url: str, model: str, api_key: str | None = None,
                 dim: int | None = None, batch: int = 64, timeout: float = 60.0):
        self.url, self.model, self.api_key = url, model, api_key
        self.batch, self.timeout = batch, timeout
        self.name = f"http:{model}"
        self.spec = f"http:{url}|{model}"
        if api_key is None:
            import os

            self.api_key = os.environ.get("PLAYMEMORY_API_KEY")
        self.dim = dim if dim is not None else self.embed(["dimension probe"]).shape[1]

    def _post(self, texts: list[str]) -> list[list[float]]:
        import json
        import urllib.request

        req = urllib.request.Request(
            self.url,
            data=json.dumps({"model": self.model, "input": texts}).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.load(r)
        rows = sorted(body["data"], key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in rows]

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            out += self._post(texts[i : i + self.batch])
        arr = np.asarray(out, dtype=np.float32).reshape(len(texts), -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class OnnxEmbedder:
    """Local ONNX sentence embedder without torch (extra: `onnx`).

    Defaults to BGE-small-en-v1.5 (384-d, CLS pooling), downloaded once to
    ~/.cache/playmemory/. Much lighter than sentence-transformers.
    """

    HF = "https://huggingface.co/{repo}/resolve/main/{file}"

    def __init__(self, repo: str = "BAAI/bge-small-en-v1.5", pooling: str = "cls",
                 max_len: int = 512, model_path: str | None = None,
                 tokenizer_path: str | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = model_path or self._fetch(repo, "onnx/model.onnx")
        tokenizer_path = tokenizer_path or self._fetch(repo, "tokenizer.json")
        self._sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self._tok = Tokenizer.from_file(tokenizer_path)
        self._tok.enable_truncation(max_len)
        self._tok.enable_padding()
        self.pooling = pooling
        self.name = self.spec = f"onnx:{repo}"
        self.dim = int(self._sess.get_outputs()[0].shape[-1])

    @classmethod
    def _fetch(cls, repo: str, file: str) -> str:
        import urllib.request
        from pathlib import Path

        dest = Path.home() / ".cache" / "playmemory" / repo.replace("/", "__") / file.replace("/", "__")
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            urllib.request.urlretrieve(cls.HF.format(repo=repo, file=file), tmp)
            tmp.rename(dest)
        return str(dest)

    def embed(self, texts: list[str], batch: int = 32) -> np.ndarray:
        outs = []
        for i in range(0, len(texts), batch):
            enc = self._tok.encode_batch(texts[i : i + batch])
            ids = np.asarray([e.ids for e in enc], dtype=np.int64)
            mask = np.asarray([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self._sess.run(None, feed)[0]
            if self.pooling == "cls":
                vec = hidden[:, 0]
            else:
                m = mask[..., None].astype(np.float32)
                vec = (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)
            outs.append(vec)
        arr = np.vstack(outs).astype(np.float32) if outs else np.zeros((0, self.dim), np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def from_spec(spec: str, api_key: str | None = None) -> Embedder:
    """Build an embedder from a short spec: hash[-DIM] | onnx[:REPO] | st[:MODEL] |
    http:URL|MODEL[|APIKEY]."""
    kind, _, rest = spec.partition(":")
    if kind.startswith("hash"):
        dim = int(kind.split("-")[1]) if "-" in kind else 256
        return HashEmbedder(dim)
    if kind == "onnx":
        return OnnxEmbedder(rest) if rest else OnnxEmbedder()
    if "/" in spec and ":" not in spec:  # bare sentence-transformers model id (older files)
        return SentenceTransformerEmbedder(spec)
    if kind == "st":
        return SentenceTransformerEmbedder(rest) if rest else SentenceTransformerEmbedder()
    if kind == "http":
        # the URL itself contains ':' so split on '|' only
        parts = rest.split("|")
        if len(parts) < 2:
            raise ValueError("http spec is http:URL|MODEL[|APIKEY]")
        return HTTPEmbedder(parts[0], parts[1], parts[2] if len(parts) > 2 else api_key)
    raise ValueError(f"unknown embedder spec {spec!r}")
