"""ask(): retrieval + answer synthesis with citations back into the memory.

The memory supplies numbered passages ("[12] ..."); a Synthesizer (any callable
`fn(question, context) -> str`) writes the answer and cites passage ids; we map
the cited ids back to frames. Bring your own LLM or use ClaudeSynthesizer /
OpenAICompatSynthesizer (stdlib HTTP, no SDK needed).
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from .frame import Frame

SYSTEM = (
    "Answer the question using ONLY the numbered memory passages. Cite the passage numbers you "
    "used like [12]. If the passages do not contain the answer, say you don't know."
)


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Frame] = field(default_factory=list)  # frames the answer cited
    context: list[Frame] = field(default_factory=list)  # every frame shown to the model
    prompt: str = ""


Synthesizer = Callable[[str, str], str]


def build_context(frames: list[Frame]) -> str:
    return "\n".join(f"[{f.id}] {f.text}" for f in frames)


def extract_citations(answer: str, frames: list[Frame]) -> list[Frame]:
    by_id = {f.id: f for f in frames}
    seen, out = set(), []
    for m in re.finditer(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]", answer):
        for n in re.split(r"[,;]", m[1]):
            i = int(n)
            if i in by_id and i not in seen:
                seen.add(i)
                out.append(by_id[i])
    return out


def _post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class ClaudeSynthesizer:
    """Anthropic Messages API over plain HTTP. Key from api_key or ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-haiku-5-5", api_key: str | None = None, max_tokens: int = 512,
                 base_url: str = "https://api.anthropic.com", timeout: float = 60.0):
        self.model, self.max_tokens, self.timeout = model, max_tokens, timeout
        self.url = base_url.rstrip("/") + "/v1/messages"
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ClaudeSynthesizer needs api_key or ANTHROPIC_API_KEY")

    def __call__(self, question: str, context: str) -> str:
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": SYSTEM,
                "messages": [{"role": "user", "content": f"Memory passages:\n{context}\n\nQuestion: {question}"}]}
        out = _post(self.url, {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, body, self.timeout)
        return "".join(b.get("text", "") for b in out["content"] if b.get("type") == "text").strip()


class OpenAICompatSynthesizer:
    """Any OpenAI-compatible /chat/completions endpoint (OpenAI, Ollama, vLLM, ...)."""

    def __init__(self, model: str, url: str = "https://api.openai.com/v1/chat/completions",
                 api_key: str | None = None, max_tokens: int = 512, timeout: float = 60.0):
        self.model, self.url, self.max_tokens, self.timeout = model, url, max_tokens, timeout
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def __call__(self, question: str, context: str) -> str:
        body = {"model": self.model, "max_tokens": self.max_tokens, "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Memory passages:\n{context}\n\nQuestion: {question}"}]}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return _post(self.url, headers, body, self.timeout)["choices"][0]["message"]["content"].strip()
