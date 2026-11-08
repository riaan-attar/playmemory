"""LLM-judged answer accuracy on LoCoMo: does the retrieved memory let a model ANSWER?

    ANTHROPIC_API_KEY=... python -m bench.qa --data locomo10.json --sample 300 \
        --answer-model claude-haiku-5-5 --judge-model claude-haiku-5-5

For each question every system builds a context under the same character budget, one LLM call
answers from it, and a second call judges the answer against the gold answer (CORRECT/WRONG).
Systems: chunks, chunks+rr, span2+rr, span3+rr (rr = ONNX cross-encoder). Results are cached in
--cache (JSONL) so an interrupted run resumes without paying twice. Category 5 (adversarial) skipped.
Cost: sample x systems calls (x2 with an LLM judge). Scoring defaults to token F1 + contains-gold, which
needs no judge and works with a small local model:  --llm ollama:hermes3:3b --judge f1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.budget import CachedEmbedder, CachedReranker
from bench.quality import CATS, load_turns
from playmemory import ClaudeSynthesizer, OpenAICompatSynthesizer, EpisodeConfig, Memory, OnnxCrossEncoder, from_spec
from playmemory.ask import build_context

class MemoryPressure(RuntimeError):
    """Free memory / swap got dangerously low; the run stops cleanly (results so far are saved)."""


def memory_guard(min_free_mb: int = 2500, max_swap_pct: float = 70.0, wait_s: int = 90):
    """Callable that blocks while RAM is tight and raises MemoryPressure if it stays tight."""
    def read():
        m = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            m[k] = int(v.split()[0]) / 1024
        swap = 100 * (1 - m["SwapFree"] / m["SwapTotal"]) if m.get("SwapTotal") else 0.0
        return m["MemAvailable"], swap

    def guard():
        try:
            avail, swap = read()
        except OSError:  # no /proc: no guard
            return
        waited = 0
        while avail < min_free_mb or swap > max_swap_pct:
            if waited >= wait_s:
                raise MemoryPressure(f"only {avail:.0f} MB available, swap {swap:.0f}% used")
            time.sleep(10)
            waited += 10
            avail, swap = read()
    return guard


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env reader (KEY=VALUE lines); real environment variables win."""
    import os

    f = Path(path)
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


SHORT_SYSTEM = ("Answer the question using ONLY the numbered memory passages. Reply with the shortest possible "
                "answer (a few words, no explanation, no citations). If the passages do not say, reply: unknown.")


def _norm(t: str) -> list[str]:
    t = re.sub(r"\b(a|an|the)\b", " ", str(t).lower())
    return re.sub(r"[^\w\s]", " ", t).split()


def f1_and_contains(gold, pred) -> tuple[float, bool]:
    """LoCoMo-style token F1, and whether every gold token appears in the answer."""
    g, p = _norm(gold), _norm(pred)
    if not g or not p:
        return 0.0, False
    common = sum(min(g.count(w), p.count(w)) for w in set(g))
    contains = all(w in p for w in g)
    if common == 0:
        return 0.0, contains
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc), contains


JUDGE_SYSTEM = ("You grade answers to questions about a long conversation. Reply with exactly one word: "
                "CORRECT if the model answer conveys the same facts as the gold answer (extra detail is fine, "
                "dates may be phrased differently), otherwise WRONG.")
SYSTEMS = ("chunks", "chunks+rr", "span2+rr", "span3+rr")


def frames_for(mem, system: str, q: str, budget: int, rr) -> list:
    if system.startswith("chunks"):
        hits = mem.search(q, k=50, mode="hybrid", rerank=rr if system.endswith("rr") else None, rerank_pool=30)
        out, used = [], 0
        for h in hits:
            n = len(h.frame.text) + 1
            if used + n <= budget:
                out.append(h.frame)
                used += n
        return sorted(out, key=lambda f: f.id)
    k = int(system[4])
    spans = mem.recall_span(q, k, budget, rerank=rr if system.endswith("rr") else None, rerank_pool=30)
    return [f for sp in spans for f in sp.frames]


def run(data, embedder, *, answer, judge, budget, sample, rr, cache_path, seed=0, systems=SYSTEMS, start=0, guard=None):
    cache = {}
    if cache_path and Path(cache_path).exists():
        for line in open(cache_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a crash can leave a half-written last line: skip it
                continue
            cache[(r["conv"], r["q"], r["system"])] = r
    out = open(cache_path, "a") if cache_path else None
    items = []
    for ci, smp in enumerate(data):
        for qa in smp["qa"]:
            if qa["category"] in CATS and qa.get("answer") is not None:
                items.append((ci, qa))
    random.Random(seed).shuffle(items)
    items = items[start : start + sample] if sample else items[start:]
    by_conv = defaultdict(list)
    for ci, qa in items:
        by_conv[ci].append(qa)
    score = defaultdict(list)
    calls = 0
    for ci, qas in sorted(by_conv.items()):
        smp = data[ci]
        mem = None  # built lazily: a fully cached conversation costs nothing
        for qa in qas:
            for system in systems:
                key = (smp["sample_id"], qa["question"], system)
                if key in cache:
                    r = cache[key]
                else:
                    if mem is None:
                        turns = load_turns(smp["conversation"])
                        mem = Memory(None, embedder, episode_config=EpisodeConfig(boundary_key="session"))
                        mem.add_many([t for _, t in turns],
                                     metas=[{"session": int(d.split(":")[0][1:]), "dia": d} for d, _ in turns],
                                     times=[float(i) for i in range(len(turns))])
                    ctx = build_context(frames_for(mem, system, qa["question"], budget, rr))
                    if guard:
                        guard()
                    raw = answer(qa["question"], ctx) or ""
                    pred = raw.strip().splitlines()[0] if raw.strip() else ""  # small models add explanations; score the answer line
                    f1, contains = f1_and_contains(qa["answer"], pred)
                    r = {"conv": smp["sample_id"], "q": qa["question"], "system": system, "cat": qa["category"],
                         "pred": pred, "f1": f1, "contains": contains, "correct": contains}
                    calls += 1
                    if judge is not None:
                        verdict = judge(f"Question: {qa['question']}\nGold answer: {qa['answer']}", f"Model answer: {pred}")
                        r["correct"] = verdict.strip().upper().startswith("CORRECT")
                        calls += 1
                    cache[key] = r
                    if out:  # durable before moving on: survives a crash or power loss
                        out.write(json.dumps(r) + "\n")
                        out.flush()
                        os.fsync(out.fileno())
                for g in ("all", CATS[r["cat"]]):
                    score[(system, g)].append(float(r["correct"]))
                    score[(system, g, "f1")].append(float(r.get("f1", 0.0)))
    return score, calls


def report(score, systems=SYSTEMS):
    print(f"{'system':<12}{'group':<13}{'correct':>9}{'F1':>8}{'n':>6}")
    for g in ("all", *CATS.values()):
        for s in systems:
            v = score.get((s, g), [])
            if v:
                print(f"{s:<12}{g:<13}{np.mean(v):>9.3f}{np.mean(score.get((s, g, 'f1'), [0.0])):>8.3f}{len(v):>6}")
        print()


class OllamaChat:
    """Ollama's native /api/chat (lets us set options such as num_gpu=0 to stay on CPU) with retries:
    a crashed runner (e.g. GPU out of memory) is restarted by the server, so a retry usually works."""

    def __init__(self, model, system, max_tokens=60, url="http://localhost:11434", options=None, retries=3):
        self.model, self.system, self.url, self.retries = model, system, url.rstrip("/"), retries
        self.options = {"num_predict": max_tokens, "temperature": 0, "num_ctx": 4096, **(options or {})}

    def __call__(self, question, context):
        import urllib.error
        import urllib.request

        body = json.dumps({"model": self.model, "stream": False, "options": self.options, "messages": [
            {"role": "system", "content": self.system},
            {"role": "user", "content": f"Memory passages:\n{context}\n\nQuestion: {question}"}]}).encode()
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url + "/api/chat", body, {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    return json.load(r)["message"]["content"].strip()
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(5 * (attempt + 1))


def make_llm(spec: str, system: str, max_tokens: int, ollama_options=None):
    kind, _, rest = spec.partition(":")
    if kind == "claude":
        return ClaudeSynthesizer(rest or "claude-haiku-5-5", max_tokens=max_tokens, system=system)
    if kind == "ollama":
        model, _, url = rest.partition("@")
        return OllamaChat(model, system, max_tokens, url or "http://localhost:11434", ollama_options)
    if kind == "openai":
        model, _, url = rest.partition("@")
        default = "http://localhost:11434/v1/chat/completions" if kind == "ollama" else "https://api.openai.com/v1/chat/completions"
        return OpenAICompatSynthesizer(model, url or default, max_tokens=max_tokens, system=system, timeout=300.0)
    raise SystemExit(f"unknown llm spec {spec!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--embedder", default="onnx")
    ap.add_argument("--sample", "--count", dest="sample", type=int, default=300,
                    help="questions in this batch (0 = all remaining)")
    ap.add_argument("--start", type=int, default=0,
                    help="first question of the batch in the fixed shuffled order (batch k starts at k*count)")
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--llm", default="claude:claude-haiku-5-5",
                    help="answering model: claude:MODEL | ollama:MODEL[@URL] | openai:MODEL[@URL]")
    ap.add_argument("--judge", default="f1",
                    help="f1 (token F1 + contains-gold, no LLM judge) | claude:MODEL | ollama:MODEL")
    ap.add_argument("--cache", default="qa_cache.jsonl")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--ollama-opt", action="append", default=[], metavar="K=V",
                    help="Ollama option, e.g. num_gpu=0 to run on CPU only (repeatable)")
    ap.add_argument("--min-free-mb", type=int, default=2500,
                    help="pause, then stop cleanly, if available RAM stays below this")
    a = ap.parse_args()
    load_dotenv()

    import fcntl  # one run per cache file: two overlapping runs once exhausted this machine's memory
    lock = open(a.cache + ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"another bench.qa run is already using {a.cache}; refusing to start a second one")

    data = json.load(open(a.data))
    emb = CachedEmbedder(from_spec(a.embedder))
    rr = None if a.no_rerank else CachedReranker(OnnxCrossEncoder())
    opts = {k: int(v) if v.lstrip("-").isdigit() else v for k, v in (o.split("=", 1) for o in a.ollama_opt)}
    answer = make_llm(a.llm, SHORT_SYSTEM, 60, opts)
    judge_llm = None if a.judge == "f1" else make_llm(a.judge, JUDGE_SYSTEM, 5)
    judge = None if judge_llm is None else (lambda q, c: judge_llm(q, c))
    systems = SYSTEMS if rr else ("chunks", "span2", "span3")
    t0 = time.perf_counter()
    try:
        score, calls = run(data, emb, answer=answer, judge=judge, budget=a.budget, sample=a.sample, rr=rr,
                           cache_path=a.cache, systems=systems, start=a.start,
                           guard=memory_guard(a.min_free_mb))
    except MemoryPressure as e:
        print(f"\nSTOPPED: {e}. Everything answered so far is saved in {a.cache}; rerun the same command to resume.")
        raise SystemExit(3)
    print(f"\nLoCoMo answer accuracy, budget={a.budget} chars, answering={a.llm}, scoring={a.judge}, "
          f"questions {a.start}-{a.start + a.sample - 1 if a.sample else 'end'} (this batch), "
          f"{calls} new calls, {time.perf_counter()-t0:.0f}s\n")
    report(score, systems)
    if a.start:  # cumulative view over everything answered so far (served from the cache, no new calls)
        def _missing(*_):
            raise RuntimeError("cumulative view needs earlier batches to be complete")
        try:
            cum, _ = run(data, emb, answer=_missing, judge=None, budget=a.budget,
                         sample=a.start + a.sample, rr=rr, cache_path=a.cache, systems=systems, start=0)
            print(f"== cumulative: questions 0-{a.start + a.sample - 1} ==\n")
            report(cum, systems)
        except RuntimeError as e:
            print(f"(no cumulative table: {e})")


if __name__ == "__main__":
    main()
