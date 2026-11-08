"""LLM-judged answer accuracy on LoCoMo: does the retrieved memory let a model ANSWER?

    ANTHROPIC_API_KEY=... python -m bench.qa --data locomo10.json --sample 300 \
        --answer-model claude-haiku-5-5 --judge-model claude-haiku-5-5

For each question every system builds a context under the same character budget, one LLM call
answers from it, and a second call judges the answer against the gold answer (CORRECT/WRONG).
Systems: chunks, chunks+rr, span2+rr, span3+rr (rr = ONNX cross-encoder). Results are cached in
--cache (JSONL) so an interrupted run resumes without paying twice. Category 5 (adversarial) skipped.
Cost: sample x systems x 2 calls (a few thousand small calls for the full set).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.budget import CachedEmbedder, CachedReranker
from bench.quality import CATS, load_turns
from playmemory import ClaudeSynthesizer, EpisodeConfig, Memory, OnnxCrossEncoder, from_spec
from playmemory.ask import build_context

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


def run(data, embedder, *, answer, judge, budget, sample, rr, cache_path, seed=0, systems=SYSTEMS):
    cache = {}
    if cache_path and Path(cache_path).exists():
        for line in open(cache_path):
            r = json.loads(line)
            cache[(r["conv"], r["q"], r["system"])] = r
    out = open(cache_path, "a") if cache_path else None
    items = []
    for ci, smp in enumerate(data):
        for qa in smp["qa"]:
            if qa["category"] in CATS and qa.get("answer") is not None:
                items.append((ci, qa))
    random.Random(seed).shuffle(items)
    items = items[:sample] if sample else items
    by_conv = defaultdict(list)
    for ci, qa in items:
        by_conv[ci].append(qa)
    score = defaultdict(list)
    calls = 0
    for ci, qas in sorted(by_conv.items()):
        smp = data[ci]
        turns = load_turns(smp["conversation"])
        mem = Memory(None, embedder, episode_config=EpisodeConfig(boundary_key="session"))
        mem.add_many([t for _, t in turns], metas=[{"session": int(d.split(":")[0][1:]), "dia": d} for d, _ in turns],
                     times=[float(i) for i in range(len(turns))])
        for qa in qas:
            for system in systems:
                key = (smp["sample_id"], qa["question"], system)
                if key in cache:
                    r = cache[key]
                else:
                    ctx = build_context(frames_for(mem, system, qa["question"], budget, rr))
                    pred = answer(qa["question"], ctx)
                    verdict = judge(f"Question: {qa['question']}\nGold answer: {qa['answer']}", f"Model answer: {pred}")
                    r = {"conv": smp["sample_id"], "q": qa["question"], "system": system, "cat": qa["category"],
                         "pred": pred, "correct": verdict.strip().upper().startswith("CORRECT")}
                    calls += 2
                    cache[key] = r
                    if out:
                        out.write(json.dumps(r) + "\n")
                        out.flush()
                for g in ("all", CATS[r["cat"]]):
                    score[(system, g)].append(float(r["correct"]))
    return score, calls


def report(score, systems=SYSTEMS):
    print(f"{'system':<12}{'group':<13}{'accuracy':>9}{'n':>6}")
    for g in ("all", *CATS.values()):
        for s in systems:
            v = score.get((s, g), [])
            if v:
                print(f"{s:<12}{g:<13}{np.mean(v):>9.3f}{len(v):>6}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--embedder", default="onnx")
    ap.add_argument("--sample", type=int, default=300, help="random questions to evaluate (0 = all)")
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--answer-model", default="claude-haiku-5-5")
    ap.add_argument("--judge-model", default="claude-haiku-5-5")
    ap.add_argument("--cache", default="qa_cache.jsonl")
    ap.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args()
    load_dotenv()
    data = json.load(open(a.data))
    emb = CachedEmbedder(from_spec(a.embedder))
    rr = None if a.no_rerank else CachedReranker(OnnxCrossEncoder())
    answer = ClaudeSynthesizer(a.answer_model, max_tokens=200)
    judge_llm = ClaudeSynthesizer(a.judge_model, max_tokens=5, system=JUDGE_SYSTEM)
    systems = SYSTEMS if rr else ("chunks", "span2", "span3")
    t0 = time.perf_counter()
    score, calls = run(data, emb, answer=answer, judge=lambda q, c: judge_llm(q, c), budget=a.budget,
                       sample=a.sample, rr=rr, cache_path=a.cache, systems=systems)
    print(f"\nLoCoMo LLM-judged accuracy, budget={a.budget} chars, {a.answer_model} answering, "
          f"{a.judge_model} judging, {calls} new calls, {time.perf_counter()-t0:.0f}s\n")
    report(score, systems)


if __name__ == "__main__":
    main()
