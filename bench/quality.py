"""Retrieval quality on LoCoMo (long conversational memory QA).

    python -m bench.quality --data locomo10.json [--embedder onnx] [--limit 10]

Metric: does the retrieved set contain the gold evidence turns (dia_ids)?
  evidence-recall@k = mean fraction of gold turns found in the top-k hits
  hit@k            = fraction of questions with ALL gold turns found
No LLM judge is needed. Category 5 (adversarial, no evidence) is skipped.
Data: https://github.com/snap-research/locomo/blob/main/data/locomo10.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict

import numpy as np

from playmemory import Memory, from_spec

CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
SYSTEMS = {
    "bm25": dict(mode="lexical"),
    "vector": dict(mode="vector"),
    "hybrid": dict(mode="hybrid"),
    "hybrid+ctx1": dict(mode="hybrid", context=1),
}


def load_turns(conv: dict) -> list[tuple[str, str]]:
    """[(dia_id, text)] in timeline order, each prefixed with speaker and session date."""
    out = []
    sessions = sorted(
        (k for k in conv if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1])
    )
    for s in sessions:
        date = conv.get(f"{s}_date_time", "")
        for turn in conv[s]:
            text = turn["text"]
            if turn.get("blip_caption"):
                text += f" [image: {turn['blip_caption']}]"
            out.append((turn["dia_id"], f"[{date}] {turn['speaker']}: {text}"))
    return out


def evaluate(data: list[dict], embedder, ks: tuple[int, ...], limit: int | None):
    rows = defaultdict(lambda: defaultdict(list))  # system -> metric key -> values
    lat = defaultdict(list)
    for sample in data[:limit]:
        turns = load_turns(sample["conversation"])
        mem = Memory(None, embedder)
        mem.add_many([t for _, t in turns], metas=[{"dia": d} for d, _ in turns],
                     times=[float(i) for i in range(len(turns))])
        dia_to_id = {d: i for i, (d, _) in enumerate(turns)}
        for qa in sample["qa"]:
            gold = {dia_to_id[d] for ev in qa.get("evidence", []) for d in re.findall(r"D\d+:\d+", ev)
                    if d in dia_to_id}
            if not gold or qa["category"] not in CATS:
                continue
            for name, kw in SYSTEMS.items():
                for k in ks:
                    t0 = time.perf_counter()
                    hits = mem.search(qa["question"], k=k, **kw)
                    if k == ks[-1]:
                        lat[name].append(time.perf_counter() - t0)
                    got = {f.id for h in hits for f in h.context()}
                    frac = len(gold & got) / len(gold)
                    for key in (("all", k), (CATS[qa["category"]], k)):
                        rows[name][key + ("recall",)].append(frac)
                        rows[name][key + ("hit",)].append(float(frac == 1.0))
    return rows, lat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--embedder", default="onnx", help="spec for playmemory.from_spec")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    args = ap.parse_args()
    data = json.load(open(args.data))
    emb = from_spec(args.embedder)
    t0 = time.perf_counter()
    rows, lat = evaluate(data, emb, tuple(args.ks), args.limit)
    n = len(rows["hybrid"][("all", args.ks[0], "recall")])
    print(f"\nLoCoMo retrieval, embedder={emb.name}, questions={n}, ({time.perf_counter()-t0:.0f}s)\n")
    groups = ["all", *CATS.values()]
    for metric, title in (("recall", "evidence-recall@k"), ("hit", "hit@k (all gold turns retrieved)")):
        print(f"== {title} ==")
        print(f"{'system':<14}{'group':<13}" + "".join(f"{'k='+str(k):>8}" for k in args.ks))
        for name in SYSTEMS:
            for g in groups:
                vals = [np.mean(rows[name][(g, k, metric)]) if rows[name][(g, k, metric)] else float("nan")
                        for k in args.ks]
                print(f"{name:<14}{g:<13}" + "".join(f"{v:>8.3f}" for v in vals))
        print()
    print("search latency p50/p95 ms (incl. query embedding, k=%d):" % args.ks[-1])
    for name, v in lat.items():
        print(f"  {name:<14}{np.percentile(v, 50)*1000:>8.1f}{np.percentile(v, 95)*1000:>8.1f}")


if __name__ == "__main__":
    main()
