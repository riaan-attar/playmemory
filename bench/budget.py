"""Does returning a timeline SPAN beat returning top-k chunks at the same size?

    python -m bench.budget --data locomo10.json [--embedder onnx] [--budgets 500 1000 2000 4000]

For every LoCoMo question each system may return at most B characters of memory.
Metric: evidence-recall (fraction of gold dialogue turns inside what was returned)
and hit (ALL gold turns returned). Systems:
  chunks      hybrid top hits, whole frames, until the budget is full
  chunks+ctx  same, but each hit brings its +-1 neighbours (kept whole or skipped)
  span1       Memory.recall_span(k=1): one scene, trimmed to the budget
  span2       Memory.recall_span(k=2): two scenes sharing the budget
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict

import numpy as np

from bench.quality import CATS, load_turns
from playmemory import EpisodeConfig, Memory, from_spec


class CachedEmbedder:
    """Memoise embeddings by text so each question is embedded once."""

    def __init__(self, inner):
        self.inner, self.dim, self.name, self.cache = inner, inner.dim, inner.name, {}

    def embed(self, texts):
        miss = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if miss:
            for t, v in zip(miss, self.inner.embed(miss)):
                self.cache[t] = v
        return np.vstack([self.cache[t] for t in texts])


def pack_chunks(hits, budget, context):
    got, used = set(), 0
    for h in hits:
        group = h.context() if context else [h.frame]
        new = [f for f in group if f.id not in got]
        cost = sum(len(f.text) + 1 for f in new)
        if used + cost > budget:
            if context:  # try the bare hit before giving up on it
                new = [h.frame] if h.frame.id not in got else []
                cost = sum(len(f.text) + 1 for f in new)
                if used + cost > budget:
                    continue
            else:
                continue
        got.update(f.id for f in new)
        used += cost
    return got


def run(data, embedder, budgets, limit, cfg):
    rows = defaultdict(list)
    sizes = defaultdict(list)
    for sample in data[:limit]:
        turns = load_turns(sample["conversation"])
        mem = Memory(None, embedder, episode_config=cfg)
        sess = []
        for d, _ in turns:
            sess.append({"session": int(d.split(":")[0][1:]), "dia": d})
        mem.add_many([t for _, t in turns], metas=sess, times=[float(i) for i in range(len(turns))])
        dia_to_id = {d: i for i, (d, _) in enumerate(turns)}
        n_ep = len(mem.episode_list())
        print(f"  {sample['sample_id']}: {len(turns)} turns, {n_ep} episodes", flush=True)
        for qa in sample["qa"]:
            gold = {dia_to_id[d] for ev in qa.get("evidence", []) for d in re.findall(r"D\d+:\d+", ev)
                    if d in dia_to_id}
            if not gold or qa["category"] not in CATS:
                continue
            q = qa["question"]
            hits = mem.search(q, k=50, context=1, mode="hybrid")
            for B in budgets:
                got = {
                    "chunks": pack_chunks(hits, B, False),
                    "chunks+ctx": pack_chunks(hits, B, True),
                    "span1": {f.id for sp in mem.recall_span(q, 1, B) for f in sp.frames},
                    "span2": {f.id for sp in mem.recall_span(q, 2, B) for f in sp.frames},
                }
                for name, ids in got.items():
                    frac = len(gold & ids) / len(gold)
                    for g in ("all", CATS[qa["category"]]):
                        rows[(name, B, g, "recall")].append(frac)
                        rows[(name, B, g, "hit")].append(float(frac == 1.0))
                    if B == budgets[-1]:
                        sizes[name].append(len(ids))
    return rows, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--embedder", default="onnx")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--budgets", type=int, nargs="+", default=[500, 1000, 2000, 4000])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=40)
    args = ap.parse_args()
    data = json.load(open(args.data))
    emb = CachedEmbedder(from_spec(args.embedder))
    cfg = EpisodeConfig(alpha=args.alpha, max_len=args.max_len, boundary_key="session")
    t0 = time.perf_counter()
    rows, sizes = run(data, emb, args.budgets, args.limit, cfg)
    systems = ["chunks", "chunks+ctx", "span1", "span2"]
    n = len(rows[("chunks", args.budgets[0], "all", "recall")])
    print(f"\nLoCoMo budget-matched retrieval, embedder={emb.name}, questions={n} ({time.perf_counter()-t0:.0f}s)")
    for metric, title in (("recall", "evidence-recall"), ("hit", "hit (all gold turns returned)")):
        for g in ("all", *CATS.values()):
            print(f"\n== {title} | {g} ==")
            print(f"{'system':<12}" + "".join(f"{'B='+str(b):>9}" for b in args.budgets))
            for s in systems:
                print(f"{s:<12}" + "".join(f"{np.mean(rows[(s, b, g, metric)]):>9.3f}" for b in args.budgets))
    print("\nmean frames returned at the largest budget:")
    for s in systems:
        print(f"  {s:<12}{np.mean(sizes[s]):.1f}")


if __name__ == "__main__":
    main()
