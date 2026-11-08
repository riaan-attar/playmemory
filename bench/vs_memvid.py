"""Head-to-head: PlayMemory vs Memvid (memvid-sdk) on LoCoMo retrieval, same embeddings.

Two environments are needed because the packages don't share a venv:

  1. (playmemory venv)  python -m bench.vs_memvid prepare  --data locomo10.json --out DIR
        embeds every turn and question once with BGE-small and writes DIR/*.npz + DIR/data.json
  2. (memvid venv: pip install memvid-sdk numpy)
        MEMVID_TELEMETRY=0 python bench/vs_memvid.py memvid --out DIR
        builds a .mv2 per conversation with the SAME vectors, runs every question, writes DIR/memvid.json
  3. (playmemory venv)  python -m bench.vs_memvid evaluate --out DIR
        runs PlayMemory on the same turns/vectors and prints recall/hit/latency/size/open-time tables

Metric: evidence-recall@k and hit@k of the gold dialogue turns (no neighbour context, one turn
per frame/document for both systems). Memvid's "sem" mode is its hybrid lex+vec engine.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
KS = (1, 5, 10)


def _load_turns(conv: dict):
    out = []
    sessions = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1]))
    for s in sessions:
        date = conv.get(f"{s}_date_time", "")
        for turn in conv[s]:
            text = turn["text"] + (f" [image: {turn['blip_caption']}]" if turn.get("blip_caption") else "")
            out.append((turn["dia_id"], f"[{date}] {turn['speaker']}: {text}"))
    return out


def prepare(args):
    import numpy as np

    from playmemory import from_spec

    emb = from_spec("onnx")
    data = json.load(open(args.data))[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for ci, sample in enumerate(data):
        turns = _load_turns(sample["conversation"])
        d2i = {d: i for i, (d, _) in enumerate(turns)}
        qs = []
        for qa in sample["qa"]:
            gold = sorted({d for ev in qa.get("evidence", []) for d in re.findall(r"D\d+:\d+", ev) if d in d2i})
            if gold and qa["category"] in CATS:
                qs.append({"q": qa["question"], "gold": gold, "cat": qa["category"]})
        t0 = time.perf_counter()
        tv = emb.embed([t for _, t in turns])
        qv = emb.embed([q["q"] for q in qs])
        np.savez(out / f"emb_{ci}.npz", turns=tv.astype("float32"), questions=qv.astype("float32"))
        meta.append({"id": sample["sample_id"], "turns": turns, "questions": qs})
        print(f"{sample['sample_id']}: {len(turns)} turns, {len(qs)} questions, {time.perf_counter()-t0:.0f}s", flush=True)
    json.dump({"embedder": emb.name, "dim": emb.dim, "convs": meta}, open(out / "data.json", "w"))


def memvid(args):
    os.environ.setdefault("MEMVID_TELEMETRY", "0")
    import memvid_sdk
    import numpy as np

    out = Path(args.out)
    D = json.load(open(out / "data.json"))
    res = {"version": getattr(memvid_sdk, "__version__", "2.0.160"), "convs": []}
    work = Path(tempfile.mkdtemp(prefix="mv2_"))
    for ci, c in enumerate(D["convs"]):
        E = np.load(out / f"emb_{ci}.npz")
        path = str(work / f"{c['id']}.mv2")
        t0 = time.perf_counter()
        m = memvid_sdk.create(path, enable_vec=True, enable_lex=True)
        docs = [{"title": d, "label": "turn", "text": t, "metadata": {"dia": d}} for d, t in c["turns"]]
        m.put_many(docs, embeddings=E["turns"].tolist(),
                   embedding_identity={"provider": "onnx", "model": D["embedder"], "dimension": D["dim"]})
        m.commit()
        build = time.perf_counter() - t0
        m.close()
        size = os.path.getsize(path)
        t0 = time.perf_counter()
        m = memvid_sdk.use("basic", path, mode="open", enable_vec=True, enable_lex=True)
        open_s = time.perf_counter() - t0
        row = {"id": c["id"], "build_s": build, "open_s": open_s, "bytes": size, "sem": [], "lex": [],
               "lat_sem": [], "lat_lex": [], "errors": {"sem": 0, "lex": 0}, "first_error": None}
        for qi, q in enumerate(c["questions"]):
            qv = E["questions"][qi].tolist()
            for mode, key in (("sem", "sem"), ("lex", "lex")):
                t0 = time.perf_counter()
                try:
                    r = m.find(q["q"], k=max(KS), mode=mode, query_embedding=qv, query_embedding_model=D["embedder"])
                    hits = [h.get("title") for h in r["hits"]]
                except Exception as e:  # counted as "no results" and reported, never hidden
                    hits = []
                    row["errors"][key] += 1
                    row["first_error"] = row["first_error"] or {"mode": mode, "query": q["q"], "error": f"{type(e).__name__}: {str(e)[:120]}"}
                row["lat_" + key].append(time.perf_counter() - t0)
                row[key].append(hits)
        m.close()
        res["convs"].append(row)
        print(f"{c['id']}: build {build:.1f}s open {open_s*1000:.0f}ms {size/1e6:.1f}MB errors={row['errors']} {row['first_error'] or ''}", flush=True)
    json.dump(res, open(out / "memvid.json", "w"))


class _Pre:
    """Embedder serving the precomputed question vectors (and turns via vectors=)."""

    def __init__(self, name, dim, qtexts, qvecs):
        self.name, self.spec, self.dim = name, name, dim
        self._m = dict(zip(qtexts, qvecs))

    def embed(self, texts):
        import numpy as np
        return np.vstack([self._m[t] for t in texts])


def _score(returned_dia: list[list[str]], questions, k):
    import numpy as np

    rec, hit = {}, {}
    for r, q in zip(returned_dia, questions):
        got, gold = set(r[:k]), set(q["gold"])
        frac = len(got & gold) / len(gold)
        for g in ("all", CATS[q["cat"]]):
            rec.setdefault(g, []).append(frac)
            hit.setdefault(g, []).append(float(frac == 1.0))
    return {g: float(np.mean(v)) for g, v in rec.items()}, {g: float(np.mean(v)) for g, v in hit.items()}


def evaluate(args):
    import numpy as np

    from playmemory import Memory

    out = Path(args.out)
    D = json.load(open(out / "data.json"))
    MV = json.load(open(out / "memvid.json")) if (out / "memvid.json").exists() else None
    systems: dict[str, list] = {"pm-lexical": [], "pm-vector": [], "pm-hybrid": []}
    lat = {k: [] for k in systems}
    pm = {"build_s": 0.0, "open_s": 0.0, "bytes": 0, "bytes_f16": 0}
    questions_all = []
    tmp = Path(tempfile.mkdtemp(prefix="pm_"))
    for ci, c in enumerate(D["convs"]):
        E = np.load(out / f"emb_{ci}.npz")
        qs = c["questions"]
        questions_all += qs
        emb = _Pre(D["embedder"], D["dim"], [q["q"] for q in qs], E["questions"])
        for variant, kw in (("", {}), ("_f16", {"vec_dtype": "float16"})):
            path = tmp / f"{c['id']}{variant}.pmem"
            t0 = time.perf_counter()
            m = Memory(path, emb, **kw)
            m.add_many([t for _, t in c["turns"]], metas=[{"dia": d} for d, _ in c["turns"]],
                       times=[float(i) for i in range(len(c["turns"]))], vectors=E["turns"])
            m.checkpoint()
            m.close()
            if not variant:
                pm["build_s"] += time.perf_counter() - t0
                pm["bytes"] += path.stat().st_size
                t0 = time.perf_counter()
                m = Memory(path, emb, read_only=True)
                pm["open_s"] += time.perf_counter() - t0
                for q in qs:
                    for mode in ("lexical", "vector", "hybrid"):
                        t1 = time.perf_counter()
                        hits = m.search(q["q"], k=max(KS), mode=mode)
                        lat["pm-" + mode].append(time.perf_counter() - t1)
                        systems["pm-" + mode].append([h.frame.meta["dia"] for h in hits])
                m.close()
            else:
                pm["bytes_f16"] += path.stat().st_size
    if MV:
        systems["memvid-sem(hybrid)"] = [r for row in MV["convs"] for r in row["sem"]]
        systems["memvid-lex"] = [r for row in MV["convs"] for r in row["lex"]]
        lat["memvid-sem(hybrid)"] = [x for row in MV["convs"] for x in row["lat_sem"]]
        lat["memvid-lex"] = [x for row in MV["convs"] for x in row["lat_lex"]]
    n = len(questions_all)
    print(f"\nLoCoMo head-to-head, {n} questions, embedder {D['embedder']} (same vectors for both)\n")
    for metric_i, title in ((0, "evidence-recall@k"), (1, "hit@k (all gold turns)")):
        for g in ("all", *CATS.values()):
            print(f"== {title} | {g} ==")
            print(f"{'system':<22}" + "".join(f"{'k='+str(k):>8}" for k in KS))
            for name, rets in systems.items():
                vals = [_score(rets, questions_all, k)[metric_i].get(g, float('nan')) for k in KS]
                print(f"{name:<22}" + "".join(f"{v:>8.3f}" for v in vals))
            print()
    print("search latency ms (p50 / p95), query vector precomputed:")
    for name, v in lat.items():
        print(f"  {name:<22}{np.percentile(v, 50)*1000:>8.2f} /{np.percentile(v, 95)*1000:>8.2f}")
    print("\nstorage (10 conversations):")
    print(f"  playmemory  {pm['bytes']/1e6:7.1f} MB (float32)  {pm['bytes_f16']/1e6:7.1f} MB (float16)  build {pm['build_s']:.1f}s  open {pm['open_s']*1000/len(D['convs']):.0f} ms/file")
    if MV:
        errs = {m: sum(r["errors"][m] for r in MV["convs"]) for m in ("sem", "lex")}
        print(f"  memvid query errors (counted as empty results): {errs}")
        print(f"  memvid      {sum(r['bytes'] for r in MV['convs'])/1e6:7.1f} MB                     build {sum(r['build_s'] for r in MV['convs']):.1f}s  open {sum(r['open_s'] for r in MV['convs'])*1000/len(MV['convs']):.0f} ms/file")
    json.dump({"systems": {k: v for k, v in systems.items()}, "n": n}, open(out / "eval.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("prepare", prepare), ("memvid", memvid), ("evaluate", evaluate)):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        if name == "prepare":
            p.add_argument("--data", required=True)
            p.add_argument("--limit", type=int, default=10)
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)
