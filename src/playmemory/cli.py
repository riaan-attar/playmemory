"""playmemory CLI:  playmemory FILE COMMAND ...   (see `playmemory --help`)."""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

from .embed import HashEmbedder, from_spec
from .memory import Memory
from .store import FrameStore


def _peek_embedder_name(path: str) -> tuple[str, int]:
    import struct

    with open(path, "rb") as f:
        head = f.read(10)
        magic, _ver, dim, nlen = struct.unpack("<4sHHH", head)
        return f.read(nlen).decode(), dim


def _open(args, read_only=False) -> Memory:
    spec = args.embedder
    if spec is None:
        try:
            name, dim = _peek_embedder_name(args.file)
        except FileNotFoundError:
            name, dim = "hash-256", 256
        spec = name if name.startswith("hash") else name.replace("onnx:", "onnx:", 1)
    emb = from_spec(spec)
    return Memory(args.file, emb, read_only=read_only)


def _frame_dict(f, score=None):
    d = {"id": f.id, "t": f.t, "text": f.text, "meta": f.meta}
    if score is not None:
        d["score"] = round(score, 5)
    return d


def _print(obj):
    print(json.dumps(obj, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="playmemory", description="Timeline memory for AI systems")
    ap.add_argument("file", help="memory file (.pmem)")
    ap.add_argument("--embedder", help="hash[-DIM] | onnx[:REPO] | st[:MODEL] | http:URL|MODEL[|KEY] "
                    "(default: the one the file was created with)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="add text (args, or lines from stdin)")
    a.add_argument("text", nargs="*")
    a.add_argument("--meta", help="JSON object")
    a.add_argument("--t", type=float)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--context", type=int, default=0)
    s.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "lexical"])
    s.add_argument("--as-of", type=float)
    s.add_argument("--where", help="JSON object")
    s.add_argument("--recency", type=float, help="half-life in seconds")
    k = sub.add_parser("seek"); k.add_argument("t", type=float)
    w = sub.add_parser("window"); w.add_argument("t0", type=float); w.add_argument("t1", type=float)
    f = sub.add_parser("forget"); f.add_argument("id", type=int)
    sub.add_parser("stats")
    sub.add_parser("verify")
    sub.add_parser("checkpoint")
    c = sub.add_parser("compact"); c.add_argument("--keep-history", action="store_true")
    sub.add_parser("export", help="dump all frames as JSONL")
    sub.add_parser("serve", help="run an MCP server on stdio")
    args = ap.parse_args(argv)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.cmd == "verify":
            st = FrameStore(args.file, _peek_embedder_name(args.file)[1], read_only=True)
            records, bad = st.verify()
            _print({"records": records, "bad_bytes": bad, "ok": bad == 0})
            return 0 if bad == 0 else 2
        read_only = args.cmd in ("search", "seek", "window", "stats", "export")
        m = _open(args, read_only)
    try:
        if args.cmd == "add":
            texts = args.text or [ln.rstrip("\n") for ln in sys.stdin if ln.strip()]
            meta = json.loads(args.meta) if args.meta else None
            times = [args.t] * len(texts) if args.t is not None else None
            frames = m.add_many(texts, [meta] * len(texts) if meta else None, times)
            m.commit()
            _print({"added": [f.id for f in frames]})
        elif args.cmd == "search":
            hits = m.search(args.query, args.k, args.context, mode=args.mode, as_of=args.as_of,
                            where=json.loads(args.where) if args.where else None,
                            recency_half_life=args.recency)
            for h in hits:
                d = _frame_dict(h.frame, h.score)
                if args.context:
                    d["before"] = [_frame_dict(x) for x in h.before]
                    d["after"] = [_frame_dict(x) for x in h.after]
                _print(d)
        elif args.cmd == "seek":
            fr = m.seek(args.t)
            _print(_frame_dict(fr) if fr else None)
        elif args.cmd == "window":
            for fr in m.window(args.t0, args.t1):
                _print(_frame_dict(fr))
        elif args.cmd == "forget":
            m.forget(args.id)
            m.commit()
            _print({"forgot": args.id})
        elif args.cmd == "stats":
            _print(m.stats())
        elif args.cmd == "checkpoint":
            m.checkpoint()
            _print(m.stats())
        elif args.cmd == "compact":
            _print(m.compact(keep_history=args.keep_history))
        elif args.cmd == "serve":
            from .server import MemoryServer

            MemoryServer(m).serve()
        elif args.cmd == "export":
            for i in range(len(m)):
                fr = m.get(i)
                d = _frame_dict(fr)
                d["deleted_at"] = m.store.deleted.get(i)
                _print(d)
    finally:
        m.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
