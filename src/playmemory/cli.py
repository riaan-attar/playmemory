"""playmemory CLI:  playmemory FILE COMMAND ...   (see `playmemory --help`)."""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

from .embed import HashEmbedder, from_spec
from .memory import Memory
from .store import FrameStore, read_header


def _open(args, read_only=False) -> Memory:
    emb = from_spec(args.embedder, args.api_key) if args.embedder else None
    if emb is None:
        try:
            emb = from_spec(read_header(args.file)["spec"], args.api_key)
        except FileNotFoundError:  # brand-new file: need an embedder
            import os

            if os.environ.get("PLAYMEMORY_ALLOW_DEBUG_EMBEDDER") == "1":
                emb = HashEmbedder()
            else:
                try:
                    import onnxruntime  # noqa: F401
                except ImportError:
                    raise SystemExit("new file: pass --embedder (e.g. http:URL|MODEL) or "
                                     "pip install 'playmemory[onnx]' to use the local default")
                emb = from_spec("onnx")
    return Memory(args.file, emb, read_only=read_only, password=_password(args))


def _password(args, existing: bool = True):
    """Password from --password-file / PLAYMEMORY_PASSWORD. For an existing file it is only
    used when the file is actually encrypted."""
    import os

    pw = open(args.password_file).read().rstrip("\r\n") if args.password_file else os.environ.get("PLAYMEMORY_PASSWORD")
    if pw is not None and existing:
        try:
            if not read_header(args.file)["encrypted"]:
                return None
        except FileNotFoundError:
            pass
    return pw


def _reranker():
    from .rerank import OnnxCrossEncoder

    return OnnxCrossEncoder()


def _cutoff(spec):
    if not spec or spec == "elbow":
        return spec or None
    kind, _, x = spec.partition(":")
    return (kind, float(x))


def _synth(spec: str):
    from .ask import ClaudeSynthesizer, OpenAICompatSynthesizer

    kind, _, rest = spec.partition(":")
    if kind == "claude":
        return ClaudeSynthesizer(rest) if rest else ClaudeSynthesizer()
    if kind == "openai":
        model, _, url = rest.partition("@")
        return OpenAICompatSynthesizer(model, url) if url else OpenAICompatSynthesizer(model)
    raise SystemExit(f"unknown --llm {spec!r}")


def _transcriber(model):
    if not model:
        return None
    from .ingest.audio import WhisperTranscriber

    return WhisperTranscriber(model)


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
    ap.add_argument("--api-key", default=None, help="key for http: embedders (or env PLAYMEMORY_API_KEY)")
    ap.add_argument("--password-file", help="file holding the encryption password (also env PLAYMEMORY_PASSWORD)")
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
    s.add_argument("--when", help='time filter: "auto" (read it from the query), or a phrase like "last week"')
    s.add_argument("--rerank", action="store_true", help="rescore top candidates with the local ONNX cross-encoder")
    s.add_argument("--cutoff", help='adaptive count: elbow | ratio:0.5 | cliff:0.4 | min:0.3')
    q = sub.add_parser("ask", help="answer a question from memory with an LLM, with citations")
    q.add_argument("question")
    q.add_argument("--llm", default="claude", help="claude[:MODEL] (ANTHROPIC_API_KEY) | openai:MODEL[@URL] (OPENAI_API_KEY)")
    q.add_argument("-k", type=int, default=8)
    q.add_argument("--budget", type=int, default=6000)
    q.add_argument("--spans", action="store_true", help="give the model contiguous scenes instead of top-k hits")
    q.add_argument("--when")
    st = sub.add_parser("story", help="return the timeline span(s) that cover a query")
    st.add_argument("query")
    st.add_argument("-k", type=int, default=1, help="number of spans")
    st.add_argument("--budget", type=int, default=4000, help="max characters")
    st.add_argument("--as-of", type=float)
    st.add_argument("--pointers", action="store_true", help="only print start/end ids, not text")
    rd = sub.add_parser("read", help="read a span by frame ids (inclusive)")
    rd.add_argument("start", type=int); rd.add_argument("end", type=int)
    k = sub.add_parser("seek"); k.add_argument("t", type=float)
    w = sub.add_parser("window"); w.add_argument("t0", type=float); w.add_argument("t1", type=float)
    f = sub.add_parser("forget"); f.add_argument("id", type=int)
    sub.add_parser("stats")
    sub.add_parser("verify")
    dr = sub.add_parser("doctor", help="diagnose damage; with --write salvage intact records")
    dr.add_argument("--write", action="store_true", help="write FILE.repaired")
    dr.add_argument("--replace", action="store_true", help="with --write: keep the damaged file as FILE.corrupt and replace it")
    sub.add_parser("checkpoint")
    c = sub.add_parser("compact"); c.add_argument("--keep-history", action="store_true")
    sub.add_parser("export", help="dump all frames as JSONL")
    ig = sub.add_parser("ingest", help="chunk and add files / directories")
    ig.add_argument("paths", nargs="+")
    ig.add_argument("--max-chars", type=int, default=800)
    ig.add_argument("--overlap", type=int, default=100)
    ig.add_argument("--transcribe", metavar="MODEL", help="transcribe audio/video with faster-whisper (e.g. base)")
    ig.add_argument("--append", action="store_true", help="re-add changed files instead of replacing their old chunks")
    sv = sub.add_parser("serve", help="run an MCP server on stdio")
    sv.add_argument("--ingest-root", action="append", default=[],
                    help="allow the MCP `ingest` tool to read files under this directory (repeatable)")
    args = ap.parse_args(argv)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.cmd == "verify":
            st = FrameStore(args.file, read_header(args.file)["dim"], read_only=True, password=_password(args))
            records, bad = st.verify()
            _print({"records": records, "bad_bytes": bad, "ok": bad == 0})
            return 0 if bad == 0 else 2
        if args.cmd == "doctor":
            from dataclasses import asdict

            from .doctor import diagnose, repair

            pw = _password(args)
            rep = (repair(args.file, replace=args.replace, password=pw) if args.write or args.replace
                   else diagnose(args.file, password=pw))
            _print(asdict(rep))
            return 0 if rep.ok else 2
        read_only = args.cmd in ("search", "ask", "story", "read", "seek", "window", "stats", "export")
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
                            recency_half_life=args.recency, when=args.when,
                            rerank=_reranker() if args.rerank else None, cutoff=_cutoff(args.cutoff))
            for h in hits:
                d = _frame_dict(h.frame, h.score)
                if args.context:
                    d["before"] = [_frame_dict(x) for x in h.before]
                    d["after"] = [_frame_dict(x) for x in h.after]
                _print(d)
        elif args.cmd == "ingest":
            from dataclasses import asdict

            rep = m.ingest(args.paths, max_chars=args.max_chars, overlap=args.overlap,
                           on_change="append" if args.append else "replace",
                           transcriber=_transcriber(args.transcribe))
            m.commit()
            d = asdict(rep)
            d.pop("frame_ids")
            _print(d)
        elif args.cmd == "ask":
            ans = m.ask(args.question, _synth(args.llm), k=args.k, budget_chars=args.budget,
                        spans=args.spans, when=args.when)
            _print({"answer": ans.text, "citations": [_frame_dict(f) for f in ans.citations],
                    "context_frames": [f.id for f in ans.context]})
        elif args.cmd == "story":
            for sp in m.recall_span(args.query, args.k, args.budget, as_of=args.as_of):
                d = {"start_id": sp.start_id, "end_id": sp.end_id, "t0": sp.t0, "t1": sp.t1,
                     "scene_frames": sp.episode_len, "summary": sp.summary, "score": round(sp.score, 4)}
                if not args.pointers:
                    d["frames"] = [_frame_dict(f) for f in sp.frames]
                _print(d)
        elif args.cmd == "read":
            for fr in m.read(args.start, args.end):
                _print(_frame_dict(fr))
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

            MemoryServer(m, ingest_roots=args.ingest_root).serve()
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
