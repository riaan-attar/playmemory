"""Minimal MCP (Model Context Protocol) server over stdio. Stdlib only.

    playmemory memory.pmem serve

Tools: remember, recall, forget, seek, timeline, profile, stats. Any MCP client
(Claude Desktop/Code, Cursor, ...) can then use the file as long-term memory.
"""
from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from .cards import Extractor, RuleExtractor, profile, remember
from .memory import Memory

PROTOCOL = "2024-11-05"


def _frame(f) -> dict:
    return {"id": f.id, "t": f.t, "text": f.text, "meta": f.meta}


def _tools() -> list[dict]:
    num, text, integer = {"type": "number"}, {"type": "string"}, {"type": "integer"}
    return [
        {"name": "remember", "description": "Store a memory (and extract structured facts from it).",
         "inputSchema": {"type": "object", "required": ["text"], "properties": {
             "text": text, "meta": {"type": "object"}, "extract_cards": {"type": "boolean", "default": True}}}},
        {"name": "recall", "description": "Search memory (hybrid keyword + semantic). Returns hits with surrounding timeline context.",
         "inputSchema": {"type": "object", "required": ["query"], "properties": {
             "query": text, "k": integer, "context": integer, "as_of": num,
             "mode": {"enum": ["hybrid", "vector", "lexical"]}, "where": {"type": "object"},
             "when": {"type": "string", "description": "\"auto\" or a phrase like \"last week\" / \"in May 2023\""}}}},
        {"name": "recall_story", "description": "Find the stretch(es) of the timeline that cover a topic and return them as "
         "contiguous spans: pointers (start_id, end_id, t0, t1, summary) plus the text. Prefer this for 'what happened with X' "
         "questions; use read_span to fetch more of a span.",
         "inputSchema": {"type": "object", "required": ["query"], "properties": {
             "query": text, "k": integer, "budget_chars": integer, "as_of": num, "when": text,
             "pointers_only": {"type": "boolean"}}}},
        {"name": "read_span", "description": "Press play on a span: the frames between two ids (inclusive).",
         "inputSchema": {"type": "object", "required": ["start_id", "end_id"], "properties": {
             "start_id": integer, "end_id": integer, "as_of": num}}},
        {"name": "forget", "description": "Forget a memory by id (history is kept for as_of queries).",
         "inputSchema": {"type": "object", "required": ["id"], "properties": {"id": integer}}},
        {"name": "seek", "description": "The memory in effect at a point in time.",
         "inputSchema": {"type": "object", "required": ["t"], "properties": {"t": num}}},
        {"name": "timeline", "description": "All memories between two times.",
         "inputSchema": {"type": "object", "required": ["t0", "t1"], "properties": {"t0": num, "t1": num}}},
        {"name": "profile", "description": "Current structured beliefs about a subject (default: the user).",
         "inputSchema": {"type": "object", "properties": {"subject": text, "as_of": num}}},
        {"name": "stats", "description": "Memory file statistics.", "inputSchema": {"type": "object", "properties": {}}},
    ]


class MemoryServer:
    def __init__(self, memory: Memory, extractor: Extractor | None = None):
        self.mem = memory
        self.extractor = extractor or RuleExtractor()

    # tool dispatch --------------------------------------------------------------
    def call(self, name: str, a: dict[str, Any]) -> Any:
        m = self.mem
        if name == "remember":
            if a.get("extract_cards", True):
                raw, cards = remember(m, a["text"], self.extractor, meta=a.get("meta"))
                out = {"id": raw.id, "cards": [_frame(c) for c in cards]}
            else:
                out = {"id": m.add(a["text"], a.get("meta")).id, "cards": []}
            m.commit()
            return out
        if name == "recall":
            hits = m.search(a["query"], a.get("k", 5), a.get("context", 0), mode=a.get("mode", "hybrid"),
                            as_of=a.get("as_of"), where=a.get("where"), when=a.get("when"))
            return [{"score": round(h.score, 4), "frame": _frame(h.frame),
                     "before": [_frame(f) for f in h.before], "after": [_frame(f) for f in h.after]}
                    for h in hits]
        if name == "recall_story":
            spans = m.recall_span(a["query"], a.get("k", 1), a.get("budget_chars", 4000), as_of=a.get("as_of"),
                                 when=a.get("when"))
            return [{"start_id": sp.start_id, "end_id": sp.end_id, "t0": sp.t0, "t1": sp.t1,
                     "scene_frames": sp.episode_len, "summary": sp.summary, "score": round(sp.score, 4),
                     **({} if a.get("pointers_only") else {"frames": [_frame(f) for f in sp.frames]})}
                    for sp in spans]
        if name == "read_span":
            return [_frame(f) for f in m.read(int(a["start_id"]), int(a["end_id"]), a.get("as_of"))]
        if name == "forget":
            m.forget(int(a["id"]))
            m.commit()
            return {"forgot": int(a["id"])}
        if name == "seek":
            f = m.seek(a["t"])
            return _frame(f) if f else None
        if name == "timeline":
            return [_frame(f) for f in m.window(a["t0"], a["t1"])]
        if name == "profile":
            return profile(m, a.get("subject", "user"), a.get("as_of"))
        if name == "stats":
            return m.stats()
        raise KeyError(f"unknown tool {name!r}")

    # json-rpc -------------------------------------------------------------------
    def handle(self, msg: dict) -> dict | None:
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:  # notification
            return None
        try:
            if method == "initialize":
                result = {"protocolVersion": PROTOCOL, "capabilities": {"tools": {}},
                          "serverInfo": {"name": "playmemory", "version": "0.3.0"}}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": _tools()}
            elif method == "tools/call":
                p = msg.get("params", {})
                try:
                    data = self.call(p["name"], p.get("arguments") or {})
                    result = {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]}
                except Exception as e:  # tool errors are reported in-band
                    result = {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
            return {"jsonrpc": "2.0", "id": mid, "result": result}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(e)}}

    def serve(self, rfile: TextIO = sys.stdin, wfile: TextIO = sys.stdout) -> None:
        for line in rfile:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                wfile.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                        "error": {"code": -32700, "message": "parse error"}}) + "\n")
                wfile.flush()
                continue
            resp = self.handle(msg)
            if resp is not None:
                wfile.write(json.dumps(resp, ensure_ascii=False) + "\n")
                wfile.flush()
