"""File loaders -> list[Block]. Text, Markdown, HTML, CSV/TSV and JSON/JSONL need
nothing extra; PDF/DOCX/XLSX/PPTX use the optional `docs` extra, imported lazily."""
from __future__ import annotations

import csv
import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path

from .chunker import Block

TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".log", ".text"}
HTML_EXT = {".html", ".htm"}
TABLE_EXT = {".csv", ".tsv"}
JSON_EXT = {".json", ".jsonl", ".ndjson"}
DOC_EXT = {".pdf", ".docx", ".xlsx", ".pptx"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".mkv", ".webm", ".mov"}
SUPPORTED = TEXT_EXT | HTML_EXT | TABLE_EXT | JSON_EXT | DOC_EXT


class MissingDependency(ImportError):
    pass


def _need(mod: str, extra: str = "docs"):
    try:
        return __import__(mod)
    except ImportError as e:
        raise MissingDependency(f"reading this file needs `{mod}`: pip install 'playmemory[{extra}]'") from e


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):  # UTF-16 only when a BOM says so
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeError:
        return raw.decode("cp1252", errors="replace")


_MD_HEAD = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def load_text(path: Path) -> list[Block]:
    blocks, para, in_table, tbl = [], [], False, []

    def flush_para():
        if para:
            blocks.append(Block("para", " ".join(para)))
            para.clear()

    def flush_tbl():
        if tbl:
            blocks.append(Block("table", "\n".join(tbl)))
            tbl.clear()

    for line in _read_text(path).splitlines():
        m = _MD_HEAD.match(line)
        if line.lstrip().startswith("|"):
            flush_para()
            tbl.append(line.strip())
            continue
        flush_tbl()
        if m:
            flush_para()
            blocks.append(Block("heading", m[2], level=len(m[1])))
        elif not line.strip():
            flush_para()
        else:
            para.append(line.strip())
    flush_para()
    flush_tbl()
    return blocks


class _HTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._buf: list[str] = []
        self._tag: str | None = None
        self._skip = 0
        self._row: list[str] | None = None
        self._rows: list[str] = []

    def _flush(self):
        text = " ".join("".join(self._buf).split())
        self._buf = []
        if text and self._row is None:
            if self._tag and self._tag[0] == "h" and self._tag[1:].isdigit():
                self.blocks.append(Block("heading", text, level=int(self._tag[1:])))
            else:
                self.blocks.append(Block("para", text))

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip += 1
        elif tag in ("p", "div", "li", "br", "section", "article", "ul", "ol") or (tag[0] == "h" and tag[1:].isdigit()):
            self._flush()
            self._tag = tag
        elif tag == "table":
            self._flush()
            self._rows = []
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._buf = []

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip = max(self._skip - 1, 0)
        elif tag in ("td", "th") and self._row is not None:
            self._row.append(" ".join("".join(self._buf).split()))
            self._buf = []
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self._rows.append("| " + " | ".join(self._row) + " |")
            self._row = None
        elif tag == "table":
            if self._rows:
                self.blocks.append(Block("table", "\n".join(self._rows)))
            self._rows = []
        elif tag in ("p", "div", "li", "section", "article") or (tag[0] == "h" and tag[1:].isdigit()):
            self._flush()
            self._tag = None

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data)

    def close(self):
        super().close()
        self._flush()


def load_html(path: Path) -> list[Block]:
    p = _HTML()
    p.feed(_read_text(path))
    p.close()
    return p.blocks


def load_table(path: Path, rows_per_block: int = 20) -> list[Block]:
    text = _read_text(path)
    delim = "\t" if path.suffix.lower() == ".tsv" else (csv.Sniffer().sniff(text[:2048], ",;\t|").delimiter if text.strip() else ",")
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    return _rows_to_blocks(rows, rows_per_block, {})


def _rows_to_blocks(rows: list[list], per: int, extra: dict, page=None) -> list[Block]:
    rows = [[("" if c is None else str(c)) for c in r] for r in rows if any(c not in (None, "") for c in r)]
    if not rows:
        return []
    head = "| " + " | ".join(rows[0]) + " |"
    out = []
    for i in range(1, len(rows), per):
        body = ["| " + " | ".join(r) + " |" for r in rows[i : i + per]]
        out.append(Block("table", "\n".join([head, *body]), page=page, extra=dict(extra)))
    return out or [Block("table", head, page=page, extra=dict(extra))]


def load_json(path: Path) -> list[Block]:
    text = _read_text(path)
    items = []
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        items = data if isinstance(data, list) else [data]
    else:
        items = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    blocks = []
    for it in items:
        if isinstance(it, dict):
            body = it.get("text") or it.get("content") or it.get("body")
            body = body if isinstance(body, str) else json.dumps(it, ensure_ascii=False)
        else:
            body = it if isinstance(it, str) else json.dumps(it, ensure_ascii=False)
        blocks.append(Block("para", body))
    return blocks


def load_pdf(path: Path) -> list[Block]:
    pypdf = _need("pypdf")
    blocks = []
    for i, page in enumerate(pypdf.PdfReader(str(path)).pages, start=1):
        text = page.extract_text() or ""
        for para in re.split(r"\n\s*\n", text):
            para = " ".join(para.split())
            if para:
                blocks.append(Block("para", para, page=i))
    return blocks


def load_docx(path: Path) -> list[Block]:
    docx = _need("docx")
    d = docx.Document(str(path))
    blocks: list[Block] = []
    body = d.element.body
    paras = {p._p: p for p in d.paragraphs}
    tables = {t._tbl: t for t in d.tables}
    for el in body.iterchildren():
        if el in paras:
            p = paras[el]
            style = (p.style.name or "") if p.style is not None else ""
            m = re.match(r"Heading (\d)", style)
            if m:
                blocks.append(Block("heading", p.text, level=int(m[1])))
            elif style == "Title":
                blocks.append(Block("heading", p.text, level=1))
            elif p.text.strip():
                blocks.append(Block("para", p.text))
        elif el in tables:
            rows = [[c.text.strip() for c in r.cells] for r in tables[el].rows]
            blocks += _rows_to_blocks(rows, 20, {})
    return blocks


def load_xlsx(path: Path) -> list[Block]:
    openpyxl = _need("openpyxl")
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    blocks = []
    for ws in wb.worksheets:
        blocks.append(Block("heading", ws.title, level=1))
        blocks += _rows_to_blocks([list(r) for r in ws.iter_rows(values_only=True)], 20, {"sheet": ws.title})
    return blocks


def load_pptx(path: Path) -> list[Block]:
    pptx = _need("pptx")
    blocks = []
    for i, slide in enumerate(pptx.Presentation(str(path)).slides, start=1):
        title = slide.shapes.title.text.strip() if slide.shapes.title is not None and slide.shapes.title.has_text_frame else ""
        if title:
            blocks.append(Block("heading", title, level=1, page=i))
        for sh in slide.shapes:
            if sh.has_text_frame and sh is not slide.shapes.title:
                text = " ".join(p.text for p in sh.text_frame.paragraphs if p.text.strip())
                if text:
                    blocks.append(Block("para", text, page=i))
            elif getattr(sh, "has_table", False) and sh.has_table:
                rows = [[c.text.strip() for c in r.cells] for r in sh.table.rows]
                blocks += _rows_to_blocks(rows, 20, {}, page=i)
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            blocks.append(Block("para", "Notes: " + slide.notes_slide.notes_text_frame.text.strip(), page=i))
    return blocks


def load(path: str | Path) -> list[Block]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in TEXT_EXT:
        return load_text(p)
    if ext in HTML_EXT:
        return load_html(p)
    if ext in TABLE_EXT:
        return load_table(p)
    if ext in JSON_EXT:
        return load_json(p)
    return {".pdf": load_pdf, ".docx": load_docx, ".xlsx": load_xlsx, ".pptx": load_pptx}[ext](p)
