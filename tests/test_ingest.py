import json
import os

import pytest

from playmemory import Memory
from playmemory.cli import main
from playmemory.ingest.chunker import Block, chunk_blocks
from playmemory.ingest.loaders import MissingDependency, load
from playmemory.server import MemoryServer


def mem():
    return Memory(embedder="debug")


# ---- chunker -------------------------------------------------------------------
def test_headings_paths_sections_and_boundaries():
    blocks = [Block("heading", "Guide", 1), Block("para", "intro text"),
              Block("heading", "Install", 2), Block("para", "run pip install"),
              Block("heading", "Usage", 2), Block("para", "call search"),
              Block("heading", "Other", 1), Block("para", "unrelated")]
    ch = chunk_blocks(blocks, source="doc.md")
    assert [c.meta["heading"] for c in ch] == ["Guide", "Guide > Install", "Guide > Usage", "Other"]
    assert len({c.meta["session"] for c in ch}) == 4  # one scene boundary per section
    assert [c.meta["chunk"] for c in ch] == [0, 1, 2, 3]


def test_paragraph_packing_and_long_paragraph_split():
    short = [Block("para", f"sentence number {i} is short.") for i in range(10)]
    ch = chunk_blocks(short, max_chars=100)
    assert 1 < len(ch) < 10 and all(len(c.text) <= 130 for c in ch)
    long = [Block("para", " ".join(f"This is sentence {i}." for i in range(60)))]
    ch = chunk_blocks(long, max_chars=200, overlap=30)
    assert len(ch) > 5 and all(len(c.text) <= 260 for c in ch)
    nospace = chunk_blocks([Block("para", "x" * 1000)], max_chars=300, overlap=50)
    assert len(nospace) >= 4 and sum(len(c.text) for c in nospace) >= 1000  # hard split with overlap


def test_table_rows_stay_together_and_header_repeats():
    rows = "\n".join(["| name | qty |"] + [f"| item{i} | {i} |" for i in range(40)])
    ch = chunk_blocks([Block("table", rows)], max_chars=200)
    assert len(ch) > 2 and all(c.text.startswith("| name | qty |") for c in ch)
    assert all(c.meta["kind"] == "table" for c in ch)
    small = chunk_blocks([Block("table", "| a | b |\n| 1 | 2 |")])
    assert len(small) == 1


def test_page_change_flushes_chunk():
    ch = chunk_blocks([Block("para", "page one text", page=1), Block("para", "page two text", page=2)])
    assert [c.meta["page"] for c in ch] == [1, 2]


# ---- stdlib loaders ------------------------------------------------------------
def test_markdown_html_csv_json_loaders(tmp_path):
    (tmp_path / "a.md").write_text("# Title\n\nHello world.\nSecond line.\n\n## Sub\n\n| a | b |\n| 1 | 2 |\n")
    b = load(tmp_path / "a.md")
    assert [x.kind for x in b] == ["heading", "para", "heading", "table"]
    assert b[1].text == "Hello world. Second line."

    (tmp_path / "p.html").write_text("<html><head><style>x{}</style></head><body><h1>Head</h1>"
                                      "<p>Para <b>bold</b> text</p><script>evil()</script>"
                                      "<table><tr><th>k</th><th>v</th></tr><tr><td>x</td><td>1</td></tr></table></body></html>")
    b = load(tmp_path / "p.html")
    assert (b[0].kind, b[0].text) == ("heading", "Head")
    assert b[1].text == "Para bold text" and not any("evil" in x.text for x in b)
    assert b[-1].kind == "table" and "| k | v |" in b[-1].text

    (tmp_path / "d.csv").write_text("name,city\nann,paris\nbob,berlin\n")
    b = load(tmp_path / "d.csv")
    assert b[0].kind == "table" and "| ann | paris |" in b[0].text and b[0].text.startswith("| name | city |")

    (tmp_path / "x.jsonl").write_text('{"text": "first note"}\n{"content": "second note"}\n{"k": 1}\n')
    assert [x.text for x in load(tmp_path / "x.jsonl")][:2] == ["first note", "second note"]
    (tmp_path / "y.json").write_text(json.dumps(["a plain string", {"body": "from body"}]))
    assert [x.text for x in load(tmp_path / "y.json")] == ["a plain string", "from body"]


def test_utf16_and_latin1_text(tmp_path):
    (tmp_path / "u.txt").write_bytes("héllo 東京".encode("utf-16"))
    assert load(tmp_path / "u.txt")[0].text == "héllo 東京"
    (tmp_path / "l.txt").write_bytes("café".encode("latin-1"))
    assert load(tmp_path / "l.txt")[0].text == "café"


# ---- optional document formats (real files built with the same libs) -----------
def test_docx_roundtrip(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Report", 1)
    d.add_paragraph("Revenue grew in the third quarter.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text, t.cell(1, 0).text, t.cell(1, 1).text = "region", "sales", "emea", "42"
    d.add_heading("Outlook", 2)
    d.add_paragraph("Expect more.")
    d.save(tmp_path / "r.docx")
    b = load(tmp_path / "r.docx")
    assert [(x.kind, x.text) for x in b][:2] == [("heading", "Report"), ("para", "Revenue grew in the third quarter.")]
    assert any(x.kind == "table" and "| emea | 42 |" in x.text for x in b)
    assert b[-2].kind == "heading" and b[-2].level == 2


def test_xlsx_and_pptx(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Q3"
    ws.append(["name", "amount"])
    ws.append(["acme", 1200])
    wb.save(tmp_path / "s.xlsx")
    b = load(tmp_path / "s.xlsx")
    assert b[0].kind == "heading" and b[0].text == "Q3"
    assert b[1].kind == "table" and "| acme | 1200 |" in b[1].text and b[1].extra == {"sheet": "Q3"}

    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Roadmap"
    slide.placeholders[1].text = "Ship the ingest feature"
    slide.notes_slide.notes_text_frame.text = "mention docs"
    prs.save(tmp_path / "d.pptx")
    b = load(tmp_path / "d.pptx")
    assert (b[0].kind, b[0].text, b[0].page) == ("heading", "Roadmap", 1)
    assert any("Ship the ingest feature" in x.text for x in b) and any(x.text.startswith("Notes:") for x in b)


def _pdf(path, text):
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    x = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offs)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, x)
    path.write_bytes(out)


def test_pdf_pages(tmp_path):
    pytest.importorskip("pypdf")
    _pdf(tmp_path / "p.pdf", "Quarterly results were strong")
    b = load(tmp_path / "p.pdf")
    assert b and b[0].page == 1 and "Quarterly results were strong" in b[0].text


def test_missing_optional_dependency_message(tmp_path, monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **kw):
        if name == "pypdf":
            raise ImportError("no pypdf")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake)
    (tmp_path / "x.pdf").write_bytes(b"%PDF")
    with pytest.raises(MissingDependency, match=r"playmemory\[docs\]"):
        load(tmp_path / "x.pdf")


# ---- Memory.ingest -------------------------------------------------------------
def test_ingest_directory_search_and_scene_boundaries(tmp_path):
    (tmp_path / "setup.md").write_text("# Setup\n\nInstall with pip install playmemory.\n\n# Usage\n\nCall search to recall.\n")
    (tmp_path / "notes.txt").write_text("Meeting about the zebra project on Friday.")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "t.csv").write_text("k,v\na,1\n")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.txt").write_text("should be skipped")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    m = mem()
    rep = m.ingest(tmp_path)
    assert rep.files == 3 and rep.unsupported == [str(tmp_path / "image.png")] and not rep.failed
    assert m.search("zebra project", k=1, mode="lexical")[0].frame.meta["source"].endswith("notes.txt")
    h = m.search("pip install", k=1, mode="lexical")[0].frame
    assert h.meta["heading"] == "Setup" and "session" in h.meta
    assert not any("secret" in m.get(i).text for i in range(len(m)))
    starts = [a for a, _ in m.episode_list()]
    assert len(starts) >= 3  # sections/files became separate scenes


def test_ingest_is_idempotent_and_replaces_changed_files(tmp_path):
    f = tmp_path / "n.txt"
    f.write_text("original zebra content")
    m = mem()
    assert m.ingest(f).chunks == 1
    rep = m.ingest(f)
    assert rep.skipped_unchanged and rep.chunks == 0 and len(m) == 1
    f.write_text("revised giraffe content")
    rep = m.ingest(f)
    assert rep.replaced and rep.chunks == 1 and len(m) == 2
    assert m.search("zebra", k=3, mode="lexical") == []  # old chunk forgotten
    assert m.search("giraffe", k=1, mode="lexical")[0].frame.text == "revised giraffe content"
    past = m.search("zebra", k=3, mode="lexical", as_of=m.get(0).t + 1e-9)
    assert past and past[0].frame.id == 0  # ...but still visible by time travel
    f.write_text("third version")
    rep = m.ingest(f, on_change="append")
    assert not rep.replaced and len(m.find({"source": str(f.resolve())})) == 2


def test_ingest_one_bad_file_does_not_abort(tmp_path):
    (tmp_path / "good.txt").write_text("good file text")
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "doc.pdf").write_bytes(b"not really a pdf")
    m = mem()
    rep = m.ingest(tmp_path)
    assert rep.files == 1 and "bad.json" in " ".join(rep.failed) and "doc.pdf" in " ".join(rep.failed)


def test_ingest_audio_frames_use_transcriber_offsets(tmp_path):
    (tmp_path / "talk.mp3").write_bytes(b"fake audio")
    segs = [(0.0, 4.0, "welcome everyone"), (4.0, 9.5, "today we discuss zebras"), (9.5, 12.0, "   ")]
    m = mem()
    assert m.ingest(tmp_path / "talk.mp3").unsupported  # no transcriber -> reported, not crashed
    rep = m.ingest(tmp_path / "talk.mp3", transcriber=lambda p: segs)
    assert rep.chunks == 2
    f = m.search("zebras", k=1, mode="lexical")[0].frame
    assert f.meta["kind"] == "audio" and (f.meta["start"], f.meta["end"]) == (4.0, 9.5)
    assert m.get(1).t - m.get(0).t == pytest.approx(4.0)


def test_ingest_into_memory_with_future_frames(tmp_path):
    (tmp_path / "a.txt").write_text("later file")
    m = mem()
    m.add("future frame", t=4e12)
    assert m.ingest(tmp_path / "a.txt").chunks == 1  # timeline stays monotonic


# ---- CLI + MCP -----------------------------------------------------------------
def test_cli_ingest(tmp_path, capsys):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# A\n\nalpha text here\n")
    f = str(tmp_path / "m.pmem")
    assert main([f, "--embedder", "hash-256", "ingest", str(d)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["files"] == 1 and out["chunks"] == 1
    assert main([f, "ingest", str(d)]) == 0
    assert json.loads(capsys.readouterr().out)["skipped_unchanged"]
    assert main([f, "search", "alpha", "--mode", "lexical", "-k", "1"]) == 0
    assert "alpha text here" in capsys.readouterr().out


def test_mcp_ingest_is_root_restricted(tmp_path):
    from test_server import call, rpc

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "n.txt").write_text("mcp ingested zebra note")
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    plain = MemoryServer(mem())
    assert "ingest" not in {t["name"] for t in rpc(plain, "tools/list")["result"]["tools"]}
    r = rpc(plain, "tools/call", {"name": "ingest", "arguments": {"path": str(allowed / "n.txt")}})["result"]
    assert r["isError"] and "disabled" in r["content"][0]["text"]
    srv = MemoryServer(mem(), ingest_roots=[str(allowed)])
    assert "ingest" in {t["name"] for t in rpc(srv, "tools/list")["result"]["tools"]}
    res, err = call(srv, "ingest", path=str(allowed))
    assert not err and res["chunks"] == 1
    r = rpc(srv, "tools/call", {"name": "ingest", "arguments": {"path": str(outside)}})["result"]
    assert r["isError"] and "outside the allowed" in r["content"][0]["text"]
    r = rpc(srv, "tools/call", {"name": "ingest", "arguments": {"path": str(allowed / ".." / "outside.txt")}})["result"]
    assert r["isError"]  # path traversal is resolved before the check
