import os

import pytest

pytest.importorskip("cryptography")

from playmemory import Memory  # noqa: E402
from playmemory.cli import main  # noqa: E402
from playmemory.crypto import EncryptedFileError, WrongPasswordError  # noqa: E402
from playmemory.doctor import diagnose, repair  # noqa: E402
from playmemory.store import CorruptFileError, read_header  # noqa: E402

PW = "correct horse battery staple"
SECRET = "the launch code is swordfish42"


def make(path, n=20, **kw):
    with Memory(path, "debug", password=PW, **kw) as m:
        m.add_many([f"note {i} about topic{i % 4}" for i in range(n)] + [SECRET],
                   times=[float(i) for i in range(n + 1)], metas=[{"k": i % 3} for i in range(n + 1)])
        m.forget(2, t=100.0)
        m.checkpoint()
        m.add("tail record after checkpoint", t=200.0)


def test_nothing_readable_in_the_file(tmp_path):
    p = tmp_path / "e.pmem"
    make(p)
    raw = p.read_bytes()
    assert b"swordfish42" not in raw and b"topic1" not in raw and b"launch code" not in raw
    assert read_header(p)["encrypted"] and read_header(p)["spec"] == "hash-256"  # header stays readable


def test_roundtrip_search_checkpoint_tombstones_and_append(tmp_path):
    p = tmp_path / "e.pmem"
    make(p)
    with Memory(p, password=PW) as m:
        assert len(m) == 22 and m.stats()["encrypted"]
        assert m.search("swordfish42", k=1, mode="lexical")[0].frame.text == SECRET
        assert m.search("launch code", k=1, mode="vector")[0].frame.text == SECRET
        assert 2 in m.store.deleted and m.get(21).text == "tail record after checkpoint"
        assert m.find({"k": 1})  # metadata survives
        m.add("added later", t=300.0)
    with Memory(p, password=PW) as m2:
        assert len(m2) == 23 and m2.store.verify()[1] == 0


def test_requires_correct_password(tmp_path):
    p = tmp_path / "e.pmem"
    make(p)
    with pytest.raises(EncryptedFileError):
        Memory(p)
    with pytest.raises(WrongPasswordError):
        Memory(p, password="nope")
    with pytest.raises(ValueError, match="not encrypted"):
        q = tmp_path / "plain.pmem"
        Memory(q, "debug").close()
        Memory(q, password=PW)
    Memory(p, password=PW).close()  # failed attempts did not leave it locked


def test_tampering_is_detected(tmp_path):
    p = tmp_path / "e.pmem"
    make(p)
    with Memory(p, password=PW) as m:
        off = m.store._offsets[5]
    raw = bytearray(p.read_bytes())
    raw[off + 40] ^= 0x01  # flip a payload bit...
    import zlib
    n = int.from_bytes(raw[off:off + 4], "little")
    body = bytes(raw[off + 4: off + 4 + n])
    raw[off + 4 + n: off + 8 + n] = zlib.crc32(body).to_bytes(4, "little")  # ...and fix the CRC like an attacker would
    p.write_bytes(bytes(raw))
    with pytest.raises(CorruptFileError, match="authentication"):
        Memory(p, password=PW)


def test_records_cannot_be_swapped(tmp_path):
    p = tmp_path / "e.pmem"
    with Memory(p, "debug", password=PW) as m:
        m.add_many(["alpha record " * 3, "bravo record " * 3, "charlie record " * 3])
        o = list(m.store._offsets) + [m.store.nbytes]
    raw = p.read_bytes()
    a, b = raw[o[0]:o[1]], raw[o[1]:o[2]]
    if len(a) == len(b):
        p.write_bytes(raw[:o[0]] + b + a + raw[o[2]:])  # valid CRCs, wrong positions
        with pytest.raises(CorruptFileError):
            Memory(p, password=PW)


def test_compact_keeps_encryption_and_new_salt(tmp_path):
    p = tmp_path / "e.pmem"
    make(p)
    kdf_before = read_header(p)["kdf"]
    with Memory(p, password=PW) as m:
        m.compact()
        assert m.search("swordfish42", k=1, mode="lexical")
    assert read_header(p)["encrypted"] and read_header(p)["kdf"] != kdf_before
    assert b"swordfish42" not in p.read_bytes()


def test_encrypted_with_float16_and_ann(tmp_path):
    pytest.importorskip("faiss")
    p = tmp_path / "e.pmem"
    with Memory(p, "debug", password=PW, vec_dtype="float16", ann_threshold=50) as m:
        m.add_many([f"entry {i} topic{i % 7} word{i % 11}" for i in range(120)], times=[float(i) for i in range(120)])
        m.checkpoint()
        assert m.ann is not None
    with Memory(p, password=PW, ann_threshold=50) as m2:
        assert m2.ann is not None and len(m2.ann) == 120
        assert m2.search("entry 17 topic3 word6", k=1, mode="vector")[0].frame.id == 17


def test_doctor_on_encrypted_file(tmp_path):
    p = tmp_path / "e.pmem"
    make(p, n=30)
    with Memory(p, password=PW) as m:
        off = m.store._offsets[10]
    raw = bytearray(p.read_bytes())
    raw[off + 12] ^= 0xFF
    p.write_bytes(bytes(raw))
    rep = diagnose(p)  # damage map works without the password
    assert not rep.ok and len(rep.gaps) == 1
    with pytest.raises(EncryptedFileError):
        repair(p)
    rep = repair(p, password=PW)
    with Memory(rep.repaired_path, password=PW) as m2:
        assert len(m2) == rep.frames_salvaged and m2.search("swordfish42", k=1, mode="lexical")
    assert b"swordfish42" not in open(rep.repaired_path, "rb").read()


def test_cli_password_file_and_env(tmp_path, capsys, monkeypatch):
    f, pwf = str(tmp_path / "c.pmem"), tmp_path / "pw.txt"
    pwf.write_text(PW + "\n")
    assert main([f, "--embedder", "hash-256", "--password-file", str(pwf), "add", SECRET]) == 0
    capsys.readouterr()
    assert main([f, "--password-file", str(pwf), "search", "swordfish42", "--mode", "lexical", "-k", "1"]) == 0
    assert SECRET in capsys.readouterr().out
    with pytest.raises(EncryptedFileError):
        main([f, "stats"])
    monkeypatch.setenv("PLAYMEMORY_PASSWORD", PW)
    assert main([f, "stats"]) == 0 and '"encrypted": true' in capsys.readouterr().out
    assert main([f, "verify"]) == 0
    # a password in the environment must not break plain (unencrypted) files
    plain = str(tmp_path / "plain.pmem")
    monkeypatch.delenv("PLAYMEMORY_PASSWORD")
    main([plain, "--embedder", "hash-256", "add", "hello plain"])
    capsys.readouterr()
    monkeypatch.setenv("PLAYMEMORY_PASSWORD", PW)
    assert main([plain, "stats"]) == 0


# ---- signatures (no cryptography needed, but kept here with the other security tests) ----
def test_signature_detects_modification_and_reports_unsigned_tail(tmp_path):
    p = tmp_path / "s.pmem"
    key = b"signing-key"
    with Memory(p, "debug") as m:
        m.add_many(["one important fact", "another important fact"])
        assert m.verify_signature(key) == (False, m.store.nbytes - m.store._pos0)  # not signed yet
        m.sign(key)
        ok, tail = m.verify_signature(key)
        assert ok and tail == 0
        assert m.verify_signature(b"other key")[0] is False
        m.add("appended after signing")
        ok, tail = m.verify_signature(key)
        assert ok and tail > 0  # prefix still authentic; new data reported as unsigned
        m.sign(key)
        assert m.verify_signature(key) == (True, 0)
    raw = bytearray(p.read_bytes())
    i = raw.index(b"important fact")
    raw[i] ^= 0x20  # tamper with the signed content, keep CRCs consistent
    import zlib
    off = None
    with Memory(p, "debug", read_only=True) as m:
        off = m.store._offsets[0]
    n = int.from_bytes(raw[off:off + 4], "little")
    raw[off + 4 + n: off + 8 + n] = zlib.crc32(bytes(raw[off + 4: off + 4 + n])).to_bytes(4, "little")
    p.write_bytes(bytes(raw))
    with Memory(p, "debug", read_only=True) as m:
        assert m.verify_signature(key)[0] is False
