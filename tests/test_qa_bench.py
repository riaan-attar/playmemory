import json

import pytest

from bench.qa import SYSTEMS, f1_and_contains, frames_for, report, run
from playmemory import EpisodeConfig, Memory

CONV = {"speaker_a": "A", "speaker_b": "B", "session_1_date_time": "1 May 2023", "session_1": [
    {"speaker": "A", "dia_id": f"D1:{i}", "text": t} for i, t in enumerate(
        ["I adopted a dog named Rex", "Rex loves the park", "I work as a nurse", "Nursing is tiring",
         "We plan a trip to Rome", "Rome has great food"], start=1)]}


def data():
    qa = [{"question": "What is the dog called?", "answer": "Rex", "category": 4, "evidence": ["D1:1"]},
          {"question": "Where is the trip?", "answer": "Rome", "category": 4, "evidence": ["D1:5"]},
          {"question": "Adversarial?", "answer": None, "category": 5, "evidence": []}]
    return [{"sample_id": "conv-x", "conversation": CONV, "qa": qa}]


def test_pipeline_with_fake_llm_and_cache(tmp_path):
    cache = tmp_path / "c.jsonl"
    seen = []

    def answer(q, ctx):
        seen.append((q, ctx))
        return "Rex" if "dog" in q else "Rome"

    judge = lambda q, c: "CORRECT" if c.split("Model answer:")[1].strip() in q else "WRONG"
    from playmemory import HashEmbedder
    score, calls = run(data(), HashEmbedder(), answer=answer, judge=judge, budget=400, sample=0, rr=None,
                       cache_path=str(cache), systems=("chunks", "span2"))
    assert calls == 2 * 2 * 2  # 2 questions x 2 systems x (answer + judge); adversarial skipped
    assert 'f1' in __import__('json').loads(cache.read_text().splitlines()[0])
    assert score[("chunks", "all")] == [1.0, 1.0] and score[("span2", "single-hop")] == [1.0, 1.0]
    assert any("[1]" in ctx and "Rex" in ctx for q, ctx in seen)  # passages arrive numbered with their ids
    # second run is served entirely from the cache
    score2, calls2 = run(data(), HashEmbedder(), answer=lambda *a: 1 / 0, judge=lambda *a: 1 / 0, budget=400,
                         sample=0, rr=None, cache_path=str(cache), systems=("chunks", "span2"))
    assert calls2 == 0 and score2 == score
    assert len(cache.read_text().splitlines()) == 4 and json.loads(cache.read_text().splitlines()[0])["correct"] is True


def test_frames_for_respects_budget_and_systems():
    m = Memory(embedder="debug", episode_config=EpisodeConfig())
    m.add_many([f"turn {i} about topic{i % 3} filler words here" for i in range(30)], metas=[{"session": i // 10} for i in range(30)],
               times=[float(i) for i in range(30)])
    for s in ("chunks", "span2", "span3"):
        fr = frames_for(m, s, "topic1", 150, None)
        assert fr and sum(len(f.text) + 1 for f in fr) <= 150 + 60
    assert "chunks+rr" in SYSTEMS


def test_report_prints(capsys):
    report({("chunks", "all"): [1.0, 0.0], ("chunks", "single-hop"): [1.0]}, systems=("chunks",))
    out = capsys.readouterr().out
    assert "0.500" in out and "single-hop" in out


def test_f1_and_contains():
    assert f1_and_contains("7 May 2023", "She went on 7 May 2023.") == (pytest.approx(2 / 3), True)
    assert f1_and_contains("Rex", "The dog is called Rex") [1] is True
    assert f1_and_contains("Rome", "Paris") == (0.0, False)
    assert f1_and_contains("the red car", "A red car")[0] == 1.0  # articles ignored
    assert f1_and_contains("x", "") == (0.0, False)


def test_pipeline_without_llm_judge_uses_f1(tmp_path):
    from playmemory import HashEmbedder
    score, calls = run(data(), HashEmbedder(), answer=lambda q, c: "It is Rex" if "dog" in q else "Paris",
                       judge=None, budget=400, sample=0, rr=None, cache_path=None, systems=("chunks",))
    assert calls == 2 and score[("chunks", "all")] == [1.0, 0.0]  # contains-gold decides
    assert score[("chunks", "all", "f1")][0] >= 0.5 and score[("chunks", "all", "f1")][1] == 0.0


def test_batches_are_disjoint_complete_and_crash_tolerant(tmp_path):
    from playmemory import HashEmbedder
    cache = tmp_path / "c.jsonl"
    asked = []

    def answer(q, ctx):
        asked.append(q)
        return "Rex"

    kw = dict(answer=answer, judge=None, budget=400, rr=None, cache_path=str(cache), systems=("chunks",))
    run(data(), HashEmbedder(), sample=1, start=0, **kw)
    run(data(), HashEmbedder(), sample=1, start=1, **kw)
    assert len(asked) == 2 and len(set(asked)) == 2  # two batches of one question: no overlap, nothing missed
    lines = cache.read_text().splitlines()
    assert len(lines) == 2
    # simulate a crash mid-write: a truncated last line must be ignored, not break the resume
    cache.write_text("\n".join(lines) + "\n" + lines[0][:25])
    asked.clear()
    score, calls = run(data(), HashEmbedder(), sample=0, start=0, **kw)
    assert calls == 0 and asked == [] and len(score[("chunks", "all")]) == 2


def test_memory_guard_passes_when_free_and_raises_when_not(monkeypatch):
    import bench.qa as qa

    guard = qa.memory_guard(min_free_mb=1, max_swap_pct=100.0, wait_s=0)
    guard()  # plenty of memory: returns
    tight = qa.memory_guard(min_free_mb=10**9, wait_s=0)  # impossible requirement
    with pytest.raises(qa.MemoryPressure, match="available"):
        tight()


def test_run_stops_cleanly_under_pressure_and_keeps_saved_rows(tmp_path):
    import bench.qa as qa
    from playmemory import HashEmbedder

    cache = tmp_path / "c.jsonl"
    calls = {"n": 0}

    def guard():
        calls["n"] += 1
        if calls["n"] == 2:
            raise qa.MemoryPressure("test")

    with pytest.raises(qa.MemoryPressure):
        run(data(), HashEmbedder(), answer=lambda q, c: "Rex", judge=None, budget=400, sample=0, rr=None,
            cache_path=str(cache), systems=("chunks",), guard=guard)
    assert len(cache.read_text().splitlines()) == 1  # the first answer was durably saved before the stop
