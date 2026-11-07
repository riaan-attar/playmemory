from playmemory import Memory
from playmemory.cards import Card, LLMExtractor, RuleExtractor, profile, remember


def test_rule_extractor_basics():
    cards = RuleExtractor().extract(
        "My name is Riaan. I live in Paris, and I work at Acme Corp. I love hiking. I'm vegetarian. I want to learn Rust."
    )
    got = {(c.predicate, c.value) for c in cards}
    assert ("name", "Riaan") in got
    assert ("lives_in", "Paris") in got
    assert ("works_at", "Acme Corp") in got
    assert ("likes", "hiking") in got
    assert ("diet", "vegetarian") in got
    assert ("wants_to", "learn Rust") in got


def test_exclusive_slot_supersedes_and_time_travels():
    m = Memory()
    remember(m, "I live in Paris.", t=1.0)
    remember(m, "I moved to Berlin last week.", t=10.0)
    assert profile(m)["lives_in"] == ["Berlin"]
    assert profile(m, as_of=5.0)["lives_in"] == ["Paris"]
    hits = m.search("where does the user live", k=3, where={"kind": "card"})
    assert [h.frame.meta["value"] for h in hits] == ["Berlin"]


def test_non_exclusive_preferences_accumulate_and_dedupe():
    m = Memory()
    remember(m, "I love tea.", t=1.0)
    remember(m, "I enjoy coffee.", t=2.0)
    remember(m, "I love tea.", t=3.0)  # duplicate belief
    assert sorted(profile(m)["likes"]) == ["coffee", "tea"]


def test_llm_extractor_adapter():
    ex = LLMExtractor(lambda text: [{"kind": "fact", "subject": "Bob", "predicate": "owns", "value": "a boat"}])
    m = Memory()
    raw, cards = remember(m, "Bob bought a boat", ex, t=1.0)
    assert cards[0].meta["subject"] == "Bob" and cards[0].meta["source"] == raw.id
    assert Card("fact", "Bob", "owns", "a boat").key == "bob|fact|owns"
