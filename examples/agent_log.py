"""Conversation log as timeline memory: search, then recall surrounding turns."""
from playmemory import Memory

m = Memory()
turns = [
    "user: I'm planning a trip to Japan in April",
    "assistant: Cherry blossom season, great choice. Tokyo and Kyoto?",
    "user: Kyoto mostly, and I'm vegetarian",
    "assistant: Kyoto has excellent shojin ryori temple cuisine",
    "user: unrelated, my laptop fan is loud",
    "assistant: try cleaning the vents with compressed air",
]
m.add_many(turns, times=[float(i) for i in range(len(turns))])

for hit in m.search("what food should I eat given my diet", k=1, context=1):
    print(f"match (score {hit.score:.2f}):")
    for f in hit.context():
        marker = ">>" if f.id == hit.frame.id else "  "
        print(f"  {marker} [{f.t:g}] {f.text}")
print("state at t=2.5:", m.seek(2.5).text)
