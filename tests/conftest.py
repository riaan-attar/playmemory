import os

# Tests use the lexical debug embedder; production code must pass a real one.
os.environ.setdefault("PLAYMEMORY_ALLOW_DEBUG_EMBEDDER", "1")
