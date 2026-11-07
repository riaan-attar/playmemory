# PlayMemory file format (v3)

One append-only file. All integers little-endian. The reader also opens **v2** files
(identical records, shorter header); `compact()` rewrites a v2 file as v3.

```
header | record*
```

## Header (v3)
| field | type | notes |
|---|---|---|
| magic | 4 bytes | `PMEM` |
| version | u16 | 3 (2 is read-only compatible) |
| dim | u16 | embedding dimension |
| name_len, name | u16, bytes | embedder identity; a file only opens with the same name + dim |
| spec_len, spec | u16, bytes | string `from_spec()` can rebuild the embedder from (never a secret) |
| flags | u32 | bit0 `FLAG_VEC_F16` vectors stored as float16; bit1 `FLAG_ENCRYPTED` |
| kdf_len, kdf | u16, bytes | only when encrypted: salt(16) + log2(N) + r + p + sealed key-check value |

The header is never encrypted, so a file can be identified and its embedder rebuilt without the password.

## Record
```
u32 length | u8 type | payload | u32 crc32(type + payload)
```
`length` counts type + payload. The CRC covers the bytes as stored (ciphertext when encrypted), so
`verify` and `doctor` work without a key. A record that is cut short or fails its CRC at the end of
the file is a torn write and is dropped on open; the same inside the file is reported by `verify`
and salvaged by `doctor`.

| type | name | payload |
|---|---|---|
| 1 | frame | `u32 json_len` + JSON `{i, t, text, meta}` + vector (`dim` x float32 or float16) |
| 5 | frame (compressed) | same, but the JSON part is zlib-compressed (used only when it shrinks by 10%+) |
| 2 | tombstone | `u64 frame_id` + `f64 deleted_at` |
| 3 | checkpoint | `.npz`: record offsets, times, metadata, BM25 postings, episode boundaries, event/vector-update offsets |
| 4 | ann | `u64 frames_covered` + serialized faiss HNSW graph |
| 6 | signature | HMAC-SHA256 over every byte before this record |
| 7 | event | JSON session/audit event (op, params, results, snapshot counters) |
| 8 | vector update | `u64 frame_id` + vector: a later embedding for an existing frame (progressive enrichment) |

Frame id = order of frame records. Frames, tombstones and vector updates are the source of truth;
checkpoint and ann records are derived caches (safe to ignore or drop; `compact()` discards them).
Unknown record types are skipped.

## Encryption
When `FLAG_ENCRYPTED` is set, every payload except signatures is `nonce(12) || AES-256-GCM(ciphertext + tag)`
under a key from scrypt(password, salt, N=2^15, r=8, p=1). The associated data is `u8 type + u64 file offset`,
so records cannot be moved or swapped without failing authentication. Checkpoints, ANN graphs, events and
vector updates are sealed like frames.

## Recovery rules
1. Open parses the header, finds the last checkpoint by hopping record lengths, loads it if its CRC holds,
   then replays only the records after it (full CRC check).
2. The first record that is truncated or fails its CRC ends the valid prefix; the tail is dropped
   (truncated unless the file is opened read-only).
3. `doctor` instead resynchronises on the next record whose length, type and CRC are all valid, and copies
   every intact frame, tombstone and vector update into a fresh file, remapping ids.
