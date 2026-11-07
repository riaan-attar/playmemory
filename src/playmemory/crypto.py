"""Password encryption (AES-256-GCM, scrypt KDF) and HMAC signatures.

Every record payload is sealed individually; the header (embedder spec, dimension, KDF
parameters) stays readable. Each record's file offset and type are bound in as
associated data, so records cannot be moved, swapped or replayed elsewhere without
failing authentication. Needs `cryptography` (extra `crypto`) for AES-GCM; scrypt and
HMAC are stdlib.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import struct

DEFAULT_LOGN, DEFAULT_R, DEFAULT_P = 15, 8, 1
_CHECK_PLAIN = b"playmemory"
_NONCE = 12


class WrongPasswordError(ValueError):
    """The password does not unlock this file."""


class EncryptedFileError(ValueError):
    """The file is encrypted and no password was given."""


def _aesgcm():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:
        raise ImportError("encryption needs the 'crypto' extra: pip install 'playmemory[crypto]'") from e
    return AESGCM


def derive_key(password: str | bytes, salt: bytes, logn: int, r: int, p: int) -> bytes:
    pw = password.encode() if isinstance(password, str) else password
    return hashlib.scrypt(pw, salt=salt, n=1 << logn, r=r, p=p, dklen=32, maxmem=1 << 28)


class Cipher:
    def __init__(self, key: bytes):
        self._gcm = _aesgcm()(key)

    @staticmethod
    def _aad(rtype: int, off: int) -> bytes:
        return struct.pack("<BQ", rtype, off)

    def seal(self, payload: bytes, rtype: int, off: int) -> bytes:
        nonce = os.urandom(_NONCE)
        return nonce + self._gcm.encrypt(nonce, payload, self._aad(rtype, off))

    def unseal(self, blob: bytes, rtype: int, off: int) -> bytes:
        from cryptography.exceptions import InvalidTag

        try:
            return self._gcm.decrypt(blob[:_NONCE], blob[_NONCE:], self._aad(rtype, off))
        except InvalidTag as e:
            raise ValueError("record failed authentication (tampered, moved, or wrong key)") from e


def new_kdf_block(password: str | bytes) -> tuple[bytes, Cipher]:
    """-> (kdf bytes to store in the header, cipher). The block holds salt, cost
    parameters and an encrypted check value so a wrong password fails at open."""
    salt = os.urandom(16)
    key = derive_key(password, salt, DEFAULT_LOGN, DEFAULT_R, DEFAULT_P)
    c = Cipher(key)
    check = c.seal(_CHECK_PLAIN, 0, 0)
    return salt + bytes([DEFAULT_LOGN, DEFAULT_R, DEFAULT_P]) + check, c


def open_kdf_block(kdf: bytes, password: str | bytes) -> Cipher:
    salt, (logn, r, p), check = kdf[:16], kdf[16:19], kdf[19:]
    if logn > 22:  # refuse absurd cost parameters from an untrusted header
        raise ValueError("unreasonable KDF parameters")
    c = Cipher(derive_key(password, salt, logn, r, p))
    try:
        if c.unseal(check, 0, 0) != _CHECK_PLAIN:
            raise ValueError
    except ValueError as e:
        raise WrongPasswordError("wrong password") from e
    return c


def hmac_sha256(key: bytes, chunks) -> bytes:
    h = hmac.new(key, digestmod=hashlib.sha256)
    for c in chunks:
        h.update(c)
    return h.digest()
