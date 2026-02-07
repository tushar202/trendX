from __future__ import annotations

import hashlib
import re
from typing import Iterable


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _hash64(token: str) -> int:
    h = hashlib.sha1(token.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def simhash(text: str) -> int:
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0

    v = [0] * 64
    for tok in tokens:
        h = _hash64(tok)
        for i in range(64):
            bit = 1 if (h >> i) & 1 else -1
            v[i] += bit

    fingerprint = 0
    for i, val in enumerate(v):
        if val >= 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()
