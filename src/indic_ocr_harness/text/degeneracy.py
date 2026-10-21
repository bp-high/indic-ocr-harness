"""Degenerate-output test ("test D"): near-repeating junk that the exact-repeat loop detector misses.

Thresholds are label-free: junk reads have very low character-trigram diversity and compress far better than
legitimate text (a low zlib ratio). Tables are judged per cell and by their final rows, never by whole-table
compression: registers of identical rows compress like loops.
"""

from __future__ import annotations

import collections
import math
import re
import zlib

from .scripts import AKSHARA
from .tables import TR, row_text


def _stats(s: str) -> tuple[int, float, float, float, float]:
    """(aksharas, zlib ratio, akshara entropy in bits, top-2 akshara share, character-trigram diversity)."""
    aks = AKSHARA.findall(s)
    n = len(aks)
    if not n:
        return 0, 1.0, 0.0, 0.0, 1.0
    b = s.encode()
    z = len(zlib.compress(b, 9)) / max(1, len(b))
    cnt = collections.Counter(aks)
    h = -sum(c / n * math.log2(c / n) for c in cnt.values())
    top2 = sum(c for _, c in cnt.most_common(2)) / n
    body = s.replace(" ", "")
    tri = len({body[i : i + 3] for i in range(len(body) - 2)}) / max(1, len(body) - 2)
    return n, z, h, top2, tri


def _clean(text: str) -> str:
    t = re.sub(r"<[^>]+>", " ", text)
    t = re.sub(r"([.\-_…xX])\1{3,}", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def degenerate(text: str, kind: str = "text") -> bool:
    """True if a read is degenerate output (``kind`` is ``'text'`` or ``'table'``)."""
    if kind == "table" and "<t" in text:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", text, flags=re.S)
        if any(degenerate(c, "text") for c in cells if len(AKSHARA.findall(_clean(c))) >= 40):
            return True
        rows = [row_text(r) for r in TR.findall(text)]
        return len(rows) >= 5 and len(set(rows[-5:])) == 1 and bool(rows[-1])
    n, z, h, top2, tri = _stats(_clean(text))
    if n >= 120:
        return z < 0.10 or (h < 2.5 and top2 > 0.6) or tri < 0.2
    if n >= 30:
        return h < 1.5
    return False


def strip_degenerate_lines(text: str) -> str:
    """The text without its degenerate lines (a line is judged on its own once it has 30 aksharas)."""
    return "\n".join(line for line in text.split("\n") if not degenerate(line, "text"))
