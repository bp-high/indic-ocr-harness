"""Character-level alignment between two readings of the same crop: span splicing and the two-of-three vote."""

from __future__ import annotations

import re
from collections.abc import Callable

from rapidfuzz.distance import Levenshtein

Span = tuple[int, int]


def word_spans(text: str, pred: Callable[[str], bool]) -> list[Span]:
    """Character spans of the whitespace-delimited words for which ``pred(word)`` is true; adjacent ones merged."""
    spans: list[Span] = []
    for m in re.finditer(r"\S+", text):
        if pred(m.group()):
            if spans and not text[spans[-1][1] : m.start()].strip(" \t"):
                spans[-1] = (spans[-1][0], m.end())
            else:
                spans.append((m.start(), m.end()))
    return spans


def merge_spans(spans: list[Span]) -> list[Span]:
    """Sorted spans with overlapping or touching ones merged."""
    out: list[Span] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(e, out[-1][1]))
        else:
            out.append((s, e))
    return out


def splice(a: str, b: str, spans: list[Span]) -> str:
    """``a`` with each span replaced by the part of ``b`` aligned to it; the rest of ``a`` is kept.

    The edit is confined to the flagged spans, so letter flips elsewhere in a re-read do not leak in. A span that
    starts or ends inside a changed region is widened to that region; a span that would then overlap the next one is
    left as it was.
    """
    ops = [(o.tag, o.src_start, o.src_end, o.dest_start, o.dest_end) for o in Levenshtein.opcodes(a, b)]

    def start(s: int) -> tuple[int, int]:  # (s widened, position in b); an insertion at s is included
        for tag, i1, i2, j1, _ in ops:
            if s == i1:
                return s, j1
            if i1 < s < i2:
                return (s, j1 + s - i1) if tag == "equal" else (i1, j1)
        return len(a), len(b)

    def end(e: int) -> tuple[int, int]:
        for tag, i1, i2, j1, j2 in reversed(ops):
            if e == i2:
                return e, j2
            if i1 < e < i2:
                return (e, j1 + e - i1) if tag == "equal" else (i2, j2)
        return 0, 0

    out, last = [], len(a)
    for s, e in sorted(spans, reverse=True):
        (s, bs), (e, be) = start(s), end(e)
        if e > last or be < bs:  # widened into the next span: leave this one as it was
            continue
        out.append(a[e:last])
        out.append(b[bs:be])
        last = s
    out.append(a[:last])
    return "".join(reversed(out))


def agreed_edits(a: str, b: str, c: str) -> tuple[str, int]:
    """``a`` with only the edits that ``b`` and ``c`` both make, identically and at the same place.

    A two-of-three vote with ``a`` as the backbone: a change needs two independent views to agree. Returns the new
    text and the number of edits applied.
    """
    eb = {(o.src_start, o.src_end, b[o.dest_start : o.dest_end]) for o in Levenshtein.opcodes(a, b) if o.tag != "equal"}
    ec = {(o.src_start, o.src_end, c[o.dest_start : o.dest_end]) for o in Levenshtein.opcodes(a, c) if o.tag != "equal"}
    out = a
    for i1, i2, rep in sorted(eb & ec, reverse=True):
        out = out[:i1] + rep + out[i2:]
    return out, len(eb & ec)
