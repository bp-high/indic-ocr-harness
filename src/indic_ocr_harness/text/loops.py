"""Decoding-loop detection and repair.

Greedy decoding on hard crops sometimes ends in a loop: one chunk repeated until the token cap. These functions
detect a loop after the fact (``looped``), during decoding (``loop_tail``, for the online stop), trim it to one copy
(``trim_loop``), and cut a passage the decoder finished and then started over (``cut_macro_repeat``).
"""

from __future__ import annotations

import re

_ENUM = re.compile(r"^[\W\d०-९]{0,6}")


def looped(text: str, min_unit: int = 4, reps: int = 4) -> bool:
    """True if the text ends in at least ``reps`` repeats of one chunk of ``min_unit``-60 characters."""
    t = re.sub(r"\s+", " ", text[-600:])
    for n in range(min_unit, 61):
        unit = t[-n:]
        if unit.strip() and t.endswith(unit * reps):
            return True
    return False


def loop_tail(text: str, min_span: int = 200) -> bool:
    """Online loop stop: ``looped(text)``, and the repeated tail is long and holds real content.

    The tail must cover at least ``min_span`` characters with a unit that holds something besides table markup
    (repeated ``<td></td>`` is legitimate). A punctuation-only unit (a ``.....`` fill line) needs ``2 * min_span``.
    """
    if not looped(text):
        return False
    t = re.sub(r"\s+", " ", text[-1200:])
    for n in range(4, 61):
        unit = t[-n:]
        if not unit.strip():
            continue
        k = 1
        while t[: len(t) - k * n].endswith(unit):
            k += 1
        core = re.sub(r"<[^>]*>|[\s|]", "", unit)
        if k >= 4 and core and k * n >= (min_span if any(c.isalnum() for c in core) else 2 * min_span):
            return True
    return False


def trim_loop(text: str, min_unit: int = 4, reps: int = 4) -> str:
    """Cut a trailing loop (at least ``reps`` repeats of one chunk) down to a single copy of the chunk."""
    for n in range(min_unit, 61):
        unit = text[-n:]
        if not unit.strip():
            continue
        k = 1
        while text[: len(text) - k * n].endswith(unit):
            k += 1
        if k >= reps:
            return text[: len(text) - (k - 1) * n]
    return text


def cut_macro_repeat(text: str, min_len: int = 60, key_len: int = 80) -> str:
    """Cut a passage the decoder finished and started over (items 1-5, then items 1-5 again).

    The short-loop detector cannot see a repeat of long lines. Cut the text at the first line (at least
    ``min_len`` characters) whose start, ignoring list numbering, repeats the start of an earlier line. Not for
    tables: identical rows are normal there.
    """
    seen, lines = set(), text.split("\n")
    for i, line in enumerate(lines):
        t = re.sub(r"\s+", " ", _ENUM.sub("", line.strip()))
        if len(t) >= min_len:
            k = t[:key_len]
            if k in seen:
                return "\n".join(lines[:i]).rstrip()
            seen.add(k)
    return text
