"""Indic script and digit handling: script mapping, foreign-letter and digit-script gates, banned character sets.

Devanagari, Bengali, Gurmukhi, Gujarati and Oriya share one ISCII-derived layout, 0x80 apart in Unicode, so a letter
of one block maps 1:1 onto the same letter of another.
"""

from __future__ import annotations

import re

SCRIPT_BASE = {"deva": 0x0900, "beng": 0x0980, "guru": 0x0A00, "gujr": 0x0A80, "orya": 0x0B00}
"""First code point of each supported script block."""

DEVA_TO_ASCII_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# A Devanagari digit, except a ० right after a Devanagari letter: that is the abbreviation mark (मु०, न०).
DEVA_DIGIT = re.compile(r"(?<![ऀ-ॣॱ-ॿ])[०-९]|[१-९]")

# One Devanagari akshara (consonant cluster + vowel sign + modifier), an independent vowel, or any other non-space
# character. Used to measure text in units that survive different readings of the same handwriting.
AKSHARA = re.compile(
    r"[क-हक़-य़ॹ-ॿ]़?(?:्[क-हक़-य़ॹ-ॿ]़?)*"
    r"्?[ा-ौॢॣ]?[ऀ-ः]?|[ऄ-औॠॡ][ऀ-ः]?|\S"
)


def to_script(text: str, target: str = "deva") -> str:
    """Map letters of the other ISCII-parallel Brahmic blocks onto the expected script.

    The recognizer sometimes writes Marathi in Gujarati letters or Hindi words in Bengali; the page's language is
    known to the caller. The mapping is 1:1, so lengths and positions are preserved.
    """
    base = SCRIPT_BASE[target]
    out = []
    for c in text:
        o = ord(c)
        for b in SCRIPT_BASE.values():
            if b != base and b <= o < b + 0x80:
                c = chr(o - b + base)
                break
        out.append(c)
    return "".join(out)


def foreign_letters(text: str, target: str = "deva") -> tuple[int, int]:
    """(letters of other Indic scripts, all Indic letters) in ``text``; the dandas are shared by all scripts."""
    base = SCRIPT_BASE[target]
    ind = [c for c in text if 0x0900 <= ord(c) < 0x0E00 and ord(c) not in (0x0964, 0x0965)]
    return sum(1 for c in ind if not base <= ord(c) < base + 0x80), len(ind)


def minority_digits(text: str, max_share: float = 0.2, min_n: int = 2) -> str | None:
    """``'deva'`` or ``'ascii'`` when a read mixes digit scripts and one is a small minority, else None.

    A small minority is at least ``min_n`` digits and at most ``max_share`` of all digits: the recognizer switched
    digit script mid-block (a strip read 930 as १३०). A single stray digit is not evidence.
    """
    dv, asc = len(DEVA_DIGIT.findall(text)), len(re.findall("[0-9]", text))
    lo, name = min((dv, "deva"), (asc, "ascii"))
    return name if min_n <= lo <= max_share * (dv + asc) else None


def count_numbers(text: str) -> int:
    """Number of numeric tokens (digits in either script; 12/5/25 and 3.5 count once)."""
    return len(re.findall(r"\d+(?:[./]\d+)*", text.translate(DEVA_TO_ASCII_DIGITS)))


def ban_chars(part: str) -> set[str]:
    """The characters that define one banned-token set.

    ``'latex'``: math mode (``$``, backslash). ``'script:<s>'``: letters of the other Indic scripts (U+0900-0DFF
    outside script ``s``; the dandas U+0964/5 are shared). ``'digits:deva'`` / ``'digits:ascii'``: one digit script.
    """
    if part == "latex":
        return {"$", "\\"}
    if part.startswith("script:"):
        base = SCRIPT_BASE[part.split(":", 1)[1]]
        return {chr(o) for o in range(0x0900, 0x0E00) if not base <= o < base + 0x80 and o not in (0x0964, 0x0965)}
    if part == "digits:deva":
        return set("०१२३४५६७८९")
    if part == "digits:ascii":
        return set("0123456789")
    raise ValueError(f"unknown ban set {part!r}")
