"""Indic script and digit handling: script mapping, foreign-letter and digit-script gates, banned character sets.

Devanagari, Bengali, Gurmukhi, Gujarati and Oriya share one ISCII-derived layout, 0x80 apart in Unicode, so most
letters of one block map 1:1 onto the same letter of another (``to_script``). A few code points 0x80 apart are not the
same letter (Bengali য় is the plain y, not य़); ``map_script`` maps those by hand.
"""

from __future__ import annotations

import re
import unicodedata

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


# Code points of the parallel blocks that are not the same letter 0x80 apart, mapped by hand onto Devanagari: Bengali
# and Oriya YYA is the plain y (य, not य़), khanda ta is त्, the Assamese and Oriya ra and wa letters are र and व,
# Gurmukhi tippi is the anusvara and yakash a subjoined ya. A length mark only belongs inside a two-part vowel sign,
# which NFC composes first, so one left over is stray and is dropped.
_FIX_DEVA = {
    "য়": "य",
    "ৎ": "त्",
    "ৰ": "र",
    "ৱ": "व",
    "ৗ": "",
    "ੰ": "ं",
    "ੱ": "",
    "ੲ": "इ",
    "ੳ": "उ",
    "ੵ": "्य",
    "ୟ": "य",
    "ୱ": "व",
    "ୖ": "",
    "ୗ": "",
}
# Signs with no Devanagari counterpart (currency, fractions, isshar, ek onkar): left as written.
_KEEP = (
    set(range(0x09F2, 0x0A00)) | {0x0A74, 0x0A76, 0x0AF1} | set(range(0x0AF9, 0x0B00)) | {0x0B70} | set(range(0x0B72, 0x0B80))
)
# Sequences rewritten before mapping: ya + nukta is YYA; Gurmukhi iri/ura + vowel sign is one independent vowel.
_PRE_DEVA = [
    ("য়", "য়"),
    ("ଯ଼", "ୟ"),
    ("ੲਿ", "ਇ"),
    ("ੲੀ", "ਈ"),
    ("ੲੇ", "ਏ"),
    ("ੲੈ", "ਐ"),
    ("ੳੁ", "ਉ"),
    ("ੳੂ", "ਊ"),
    ("ੳੋ", "ਓ"),
]
_ADDAK = re.compile("ੱ([ਕ-ਹਖ਼-ਫ਼])")  # addak doubles the next consonant: ਪੱਕਾ = पक्का


def _other_block(o: int, base: int) -> int | None:
    for b in SCRIPT_BASE.values():
        if b != base and b <= o < b + 0x80:
            return b
    return None


def map_script(text: str, target: str = "deva") -> str:
    """Map letters of the other parallel scripts onto ``target`` as the letters they stand for.

    ``to_script`` with the non-parallel code points fixed for Devanagari (Bengali য় is य, ৎ is त्, Gurmukhi tippi
    is the anusvara, ...), two-part vowel signs composed first (NFC) and signs without a counterpart kept. A text
    with no letters of another parallel script is returned unchanged. The output can differ in length from the input.
    """
    base = SCRIPT_BASE[target]
    if not any(_other_block(ord(c), base) for c in text):
        return text
    t = unicodedata.normalize("NFC", text)
    if target == "deva":
        for a, b in _PRE_DEVA:
            t = t.replace(a, b)
        t = _ADDAK.sub(lambda m: m.group(1) + "੍" + m.group(1), t)
    out = []
    for c in t:
        o = ord(c)
        b = _other_block(o, base)
        if b is None or o in _KEEP:
            out.append(c)
        elif target == "deva" and c in _FIX_DEVA:
            out.append(_FIX_DEVA[c])
        else:
            out.append(chr(o - b + base))
    return "".join(out)


def map_script_spans(text: str, spans: list[tuple[int, int]], target: str = "deva") -> tuple[str, list[tuple[int, int]]]:
    """``(map_script(text), spans)`` with the character spans moved onto the mapped text.

    Words are mapped one by one and whitespace is kept, so a span boundary at a word edge maps exactly; one inside a
    word whose length changed moves to that word's edge.
    """
    out: list[str] = []
    pos = [0] * (len(text) + 1)
    n = 0
    for m in re.finditer(r"\S+|\s+", text):
        w = m.group()
        w2 = w if w.isspace() else map_script(w, target)
        for i in range(len(w)):
            pos[m.start() + i] = n + (i if len(w2) == len(w) else (0 if i == 0 else len(w2)))
        out.append(w2)
        n += len(w2)
    pos[len(text)] = n
    return "".join(out), [(pos[s], pos[e]) for s, e in spans]


def script_reread_plausible(first: str, reread: str, lo: float = 0.5, hi: float = 2.0, latin_slack: int = 5) -> bool:
    """Whether a script re-read that differs a lot from the mapped first read is still a plausible reading.

    When the recognizer writes a Hindi line as Bengali words, the letter map cannot fix it and the right re-read
    differs from the mapped read by more than half. Such a re-read is plausible if its length in letters is within
    ``[lo, hi]`` times the first read's and it adds at most ``latin_slack`` Latin letters: banning the other script
    sometimes makes the recognizer escape into English.
    """

    def latin(t: str) -> int:
        return len(re.findall("[A-Za-z]", t))

    def letters(t: str) -> int:
        return latin(t) + len(re.findall("[ऀ-෿]", t))

    r = letters(reread) / max(1, letters(first))
    return lo <= r <= hi and latin(reread) - latin(first) <= latin_slack


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
