"""HTML table reads: row extraction, header-band de-duplication and the outside-grid duplicate test."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from .scripts import AKSHARA, DEVA_TO_ASCII_DIGITS

TR = re.compile(r"<tr>.*?</tr>", re.S)
"""One HTML table row (the recognizer writes bare ``<tr>`` tags)."""


def row_text(tr: str) -> str:
    """The visible text of one ``<tr>`` row, whitespace-normalised."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", tr)).strip()


def drop_repeated_rows(html: str, band_html: str, min_ratio: float = 0.75, slack: int = 3) -> str:
    """Drop the header-band rows a later table strip repeats.

    A later strip of a tiled table starts with the header band stacked above it, so its output starts with the
    band's rows again (plus band rows the separate band read happened to skip). Within the first
    ``len(band rows) + slack`` rows, find the last row that repeats a band row and drop it and everything above it.
    """
    band = [row_text(r) for r in TR.findall(band_html)]
    rows = TR.findall(html)
    if not band or not rows:
        return html
    last = -1
    for i, tr in enumerate(rows[: len(band) + slack]):
        t = row_text(tr)
        if t and max(fuzz.ratio(t, b) for b in band) >= min_ratio * 100:
            last = i
    out = html
    for tr in rows[: last + 1]:
        out = out.replace(tr, "", 1)
    return re.sub(r"<thead>\s*</thead>\s*", "", out)


def contained_in(region: str, table: str, n: int = 3) -> tuple[float, float]:
    """How much of an outside-grid read is already in the table read.

    Measured in ways that survive two different readings of the same handwriting: (share of the region's akshara
    ``n``-grams found in the table read, share of its numbers found there; 0.0 when it has fewer than 2 numbers).
    Word matching missed table rows that the ruling detector placed outside the grid and that were read twice.
    """

    def strip(t: str) -> str:
        return re.sub(r"<[^>]+>", " ", t)

    def grams(x: list[str]) -> list[tuple[str, ...]]:
        return [tuple(x[i : i + n]) for i in range(len(x) - n + 1)]

    def numbers(t: str) -> list[str]:
        return re.findall(r"\d[\d.,/\-]*\d|\d{2,}", strip(t).translate(DEVA_TO_ASCII_DIGITS))

    gr, gt = grams(AKSHARA.findall(strip(region))), set(grams(AKSHARA.findall(strip(table))))
    ak = sum(g in gt for g in gr) / len(gr) if gr else 0.0
    nr, nt = numbers(region), set(numbers(table))
    nm = sum(x in nt for x in nr) / len(nr) if len(nr) >= 2 else 0.0
    return ak, nm
