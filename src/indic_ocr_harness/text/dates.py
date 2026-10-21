"""Dates in recognizer output, and the filter for dates harvested from header/footer blocks."""

from __future__ import annotations

import collections
import datetime
import re

from rapidfuzz.distance import Levenshtein

from .scripts import DEVA_TO_ASCII_DIGITS

DATE = re.compile(r"(?<![\d./-])(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{4}|\d{2})(?![\d./-])")

Date = tuple[int, int, int]


def dates_in(text: str, raw: bool = False) -> list[str]:
    """Dates in a text (digits in either script, separators ``. / -``).

    Canonical ``d/m/y`` strings, or with ``raw=True`` the dates as written (spaces removed, digits made ASCII).
    Plot numbers such as 82/17/25 are not dates (month > 12).
    """
    t = text.translate(DEVA_TO_ASCII_DIGITS)

    def ok(a: str, b: str, c: str) -> bool:
        return 1 <= int(a) <= 31 and 1 <= int(b) <= 12 and (len(c) == 2 or 1900 <= int(c) <= 2099)

    ms = [m for m in DATE.finditer(t) if ok(*m.groups())]
    return [re.sub(r"\s+", "", m.group()) if raw else f"{int(m[1])}/{int(m[2])}/{m[3]}" for m in ms]


def _digits(d: Date) -> str:
    return f"{d[0]:02d}{d[1]:02d}{d[2]:04d}"


def _year(this_year: int | None) -> int:
    return this_year if this_year is not None else datetime.date.today().year


def canon_date(d: str, this_year: int | None = None) -> Date | None:
    """(day, month, 4-digit year) of a date string; a 2-digit year is 20yy up to this year's yy, else 19yy.

    ``this_year`` defaults to the current year.
    """
    m = DATE.search(d.translate(DEVA_TO_ASCII_DIGITS))
    if not m:
        return None
    this_year = _year(this_year)
    a, b, c = int(m[1]), int(m[2]), int(m[3])
    if len(m[3]) == 2:
        c += 2000 if c <= this_year % 100 else 1900
    return a, b, c


def harvest_filter(
    md: str, harvested: dict[str, list[str]], this_year: int | None = None, year_window: bool = False
) -> dict[str, list[str]]:
    """Keep only the harvested header/footer dates that add information to the page text ``md``.

    Dropped: exact duplicates of a body date (years canonicalised); dates in the future or before 1990 (optionally
    also outside the body's own year range +-1, ``year_window``); and a near-duplicate (one digit different, or the
    same day and month) of a body date the page repeats at least twice: the header misread a date the page agrees
    on. A near-duplicate of a body date that occurs once is kept, since the body date may be the misread one. The
    year window is off by default: a header can legitimately carry a date outside the body's years. ``this_year``
    defaults to the current year.
    """
    this_year = _year(this_year)
    body = [x for x in (canon_date(d, this_year) for d in dates_in(md, raw=True)) if x]
    years = [y for _, _, y in body]
    count = collections.Counter(body)
    out: dict[str, list[str]] = {}
    for lab, ds in harvested.items():
        keep: list[str] = []
        for d in ds:
            c = canon_date(d, this_year)
            if not c or c in count or c in [canon_date(k, this_year) for k in keep]:
                continue
            if c[2] > this_year or c[2] < 1990 or (year_window and years and not min(years) - 1 <= c[2] <= max(years) + 1):
                continue
            if any(n >= 2 and (Levenshtein.distance(_digits(c), _digits(b)) == 1 or b[:2] == c[:2]) for b, n in count.items()):
                continue
            keep.append(d)
        out[lab] = keep
    return out
