"""LaTeX in recognizer output.

The recognizer's text prompt asks for LaTeX math, so handwritten fraction-style plot numbers (12 over 7) come back
as ``$\\frac{12}{7}$`` and dates sometimes as ``$14x=3x-6$``. Clean math is rewritten as plain text; math that does
not rewrite cleanly is flagged for a re-read with math tokens banned.
"""

from __future__ import annotations

import re

from .align import Span, word_spans

LATEX_RE = re.compile(r"\$|\\\(|\\\[")
"""Matches the start of any LaTeX math in a read."""


def latex_to_plain(text: str) -> str:
    r"""Rewrite LaTeX math as the plain text it denotes.

    Fixed mapping: ``\frac{a}{b}`` -> ``a/b``, ``\text{x}`` -> ``x``, ``\times`` -> ``×``, spacing commands -> space,
    other commands and braces dropped, ``$`` delimiters removed.
    """

    def conv(m: re.Match) -> str:
        s = m.group(1)
        for _ in range(3):
            s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", s)
        s = re.sub(r"\\(?:text|mathrm|textbf|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", s)
        s = s.replace("\\times", "×").replace("\\cdot", "·").replace("\\%", "%").replace("\\div", "÷")
        s = re.sub(r"\\[qs]?quad|\\[,;: !]", " ", s)
        s = re.sub(r"\^\{([^{}]*)\}|_\{([^{}]*)\}", lambda k: k.group(1) or k.group(2), s)
        s = re.sub(r"\\[a-zA-Z]+", " ", s).replace("{", "").replace("}", "")
        return re.sub(r"[ \t]+", " ", s).strip()

    text = re.sub(r"\$\$(.+?)\$\$", conv, text, flags=re.S)
    text = re.sub(r"\$(.+?)\$", conv, text)
    text = re.sub(r"\\\((.+?)\\\)", conv, text)
    return re.sub(r"\\\[(.+?)\\\]", conv, text, flags=re.S)


def latex_spans(text: str) -> list[Span]:
    r"""Spans of LaTeX math (``$..$``, ``$$..$$``, ``\(..\)``, ``\[..\]``); a lone ``$`` or command marks its word."""
    spans = [m.span() for m in re.finditer(r"\$\$.+?\$\$|\$.+?\$|\\\(.+?\\\)|\\\[.+?\\\]", text, flags=re.S)]
    rest = text
    for s, e in spans:
        rest = rest[:s] + " " * (e - s) + rest[e:]
    spans += word_spans(rest, lambda w: "$" in w or bool(re.search(r"\\[a-zA-Z(\[]", w)))
    return sorted(spans)


def suspect_latex(span: str) -> bool:
    r"""True if a LaTeX span does not rewrite cleanly: after ``latex_to_plain`` it still has Latin letters or a
    backslash (``$14x=3x-6$`` for the date 14-3-26). Clean spans (``$\frac{12}{5}$`` -> 12/5) are left to the
    plain-text rewrite: a re-read with math tokens banned can drop part of a fraction."""
    return bool(re.search(r"[A-Za-z\\]", latex_to_plain(span)))
