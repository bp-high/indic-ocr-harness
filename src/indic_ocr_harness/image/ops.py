"""Image operations on block crops (OpenCV + numpy, no model).

- ``line_height``: text line height, the trigger for zooming small handwriting
- ``seam_split``: cut a tall block into strips along minimum-ink seams, so no handwritten line or table row is sliced
- ``header_band``: the column-heading band stacked above later strips of a tiled table
- ``grid_extent``, ``crossed_by_columns``, ``has_ink``: find free text inside a Table block but outside its grid

All functions take RGB ``uint8`` arrays of shape (H, W, 3).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

Strip = tuple[np.ndarray, tuple[int, int], float]


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


def ink_masks(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ink, horizontal rules, vertical rules) binary masks; ink is dark strokes with long ruling lines removed."""
    g = _gray(img)
    h, w = g.shape
    win = max(15, (min(h, w) // 40) | 1)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, win, 15)
    hl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 15), 1)))
    vl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 15))))
    lines = cv2.dilate(hl | vl, np.ones((3, 3), np.uint8))
    ink = cv2.subtract(bw, lines)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # speckle
    return ink, hl, vl


def has_ink(img: np.ndarray, frac: float = 0.004, min_px: int = 120) -> bool:
    """True if the region holds at least ``min_px`` ink pixels and ``frac`` of its area."""
    ink, _, _ = ink_masks(img)
    n = int((ink > 0).sum())
    return n >= min_px and n >= frac * ink.size


# ------------------------------------------------------------------------------------------------------- seams


def _seam(cost: np.ndarray, y_lo: int, y_hi: int, y_target: int, move: float = 0.08, prior: float = 2e-4):
    """Minimum-cost left-to-right path confined to rows [y_lo, y_hi), moving at most 1 row per column.

    ``move`` penalises each diagonal step and ``prior`` (per row of distance, per column) pulls the path towards
    ``y_target``, so in blank areas the seam runs straight instead of drifting into the next table row.
    """
    c = cost[y_lo:y_hi] + prior * np.abs(np.arange(y_lo, y_hi) - y_target)[:, None]
    n, w = c.shape
    dp = c[:, 0].copy()
    back = np.zeros((n, w), np.int8)
    for x in range(1, w):
        up = np.r_[np.inf, dp[:-1]] + move
        dn = np.r_[dp[1:], np.inf] + move
        stack = np.vstack([up, dp, dn])
        arg = stack.argmin(0)
        dp = stack[arg, np.arange(n)] + c[:, x]
        back[:, x] = arg - 1  # -1: came from the row above, 0: same row, +1: below
    ys = np.empty(w, np.int32)
    y = int(dp.argmin())
    for x in range(w - 1, -1, -1):
        ys[x] = y
        y = int(np.clip(y + int(back[y, x]), 0, n - 1)) if x else y
    return ys + y_lo, float(dp.min())


SEAM = {
    "text": {"move": 0.08, "line_bonus": 0.04},  # bend around ascenders / descenders of handwritten lines
    "table": {"move": 1.5, "line_bonus": 0.15},  # stay on one ruling line, i.e. cut between table rows
}


def _seam_cost(img: np.ndarray, kind: str) -> np.ndarray:
    # crossing a stroke is expensive; running along a ruled horizontal line is slightly rewarded, so table cuts fall
    # on row boundaries
    ink, hl, _ = ink_masks(img)
    cost = cv2.GaussianBlur((ink > 0).astype(np.float32), (0, 0), 4) * 4 + 0.05
    cost -= SEAM[kind]["line_bonus"] * cv2.dilate((hl > 0).astype(np.float32), np.ones((5, 1), np.uint8))
    return cost


def seam_split(img: np.ndarray, n_parts: int, kind: str = "text", search_frac: float = 0.35) -> list[Strip]:
    """Split into ``n_parts`` horizontal strips along minimum-ink seams -> [(strip, (y_top, y_bottom), seam cost)].

    Pixels of a strip's bounding box that lie beyond its seams are filled with the page background colour, so a
    stroke never appears half in two strips.
    """
    if n_parts <= 1:
        return [(img, (0, img.shape[0]), 0.0)]
    h, w = img.shape[:2]
    cost = _seam_cost(img, kind)
    step = h / n_parts
    seams = [np.zeros(w, np.int32)]
    costs = []
    for k in range(1, n_parts):
        yc = int(k * step)
        lo, hi = max(seams[-1].max() + 10, int(yc - search_frac * step)), min(h - 10, int(yc + search_frac * step))
        if hi - lo < 5:
            lo, hi = max(1, yc - 5), min(h - 1, yc + 5)
        ys, cst = _seam(cost, lo, hi, yc, move=SEAM[kind]["move"])
        seams.append(ys)
        costs.append(cst)
    seams.append(np.full(w, h, np.int32))
    bgc = np.median(img.reshape(-1, 3)[::97], axis=0).astype(np.uint8)
    out = []
    rows = np.arange(h)[:, None]
    for i in range(n_parts):
        top, bot = seams[i], seams[i + 1]
        y0, y1 = int(top.min()), int(bot.max())
        strip = img[y0:y1].copy()
        r = rows[y0:y1]
        outside = (r < top[None, :]) | (r >= bot[None, :])
        strip[outside] = bgc
        out.append((strip, (y0, y1), costs[i - 1] if i else 0.0))
    return out


def header_band(img: np.ndarray, frac: float = 0.10, win: float = 0.04) -> int:
    """Height of the band stacked above every later strip of a tiled table.

    The minimum-ink row-boundary seam near ``frac`` of the block height. Generous on purpose (it may include the
    first data row or two); the rows a strip repeats from the band are dropped from its output afterwards.
    """
    h, _ = img.shape[:2]
    cost = _seam_cost(img, "table")
    yc = int(frac * h)
    lo, hi = max(1, yc - int(win * h)), min(h - 1, yc + int(win * h))
    ys, _ = _seam(cost, lo, hi, yc, move=SEAM["table"]["move"])
    return int(ys.max())


# ------------------------------------------------------------------------------------------------- line height


def _blob_height(ink: np.ndarray, width: int) -> float | None:
    """Median height of word blobs (strokes merged horizontally). Errs low on ruled paper."""
    blob = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 80), 1)))
    n, _, st, _ = cv2.connectedComponentsWithStats(blob)
    hs = [st[i, 3] for i in range(1, n) if st[i, 2] > 1.5 * st[i, 3] and st[i, 3] > 8 and st[i, 2] > 40]
    return float(np.median(hs)) if len(hs) >= 3 else None


def _component_height(ink: np.ndarray, min_aspect: float = 0.12) -> float | None:
    """Median height of the larger half of non-flat ink components (words joined by the headline). Errs low when
    strokes break into several components (thin pens, scans)."""
    n, _, st, _ = cv2.connectedComponentsWithStats(ink)
    comps = sorted(
        ((st[i, 4], st[i, 3]) for i in range(1, n) if st[i, 3] >= 8 and st[i, 3] >= min_aspect * st[i, 2] and st[i, 4] >= 30),
        reverse=True,
    )
    if len(comps) < 3:
        return None
    return float(np.median([h for _, h in comps[: max(3, len(comps) // 2)]]))


def _pitch_height(img: np.ndarray, min_lag: int = 20, min_peak: float = 0.3) -> float | None:
    """Half the text line pitch, from the autocorrelation of row darkness (no threshold, so faint or broken strokes
    do not matter). Handwriting is about half its line spacing tall. None for single lines, texture or a weak
    periodicity."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    dark = np.clip(cv2.GaussianBlur(g, (0, 0), 25) - g, 0, None)
    prof = dark.mean(1) - dark.mean()
    n = len(prof)
    hi = min(n // 2, 600)
    if n < 120 or hi <= min_lag + 5:
        return None
    ac = np.correlate(prof, prof, mode="full")[n - 1 :]
    ac = ac / (ac[0] + 1e-9)
    k = min_lag + int(np.argmax(ac[min_lag:hi]))
    return 0.5 * k if k > min_lag + 2 and ac[k] >= min_peak else None


def line_height(img: np.ndarray) -> float | None:
    """Text line height in px, or None.

    Three estimators that err low in different situations (word blobs on ruled paper, word components with broken
    strokes, half the line pitch on single lines); the largest is used, so text is only enlarged when all of them
    agree it is small.
    """
    ink, _, _ = ink_masks(img)
    hs = [h for h in (_blob_height(ink, img.shape[1]), _component_height(ink), _pitch_height(img)) if h]
    return max(hs) if hs else None


# -------------------------------------------------------------------------------------------------------- grids


def _rules(hl: np.ndarray, min_frac: float) -> list[tuple[float, float, int, int]]:
    """Long horizontal ruling-line components: [(y centre, slope in degrees, x0, x1)]."""
    _, w = hl.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(hl)
    out = []
    for i in range(1, n):
        if st[i, 2] >= min_frac * w:
            ys, xs = np.where(lab[st[i, 1] : st[i, 1] + st[i, 3], st[i, 0] : st[i, 0] + st[i, 2]] == i)
            k = np.polyfit(xs, ys, 1)[0] if len(xs) > 10 else 0.0
            out.append((float(st[i, 1] + ys.mean()), math.degrees(math.atan(k)), int(st[i, 0]), int(st[i, 0] + st[i, 2])))
    return out


def grid_extent(img: np.ndarray, min_rules: int = 3, min_frac: float = 0.5) -> tuple[int, int] | None:
    """(y_top, y_bottom) of a table's ruled grid: the first and last long horizontal rule, or None."""
    _, hl, _ = ink_masks(img)
    ys = sorted(y for y, _, _, _ in _rules(hl, min_frac))
    return (int(ys[0]), int(ys[-1])) if len(ys) >= min_rules else None


def crossed_by_columns(img: np.ndarray, cover: float = 0.8, margin: float = 0.03, min_h: int = 40) -> bool:
    """True if a vertical rule crosses at least ``cover`` of the region's inked rows.

    Such a region is part of a table's columns (a totals row, ruled notebook paper), not free text next to the
    table. Coverage is measured over the rows that carry ink, so blank paper below a totals row does not hide its
    column separators; the outer ``margin`` of the width (a form's border) is ignored.
    """
    h, w = img.shape[:2]
    if h < min_h:
        return False
    ink, _, vl = ink_masks(img)
    m = int(w * margin)
    rows = np.where(((ink | vl)[:, m : w - m] > 0).sum(1) > 0.002 * w)[0]
    if len(rows) < 10:
        return False
    band = vl[rows[0] : rows[-1] + 1, m : w - m]
    colcov = (band > 0).sum(0) / band.shape[0]
    return bool((colcov >= cover).any())
