"""The harness algorithm for one page: same two models and weights as the model card, but the harness decides what
the recognizer is shown and how its answers are assembled.

Each component follows one pattern: a trigger, a candidate read, a verification, and acceptance or rollback to the
model card's read. Per block (``read_block``):

1. **Scale** (``read_scaled``): enlarge small handwriting so a text line is >= ``zoom.min_line`` px, or cut a big
   block into strips at minimum-ink seams instead of shrinking it; a zoomed read is kept only if it beats the card
   read (``verify_zoom``), strips only if they recover clearly more (``verify_split``).
2. **Guard** (``read_crops``): a read that hits the token cap, loops or is degenerate is split in two and re-read.
3. **Repair** (``repair``): LaTeX junk, another script's letters, or a stray minority digit script are re-read once
   with those tokens banned; only the flagged spans are spliced in.
4. **Table text** (``table_text``): free text above or below a table's grid is read with the text prompt and added
   when it passes quality and duplicate gates.
5. **Vote** (``vote_block``): the page's lowest-confidence blocks are read from two shifted views; an edit is applied
   only where both views agree.

Per page: header/footer dates the page lacks are added (``harvest_dates``), passages written twice are cut, letters
are mapped to the page's script, LaTeX is rewritten as plain text, and an uncertainty sidecar is written.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from ..config import PLAIN, HarnessConfig
from ..image import ops
from ..text.align import agreed_edits, merge_spans, splice, word_spans
from ..text.dates import dates_in, harvest_filter
from ..text.degeneracy import degenerate, strip_degenerate_lines
from ..text.latex import LATEX_RE, latex_spans, latex_to_plain, suspect_latex
from ..text.loops import cut_macro_repeat
from ..text.scripts import DEVA_DIGIT, count_numbers, foreign_letters, minority_digits, to_script
from ..text.tables import TR, contained_in, drop_repeated_rows
from .flags import page_flags
from .records import block_lp, final_reads, mean_lp, piece

if TYPE_CHECKING:
    from PIL.Image import Image

    from ..model.bodhan import BodhanRuntime
    from ..model.recognizer import Recognizer

Trace = dict[str, Any]
Box = tuple[int, int, int, int]

PLAIN_MODES = ("as model card", "zoom", "less shrink")


def token_budget(native_px: int, max_new: int, per_vision_token: int = 10, lo: int = 64) -> int:
    """New-token budget of one crop from its native pixel area.

    At most ~10 text tokens per 32x32-px vision token (decoding precision collapses beyond ~10x text/vision
    compression, DeepSeek-OCR), rounded up to a power of two, within [lo, max_new]. Whole blocks keep the card's
    2,048; a small crop gets a few hundred, so a loop on a dotted fill-in line cannot hold a batch for 2,048 steps.
    """
    want = max(lo, math.ceil(per_vision_token * native_px / 1024))
    return min(max_new, 1 << (want - 1).bit_length())


def covered_frac(b: Any, others: list[Any]) -> float:
    """Share of block ``b``'s box covered by the boxes of ``others`` (on a 64x64 grid)."""
    x0, y0, x1, y1 = b.bbox_xyxy
    m = np.zeros((64, 64), bool)
    for o in others:
        a0, b0, a1, b1 = o.bbox_xyxy
        ix0, iy0 = (max(x0, a0) - x0) / (x1 - x0), (max(y0, b0) - y0) / (y1 - y0)
        ix1, iy1 = (min(x1, a1) - x0) / (x1 - x0), (min(y1, b1) - y0) / (y1 - y0)
        if ix1 > ix0 and iy1 > iy0:
            m[int(iy0 * 64) : math.ceil(iy1 * 64), int(ix0 * 64) : math.ceil(ix1 * 64)] = True
    return float(m.mean())


@dataclass
class PageOutput:
    """What ``run_page`` returns: the layout's blocks with text, the page text, traces and sidecars."""

    blocks: list[Any]
    markdown: str
    traces: dict[int, Trace] = field(default_factory=dict)
    harvested_dates: dict[str, list[str]] | None = None
    flags: dict[str, list] | None = None


class PageRun:
    """One page through one config. Construct and call ``run()``; the methods are the components."""

    def __init__(self, image: Image, layout: Any, cfg: HarnessConfig, reader: Recognizer, rt: BodhanRuntime) -> None:
        self.image, self.layout, self.reader, self.rt = image, layout, reader, rt
        self.cfg = cfg if cfg.harness else PLAIN  # harness=False: every component off
        self.crop_cfg = rt.CropConfig()
        self.text_prompt = rt.TEXT_PROMPT

    # -- reading crops ------------------------------------------------------------------------------------- #
    def prep(self, arr: np.ndarray) -> Image:
        """The image the recognizer sees for a crop: the model card's area clamp."""
        from PIL import Image as PILImage

        return self.rt.area_clamp(PILImage.fromarray(arr), self.crop_cfg)

    def budget(self, arr: np.ndarray) -> int:
        return token_budget(arr.shape[0] * arr.shape[1], self.reader.max_new) if self.cfg.token_budget else self.reader.max_new

    def read_crops(self, arrs: list[np.ndarray], prompt: str, kind: str, depth: int = 0) -> tuple[list[str], list[Trace]]:
        """Read crops with the cap/loop/degeneracy guard and the banned-token repair -> (texts, traces)."""
        cfg = self.cfg
        budgets = [self.budget(a) for a in arrs]
        res = self.reader.read([(self.prep(a), prompt) for a in arrs], max_new=budgets)
        texts, pieces = [], []
        for r, a, bud in zip(res, arrs, budgets, strict=True):
            degen = cfg.degeneracy_check and degenerate(r["text"], kind)
            if (r["hit"] or degen) and cfg.guard and depth < 2 and a.shape[0] > 200:
                sub = [s for s, _, _ in ops.seam_split(a, 2, kind)]
                t2, p2 = self.read_crops(sub, prompt, kind, depth + 1)
                texts.append("\n".join(t2))
                pieces.append({"split_after_hit": p2, "first_read": dict(piece(r, a), degenerate=degen)})
                continue
            t, p = r["text"], piece(r, a)
            if degen:  # junk that could not be split away is dropped (abstain)
                t, p = strip_degenerate_lines(t), dict(p, degenerate=True)
            if not r["hit"]:
                t, p = self.repair(t, p, a, prompt, kind, bud)
            texts.append(t)
            pieces.append(p)
        return texts, pieces

    def repair(self, text: str, first: Trace, a: np.ndarray, prompt: str, kind: str, bud: int) -> tuple[str, Trace]:
        """Constrained re-read of a finished read with LaTeX junk, another script's letters, or a small minority of
        the other digit script.

        The crop is re-read once with those tokens banned, and only the flagged spans of the first read are replaced
        by the aligned parts of the re-read. For a script fix, the recognizer itself chooses between the 1:1 offset
        mapping and the spliced re-read (teacher-forced log-prob). The re-read is dropped if it hit the cap, diverged
        (normalised edit distance > 0.5) or is degenerate.
        """
        cfg = self.cfg
        lat_spans = [sp for sp in latex_spans(text) if suspect_latex(text[sp[0] : sp[1]])] if cfg.latex_ban else []
        nf, ni = foreign_letters(text, cfg.script) if cfg.script_ban and cfg.script else (0, 0)
        scr = nf >= 3 and nf >= 0.02 * ni
        dig = minority_digits(text) if cfg.digit_ban else None
        bans = (["latex"] if lat_spans else []) + ([f"script:{cfg.script}"] if scr else []) + ([f"digits:{dig}"] if dig else [])
        if not bans:
            return text, first
        ban = "+".join(sorted(bans))
        img = self.prep(a)
        r2 = self.reader.read([(img, prompt)], max_new=[bud], ban=ban)[0]
        rec: Trace = {"repair": ban, "reread": piece(r2, a), "first_read": first}
        base = to_script(text, cfg.script) if scr else text  # 1:1 letter mapping: same length and positions
        if (
            r2["hit"]
            or Levenshtein.normalized_distance(base, r2["text"]) > 0.5
            or (cfg.degeneracy_check and degenerate(r2["text"], kind))
        ):
            return text, dict(rec, accepted=False, why="re-read hit the cap or diverged")

        def is_minority_digit(w: str) -> bool:
            return bool(DEVA_DIGIT.search(w) if dig == "deva" else re.search("[0-9]", w))

        other = lat_spans + (word_spans(text, is_minority_digit) if dig else [])
        script_spans = word_spans(text, lambda w: foreign_letters(w, cfg.script)[0] > 0) if scr else []
        new = splice(base, r2["text"], merge_spans(other + script_spans))
        rec.update(accepted=True)
        if scr:
            mapped = splice(base, r2["text"], merge_spans(other)) if other else base
            cands = [to_script(mapped, cfg.script), to_script(new, cfg.script)]
            if cands[0] != cands[1]:
                try:
                    sc: list[float] | None = [x[0] for x in self.reader.score(img, prompt, cands)]
                except Exception as ex:  # never let the arbiter break a page
                    sc = None
                    rec["score_error"] = repr(ex)[:200]
                rec["score"] = sc
                if not sc or sc[0] >= sc[1]:
                    new = cands[0]
                    rec["script_choice"] = "offset map"
                else:
                    rec["script_choice"] = "re-read"
        return new, rec

    # -- scale, strips and their verification ---------------------------------------------------------------- #
    def read_strips(self, z: np.ndarray, prompt: str, kind: str) -> tuple[str, Trace]:
        """Seam-cut a crop that is over the pixel budget into strips. Later table strips get the header band stacked
        above them; a strip whose output adds no row beyond the band is re-read without it."""
        n = math.ceil(z.shape[0] * z.shape[1] / (0.85 * self.crop_cfg.max_px))
        parts = ops.seam_split(z, n, kind)
        plain = [p for p, _, _ in parts]
        hb = ops.header_band(z) if kind == "table" and n > 1 and self.cfg.header_band else 0
        info: Trace = {"strips": n, "strip_rows": [list(r) for _, r, _ in parts]}
        if not hb:
            texts, pieces = self.read_crops(plain, prompt, kind)
            return "\n".join(t for t in texts if t), dict(info, pieces=pieces)
        gap = np.full((24, z.shape[1], 3), np.median(z.reshape(-1, 3)[::97], axis=0), np.uint8)
        banded = plain[:1] + [np.vstack([z[:hb], gap, s_]) for s_ in plain[1:]]
        band_text, band_piece = self.read_crops([z[:hb]], prompt, kind)
        texts, pieces = self.read_crops(banded, prompt, kind)
        rereads = []
        for k in range(1, len(texts)):
            kept = drop_repeated_rows(texts[k], band_text[0])
            if TR.findall(texts[k]) and not TR.findall(kept):  # the model read only the stacked band
                t2, p2 = self.read_crops([plain[k]], prompt, kind)
                kept = t2[0]
                rereads.append(k)
                pieces[k] = {"band_only_reread": p2[0], "first_read": pieces[k]}
            texts[k] = kept
        return "\n".join(t for t in texts if t), dict(
            info, header_band_px=hb, header_band_read=band_piece[0], pieces=pieces, band_rereads=rereads
        )

    def read_scaled(self, a: np.ndarray, prompt: str, kind: str) -> tuple[str, Trace]:
        """Line-height-aware scaling (see ``ZoomConfig``). A big block that the budget would shrink below
        ``min_line`` is cut into strips instead; when the card's own scale already gives ``min_line``, the crop is
        sent unchanged."""
        cfg, zoom, max_px = self.cfg, self.cfg.zoom, self.crop_cfg.max_px
        area = a.shape[0] * a.shape[1]
        s_card = min(1.0, math.sqrt(max_px / area))
        h = ops.line_height(a) if zoom else None
        s = max(s_card, min(zoom.min_line / h, zoom.max_zoom)) if zoom and h else s_card
        if area > max_px:  # never enlarge a block beyond its native pixels
            s = min(s, 1.0)
        elif s > 1.0:  # enlarge only while the crop still fits the budget: enlarging never creates strips
            s = min(s, math.sqrt(max_px / area))
        if s < s_card * (zoom.min_change if zoom else 1.0):  # small corrections only add re-read noise
            s = s_card
        info: Trace = {"line_px": round(h, 1) if h else None, "card_scale": round(s_card, 3), "scale": round(s, 3)}
        if s <= s_card * 1.02:
            texts, pieces = self.read_crops([a], prompt, kind)
            return texts[0], dict(info, mode="as model card", pieces=pieces)

        def size(f: Any) -> tuple[int, int]:
            return max(1, f(a.shape[1] * s)), max(1, f(a.shape[0] * s))

        wz, hz = size(round)
        if zoom and zoom.floor and wz * hz > max_px:  # rounding up would cross the budget: round down
            wz, hz = size(math.floor)
        z = cv2.resize(a, (wz, hz), interpolation=cv2.INTER_LANCZOS4 if s > 1 else cv2.INTER_AREA)
        if z.shape[0] * z.shape[1] <= max_px:
            texts, pieces = self.read_crops([z], prompt, kind)
            out = (texts[0], dict(info, mode="zoom" if s > 1 else "less shrink", pieces=pieces))
            return self.verify_zoom(a, out, prompt, kind) if zoom and zoom.verify_margin is not None else out
        if cfg.strip_max_aspect:  # no extreme-aspect strips
            n = math.ceil(z.shape[0] * z.shape[1] / (0.85 * max_px))
            if z.shape[1] / max(1, z.shape[0] / n) > cfg.strip_max_aspect:
                texts, pieces = self.read_crops([a], prompt, kind)
                return texts[0], dict(info, mode="as model card", why="strips would exceed the aspect cap", pieces=pieces)
        t, inf = self.read_strips(z, prompt, kind)
        if cfg.split_verify:
            return self.verify_split(a, t, dict(info, mode="strips", **inf), prompt, kind)
        return t, dict(info, mode="strips", **inf)

    def card_read(self, a: np.ndarray, prompt: str, kind: str) -> tuple[str, Trace]:
        """The model card's read of a crop (its own scale), through the guard: the fallback of every component."""
        texts, pieces = self.read_crops([a], prompt, kind)
        return texts[0], pieces[0]

    def verify_zoom(self, a: np.ndarray, out: tuple[str, Trace], prompt: str, kind: str) -> tuple[str, Trace]:
        """Keep a zoomed read only if it ended normally, is not degenerate, (optionally) adds no LaTeX the card read
        lacks, has a length within ``zoom.verify_len`` of the card read's, and its mean log-prob is no lower than the
        card read's minus ``zoom.verify_margin``. Otherwise use the card read."""
        zoom = self.cfg.zoom
        assert zoom is not None and zoom.verify_margin is not None
        t, info = out
        tc, pc = self.card_read(a, prompt, kind)
        rz, rc = final_reads(info), final_reads(pc)
        lz, lc = mean_lp(rz), mean_lp(rc)
        card_hit = any(r["hit"] for r in rc)
        why = None
        if any(r["hit"] for r in rz) or degenerate(t, kind):
            why = "zoomed read looped or is degenerate"
        elif LATEX_RE.search(t) and not LATEX_RE.search(tc) and not zoom.latex_ok:
            why = "zoom added LaTeX"
        elif (
            zoom.verify_len
            and not card_hit
            and not zoom.verify_len[0] <= len(t.strip()) / max(1, len(tc.strip())) <= zoom.verify_len[1]
        ):
            why = "length differs from the card read"
        elif lz is not None and lc is not None and lz < lc - zoom.verify_margin:
            why = "lower confidence than the card read"
        if why:
            return tc, dict(info, mode="as model card", zoom_rejected=why, zoom_lp=lz, card_lp=lc, pieces=[pc])
        return t, dict(info, zoom_lp=lz, card_lp=lc)

    def verify_split(self, a: np.ndarray, t_split: str, info_split: Trace, prompt: str, kind: str) -> tuple[str, Trace]:
        """Keep a block's strips only if the unsplit read was truncated or looped, or the split recovers clearly more
        (numbers +10 % and +2, table rows +2, or text +25 %) without a digit-script switch the unsplit read lacks.
        Degenerate strips or strips that still hit the cap fall back to the unsplit read."""
        rc = self.reader.read([(self.prep(a), prompt)], max_new=[self.budget(a)])[0]
        tc = rc["text"]
        if self.cfg.degeneracy_check:  # junk is not "recovered text"
            raw, t_split = t_split, strip_degenerate_lines(t_split)
            if t_split != raw or any(r["hit"] for r in final_reads(info_split)):
                texts, pieces = self.read_crops([a], prompt, kind)
                return texts[0], {
                    "mode": "split_verify",
                    "kept": "card",
                    "split": info_split,
                    "card": pieces[0],
                    "why": "a strip was degenerate or still hit the cap",
                    "scale": info_split.get("card_scale"),
                }
        keep = (
            rc["hit"]
            or count_numbers(t_split) >= max(1.1 * count_numbers(tc), count_numbers(tc) + 2)
            or len(TR.findall(t_split)) >= len(TR.findall(tc)) + 2
            or len(t_split) >= 1.25 * len(tc)
        )
        if keep and not rc["hit"] and minority_digits(t_split) and not minority_digits(tc):
            keep = False
        if keep:
            return t_split, {"mode": "split_verify", "kept": "split", "split": info_split, "card": piece(rc, a)}
        texts, pieces = self.read_crops([a], prompt, kind)
        return texts[0], {
            "mode": "split_verify",
            "kept": "card",
            "split": info_split,
            "card": pieces[0],
            "scale": info_split.get("card_scale"),
        }

    # -- blocks ------------------------------------------------------------------------------------------- #
    def read_block(self, a: np.ndarray, prompt: str, kind: str) -> tuple[str, Trace]:
        """One block. For tables, text above the first / below the last long rule (a date line over a form, a note
        under a table) is read with the text prompt: the table prompt says "Output only the HTML table"."""
        cfg = self.cfg
        segs = [(a, prompt, kind, "block")]
        if kind == "table" and cfg.table_text:
            ext = ops.grid_extent(a)
            if ext:
                ya, yb = max(0, ext[0] - 4), min(a.shape[0], ext[1] + 6)
                above, below = a[:ya], a[yb:]
                # free text only: a region that table columns run through is part of the table
                has_a = above.shape[0] >= 40 and ops.has_ink(above) and not ops.crossed_by_columns(above)
                has_b = below.shape[0] >= 40 and ops.has_ink(below) and not ops.crossed_by_columns(below)
                if has_a or has_b:
                    segs = (
                        ([(above, self.text_prompt, "text", "above grid")] if has_a else [])
                        + [(a[ya:yb], prompt, kind, "grid")]
                        + ([(below, self.text_prompt, "text", "below grid")] if has_b else [])
                    )
        if cfg.table_text_gated and len(segs) > 1:
            return self.table_text(a, prompt, kind, segs)
        texts, parts = [], []
        for x, pr, kd, role in segs:
            t, inf = self.read_scaled(x, pr, kd)
            texts.append(t)
            parts.append(dict(inf, role=role))
        return "\n".join(t for t in texts if t), {"mode": "segments", "parts": parts}

    def table_text(self, a: np.ndarray, prompt: str, kind: str, segs: list) -> tuple[str, Trace]:
        """Text outside a table's grid as an addition to the whole-table read, never a re-crop of the grid
        (re-cropping changed table cells).

        A region is read only if it is >= 2 line heights tall, and its read is kept only if it is not degenerate,
        has no LaTeX, is not itself a table (the ruling detector missed part of the grid), is mostly letters/digits,
        has mean log-prob >= -0.6, and is not already in the table read (60 % of its words, or with
        ``table_dup_aksharas`` half of its akshara 3-grams or numbers).
        """
        cfg = self.cfg
        tt, ti = self.read_scaled(a, prompt, kind)
        parts, extra = [dict(ti, role="table")], {}
        lh = ops.line_height(a) or 0
        tw = re.sub(r"<[^>]+>", " ", tt).split()
        for x, pr, kd, role in segs:
            if role == "grid":
                continue
            rec: Trace = {"role": role, "h": int(x.shape[0]), "line_px": round(lh, 1)}
            if lh and x.shape[0] < 2.0 * lh:
                parts.append(dict(rec, kept=False, why="region under 2 line heights"))
                continue
            t, inf = self.read_scaled(x, pr, kd)
            lp = mean_lp(final_reads(inf))
            words = t.split()
            dup = sum(1 for w in words if tw and max(fuzz.ratio(w, v) for v in tw) >= 80) / max(1, len(words))
            alnum = sum(ch.isalnum() for ch in t) / max(1, len(t.replace(" ", "")))
            ak, nm = contained_in(t, tt) if cfg.table_dup_aksharas else (0.0, 0.0)
            why = (
                "degenerate" if degenerate(t, "text")
                else "LaTeX" if LATEX_RE.search(t)
                else "read as a table: the grid continues there" if "<table" in t or TR.search(t)
                else "not mostly letters/digits" if alnum < 0.5
                else "low confidence" if lp is not None and lp < -0.6
                else "already in the table read" if dup >= 0.6
                else "already in the table read (aksharas / numbers)" if ak >= 0.5 or nm >= 0.5
                else None
            )  # fmt: skip
            dup_info = {"akshara_contained": round(ak, 2), "numbers_contained": round(nm, 2)} if cfg.table_dup_aksharas else {}
            parts.append(dict(inf, **rec, kept=why is None, why=why, dup_share=round(dup, 2), **dup_info))
            if why is None:
                extra[role] = t
        out = "\n".join(x for x in (extra.get("above grid"), tt, extra.get("below grid")) if x)
        return out, {"mode": "segments", "parts": parts}

    def vote_block(self, text: str, info: Trace, prompt: str, kind: str, box: Box, lp0: float) -> tuple[str, Trace]:
        """Two-of-three vote for one low-confidence block (see ``VoteConfig``). The two extra views go through the
        same rules as the original read; nothing changes if a view hit the cap, diverged (> 0.4 normalised edit
        distance) or has another line count. Log-prob is the trigger, not the selector."""
        v = self.cfg.vote
        assert v is not None
        x0, y0, x1, y1 = box
        views = []
        for name, (m, d) in (("pad", (v.pad, 0)), ("shift", (v.pad // 2, v.shift))):
            iw, ih = self.image.width, self.image.height
            bx = (max(0, x0 - m + d), max(0, y0 - m + d), min(iw, x1 + m + d), min(ih, y1 + m + d))
            t, inf = self.read_block(np.asarray(self.image.crop(bx)), prompt, kind)
            views.append({"view": name, "box": bx, "text": t, "info": inf})
        out = {
            "mode": "vote",
            "lp": round(lp0, 4),
            "original": info,
            "views": [{k: vv[k] for k in ("view", "box", "info")} for vv in views],
        }
        tb, tc = views[0]["text"], views[1]["text"]

        def lines(x: str) -> int:
            return len([line for line in x.split("\n") if line.strip()])

        if (
            any(r["hit"] for vv in views for r in final_reads(vv["info"]))
            or not tb.strip()
            or not tc.strip()
            or max(Levenshtein.normalized_distance(text, tb), Levenshtein.normalized_distance(text, tc)) > 0.4
            or not lines(text) == lines(tb) == lines(tc)
        ):
            return text, dict(out, applied=0, why="a view hit the cap, diverged (> 0.4) or has another line count")
        new, n = agreed_edits(text, tb, tc)
        return new, dict(out, applied=n)

    # -- the page ------------------------------------------------------------------------------------------- #
    def run(self) -> PageOutput:
        cfg, rt, image = self.cfg, self.rt, self.image
        blocks = [b.copy() for b in self.layout.blocks]
        dedup = rt.DedupConfig()
        eligible = rt.resolve_nested_equations([b for b in blocks if rt.is_transcribed(b.label)], dedup)
        eligible.sort(key=lambda b: b.order)
        traces: dict[int, Trace] = {}
        vote_cands = []
        for b in eligible:
            x0, y0, x1, y1 = (round(v) for v in b.bbox_xyxy)
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            arr = np.asarray(image.crop((x0, y0, x1, y1)))
            prompt = rt.prompt_for(b.type)
            kind = "table" if b.type == "Table" else "text"
            if not cfg.harness:  # the model card's single read
                texts, pieces = self.read_crops([arr], prompt, kind)
                b.text, traces[b.order] = texts[0], {"mode": "whole", "pieces": pieces}
                continue
            b.text, traces[b.order] = self.read_block(arr, prompt, kind)
            parts = traces[b.order].get("parts", [])
            lp = block_lp(traces[b.order])
            if (
                cfg.vote
                and lp is not None
                and lp < cfg.vote.max_lp
                and b.text.strip()
                and len(parts) == 1
                and parts[0].get("mode") in PLAIN_MODES
                and arr.shape[0] * arr.shape[1] <= self.crop_cfg.max_px
            ):
                vote_cands.append((lp, b.order, b, prompt, kind, (x0, y0, x1, y1)))
        for lp, _, b, prompt, kind, box in sorted(vote_cands, key=lambda x: x[:2])[: cfg.vote.max_blocks if cfg.vote else 0]:
            b.text, traces[b.order] = self.vote_block(b.text, traces[b.order], prompt, kind, box, lp)

        harvested: dict[str, list[str]] = {"header": [], "footer": []}
        if cfg.harvest_dates:
            # The model card skips header/footer blocks, and on records they often hold the date. Read the ones no
            # other block covers with the text prompt; keep one date per block, from a sound read only.
            for b in blocks:
                lab = b.label.lower()
                if lab in harvested and b.type not in rt.DROP_TYPES and covered_frac(b, eligible) < dedup.contain:
                    x0, y0, x1, y1 = (round(v) for v in b.bbox_xyxy)
                    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1)
                    if x1 - x0 < 8 or y1 - y0 < 8:
                        continue
                    texts, pieces = self.read_crops([np.asarray(image.crop((x0, y0, x1, y1)))], self.text_prompt, "text")
                    harvested[lab] += [] if degenerate(texts[0], "text") else dates_in(texts[0], raw=True)[:1]
                    traces[b.order] = {"mode": "harvest_dates", "read": texts[0], "dates": dates_in(texts[0]), "pieces": pieces}

        for b in blocks:
            if b.text is None:
                b.text = ""
            if b.text and cfg.dedupe and b.type != "Table":
                b.text = cut_macro_repeat(b.text)
            if b.text and cfg.script:
                b.text = to_script(b.text, cfg.script)
            if b.text and cfg.latex_plain:
                b.text = latex_to_plain(b.text)
        md = rt.reconstruct(blocks)
        out = PageOutput(blocks=blocks, markdown=md, traces=traces)
        if cfg.harvest_dates:
            new = harvest_filter(md, harvested, year_window=cfg.harvest_year_window)
            out.markdown = "\n".join(
                [" ".join(new["header"])] * bool(new["header"]) + [md] + [" ".join(new["footer"])] * bool(new["footer"])
            )
            out.harvested_dates = new
        if cfg.flags:
            out.flags = page_flags(blocks, out.markdown, traces, cell_counts=cfg.flags_cell_counts)
        return out


def run_page(image: Image, layout: Any, cfg: HarnessConfig, reader: Recognizer, rt: BodhanRuntime) -> PageOutput:
    """Read one page (PIL RGB image + its Bodhan layout) with one config."""
    return PageRun(image, layout, cfg, reader, rt).run()
