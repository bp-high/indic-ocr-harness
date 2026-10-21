"""Harness configuration: typed settings.

Every component is one field, so a config can be read, diffed and ablated (``config.replace(vote=None)``).
``HarnessConfig()`` is the default configuration; ``HarnessConfig.plain()`` turns every component off, for comparison
with the model card's pipeline.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from .text.scripts import SCRIPT_BASE


@dataclass(frozen=True)
class ZoomConfig:
    """Line-height-aware scaling of each crop, verified against the model card's read.

    The model card shrinks a crop to its pixel budget whatever the text size. Here small handwriting is enlarged so
    a text line is at least ``min_line`` px tall in the recognizer input (32 px = one 32x32 vision token), up to
    ``max_zoom``; scale changes under ``min_change`` are not worth a re-read.
    """

    min_line: int = 32
    max_zoom: float = 2.0
    min_change: float = 1.2
    floor: bool = True
    """Round the zoomed size down when rounding up would cross the pixel budget."""
    verify_margin: float | None = 0.1
    """Keep a zoomed read only if its mean token log-prob is at most this much below the card read's (None: never
    verify)."""
    verify_len: tuple[float, float] | None = None
    """Optional length-ratio window for a zoomed read relative to the card read (None: no length check)."""
    latex_ok: bool = True
    """Do not reject a zoomed read for adding LaTeX: the plain-text rewrite already fixes it."""


@dataclass(frozen=True)
class VoteConfig:
    """Two-of-three vote on a page's lowest-confidence blocks.

    Blocks with mean token log-prob under ``max_lp`` (at most ``max_blocks`` per page, lowest first) are read from
    two more views: ``pad`` px of page margin, and half that margin shifted by ``shift`` px. A character edit is
    applied only where both views make it identically.
    """

    max_lp: float = -0.3
    pad: int = 24
    shift: int = 8
    max_blocks: int = 4


@dataclass(frozen=True)
class HarnessConfig:
    """One harness configuration. ``HarnessConfig()`` is the default; change fields with ``replace``.

    With ``harness=False`` the page goes through the model card's pipeline (one read per block at the card's crop
    scale, trailing loops trimmed) and every other field is ignored.
    """

    harness: bool = True
    script: str | None = "deva"
    """Expected script of the page (``deva``, ``beng``, ``guru``, ``gujr``, ``orya``); letters of the other
    ISCII-parallel scripts are mapped onto it. None: no script handling."""

    # reading a block
    token_budget: bool = True
    """Cap each crop's new tokens by its native pixel area, so small crops cannot loop for 2,048 tokens."""
    guard: bool = True
    """A read that hits the token cap, loops or is degenerate is split in two along a seam and re-read (2 levels)."""
    degeneracy_check: bool = True
    zoom: ZoomConfig | None = field(default_factory=ZoomConfig)
    strip_max_aspect: float | None = 8.0
    """Never cut a block into strips wider than this aspect ratio; read it as the model card does instead."""
    header_band: bool = True
    """Stack a tiled table's column-heading band above each later strip."""
    split_verify: bool = True
    """Keep a block's strips only if they recover clearly more than the unsplit read."""

    # tables
    table_text: bool = True
    """Read the free text above or below a table's ruled grid with the text prompt."""
    table_text_gated: bool = True
    """Add that text to the whole-table read only if it passes the quality and duplicate gates."""
    table_dup_aksharas: bool = True
    """Also count outside-grid text as a duplicate when its aksharas or numbers are in the table read."""

    # targeted re-reads with banned tokens
    latex_ban: bool = True
    script_ban: bool = True
    digit_ban: bool = True

    vote: VoteConfig | None = field(default_factory=VoteConfig)

    # page-level post-processing
    dedupe: bool = True
    """Cut a passage the decoder wrote twice."""
    latex_plain: bool = True
    """Rewrite LaTeX math as plain text."""
    harvest_dates: bool = True
    """Read header/footer blocks that no other block covers and add only the dates the page lacks."""
    harvest_year_window: bool = False
    flags: bool = True
    """Write the uncertainty sidecar (no text change)."""
    flags_cell_counts: bool = True
    """Flag table rows whose cell count differs from the table's usual count."""

    def __post_init__(self) -> None:
        if self.script is not None and self.script not in SCRIPT_BASE:
            raise ValueError(f"script must be one of {sorted(SCRIPT_BASE)} or None, got {self.script!r}")
        if self.harness and self.script_ban and self.script is None:
            raise ValueError("script_ban needs a script")
        if self.vote is not None and self.vote.max_blocks < 0:
            raise ValueError("vote.max_blocks must be >= 0")

    # -- construction ---------------------------------------------------------------------------------------- #
    @classmethod
    def plain(cls) -> HarnessConfig:
        """Every component off: the model card's pipeline through the harness's reader (see ``PLAIN``)."""
        return PLAIN

    def replace(self, **changes: Any) -> HarnessConfig:
        """A copy with some fields changed."""
        return dataclasses.replace(self, **changes)

    def describe(self) -> dict[str, Any]:
        """All fields as plain data (for logs and output records)."""
        return dataclasses.asdict(self)


PLAIN = HarnessConfig(
    harness=False,
    script=None,
    token_budget=False,
    guard=False,
    degeneracy_check=False,
    zoom=None,
    strip_max_aspect=None,
    header_band=False,
    split_verify=False,
    table_text=False,
    table_text_gated=False,
    table_dup_aksharas=False,
    latex_ban=False,
    script_ban=False,
    digit_ban=False,
    vote=None,
    dedupe=False,
    latex_plain=False,
    harvest_dates=False,
    flags=False,
    flags_cell_counts=False,
)
"""The model card's pipeline through the harness's reader (trailing decoding loops are trimmed to one copy)."""
