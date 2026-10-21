"""Result types returned by ``Harness.read``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import HarnessConfig


@dataclass(frozen=True)
class BlockResult:
    """One layout block with its final text.

    ``trace`` records what the harness did to the block (reads, splits, zooms and their verification, repairs,
    votes): a tree of plain dicts whose leaves are reads with token count, mean log-prob and whether they hit the
    cap. Blocks the model card does not transcribe (headers, footers, pictures) have empty text.
    """

    order: int
    label: str
    type: str
    bbox: tuple[float, float, float, float]
    conf: float
    text: str
    trace: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        """The Bodhan block record (``order, label, type, bbox_xyxy, conf, text``) plus ``harness`` (the trace)."""
        r: dict[str, Any] = {
            "order": self.order,
            "label": self.label,
            "type": self.type,
            "bbox_xyxy": [round(float(v), 1) for v in self.bbox],
            "conf": round(float(self.conf), 3),
            "text": self.text,
        }
        if self.trace:
            r["harness"] = self.trace
        return r


@dataclass(frozen=True)
class PageResult:
    """The reading of one page.

    Attributes:
        text: the page in reading order (Markdown; tables as HTML), the harness's output.
        blocks: the layout blocks with their text and traces.
        flags: the uncertainty sidecar (see ``pipeline.flags.page_flags``); empty when nothing was flagged.
        harvested_dates: header/footer dates added to the text, by block kind.
        recognizer_seconds: recognizer time of every read the page used (cached reads count their original time).
        wall_seconds: wall time of this call.
    """

    image: str
    width: int
    height: int
    text: str
    blocks: list[BlockResult]
    config: HarnessConfig
    flags: dict[str, list] = field(default_factory=dict)
    harvested_dates: dict[str, list[str]] = field(default_factory=dict)
    recognizer_seconds: float = 0.0
    wall_seconds: float = 0.0

    def to_record(self) -> dict[str, Any]:
        """A JSON-ready record of the page: text, timings, the config, blocks with their traces."""
        prep: dict[str, Any] = {}
        if self.harvested_dates:
            prep["harvested_dates"] = self.harvested_dates
        if self.flags:
            prep["flags"] = self.flags
        return {
            "text": self.text,
            "sec": round(self.wall_seconds, 1),
            "image": self.image,
            "prep": prep,
            "compute_sec": round(self.recognizer_seconds, 1),
            "config": self.config.describe(),
            "blocks": [b.to_record() for b in self.blocks],
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_record(), ensure_ascii=False, **kwargs)
