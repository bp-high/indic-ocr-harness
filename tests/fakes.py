"""Stand-ins for the Bodhan runtime and recognizer, so the pipeline runs without the gated model."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from types import SimpleNamespace

from indic_ocr_harness.model.cache import ReadCache
from indic_ocr_harness.model.recognizer import Recognizer

TEXT_PROMPT = "Transcribe the text in this image."
TABLE_PROMPT = "Convert this table image to HTML."


@dataclass
class Block:
    order: int
    label: str
    type: str
    bbox_xyxy: list[float]
    conf: float
    text: str | None = None

    def as_record(self) -> dict:
        r = {
            "order": self.order,
            "label": self.label,
            "type": self.type,
            "bbox_xyxy": [round(float(v), 1) for v in self.bbox_xyxy],
            "conf": round(float(self.conf), 3),
        }
        if self.text is not None:
            r["text"] = self.text
        return r

    @classmethod
    def from_record(cls, r: dict) -> Block:
        return cls(
            int(r["order"]), r["label"], r["type"], [float(v) for v in r["bbox_xyxy"]], float(r.get("conf", 1.0)), r.get("text")
        )

    def copy(self) -> Block:
        return dataclasses.replace(self, bbox_xyxy=list(self.bbox_xyxy))


@dataclass
class PageResult:
    image: str
    width: int
    height: int
    blocks: list[Block] = field(default_factory=list)

    def as_record(self) -> dict:
        return {"image": self.image, "width": self.width, "height": self.height, "blocks": [b.as_record() for b in self.blocks]}

    @classmethod
    def from_record(cls, r: dict, strict: bool = True) -> PageResult:
        return cls(r["image"], int(r["width"]), int(r["height"]), [Block.from_record(b) for b in r.get("blocks", [])])


@dataclass(frozen=True)
class CropConfig:
    min_px_side: int = 256
    max_px_side: int = 1536

    @property
    def min_px(self) -> int:
        return self.min_px_side**2

    @property
    def max_px(self) -> int:
        return self.max_px_side**2


@dataclass(frozen=True)
class DedupConfig:
    contain: float = 0.90


def area_clamp(image, cfg=None):
    cfg = cfg or CropConfig()
    w, h = image.size
    px = w * h
    if cfg.min_px <= px <= cfg.max_px:
        return image
    s = math.sqrt((cfg.min_px if px < cfg.min_px else cfg.max_px) / px)
    return image.resize((max(1, round(w * s)), max(1, round(h * s))))


def fake_runtime() -> SimpleNamespace:
    return SimpleNamespace(
        Block=Block,
        PageResult=PageResult,
        CropConfig=CropConfig,
        DedupConfig=DedupConfig,
        TEXT_PROMPT=TEXT_PROMPT,
        DROP_TYPES=frozenset({"Figure", "Picture"}),
        is_transcribed=lambda label: label.lower() not in {"header", "footer", "image", "diagram", "chart"},
        prompt_for=lambda t: TABLE_PROMPT if t == "Table" else TEXT_PROMPT,
        area_clamp=area_clamp,
        resolve_nested_equations=lambda blocks, cfg=None: list(blocks),
        reconstruct=lambda blocks: "\n\n".join(b.text for b in sorted(blocks, key=lambda b: b.order) if b.text),
        detect=lambda model, image, name: PageResult(name, image.width, image.height, []),
    )


class ScriptedRecognizer(Recognizer):
    """A Recognizer whose model is a function of (prompt, crop size, ban): every read is computed by ``answer``."""

    def __init__(self, answer, lp: float = -0.1, **kw):
        super().__init__(fake_runtime(), ReadCache(), **kw)
        self.answer, self.lp, self.calls = answer, lp, []

    def _run(self, idx, items, keys, max_new, ban):
        for i in idx:
            img, prompt = items[i]
            self.calls.append((prompt, img.size, ban))
            text = self.answer(prompt, img.size, ban)
            n = max(1, len(text) // 3)
            self.cache[keys[i]] = {"text": text, "ntok": n, "lp": self.lp, "hit": False, "sec": 0.01, "loop": False}

    def score(self, img, prompt, texts):
        return [(-0.5, []) for _ in texts]
