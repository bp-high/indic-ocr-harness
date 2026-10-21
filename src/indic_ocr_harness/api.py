"""The public entry point: ``Harness``."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from .config import HarnessConfig
from .model.bodhan import PINNED_REVISION, BodhanRuntime, resolve_snapshot
from .model.cache import ReadCache, open_cache
from .model.recognizer import Recognizer
from .pipeline.page import run_page
from .pipeline.records import recognizer_seconds
from .results import BlockResult, PageResult

if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image

log = logging.getLogger(__name__)

ImageInput: TypeAlias = "str | os.PathLike[str] | Image | np.ndarray"
LayoutInput: TypeAlias = "dict[str, Any] | str | os.PathLike[str] | Any"


class Harness:
    """Bodhan ``indic-ocr`` with the harness around it.

    >>> from indic_ocr_harness import Harness
    >>> with Harness() as h:                      # default config, float32 on CPU
    ...     page = h.read("page.jpg")
    ...     print(page.text)

    Args:
        config: a ``HarnessConfig`` (default ``HarnessConfig()``; ``HarnessConfig.plain()`` for the model card's pipeline).
        model: a local snapshot directory, or a Hugging Face repo id (default ``bodhan-ai/indic-ocr``).
        revision: the snapshot revision to download (default: the pinned one).
        token: Hugging Face token for the gated repo (default: ``HF_TOKEN`` or the stored login). Never logged.
        model_cache_dir, local_files_only: passed to ``huggingface_hub.snapshot_download``.
        device: torch device for both models.
        dtype: recognizer dtype: ``"float32"`` (default), ``"bfloat16"``, ``"float16"``, or None for the checkpoint's.
        loop_stop: stop reads that are already looping (speed only; on by default).
        cache: read cache: a ``.json`` path (persistent, resumable), a ``ReadCache``, or None (in memory).
        max_new_tokens: token cap per read (the model card's 2,048).
        num_threads: torch CPU threads (default: torch's own choice). Sets a process-wide torch setting.
        allow_compute: False raises ``CacheMissError`` instead of running the recognizer (cache-only runs).
    """

    def __init__(
        self,
        config: HarnessConfig | None = None,
        *,
        model: str | os.PathLike[str] | None = None,
        revision: str | None = PINNED_REVISION,
        token: str | None = None,
        model_cache_dir: str | os.PathLike[str] | None = None,
        local_files_only: bool = False,
        device: str = "cpu",
        dtype: str | None = "float32",
        loop_stop: bool = True,
        cache: str | os.PathLike[str] | ReadCache | None = None,
        max_new_tokens: int = 2048,
        num_threads: int | None = None,
        allow_compute: bool = True,
    ) -> None:
        self.config = HarnessConfig() if config is None else config
        snap = resolve_snapshot(
            model, revision=revision, token=token, cache_dir=model_cache_dir, local_files_only=local_files_only
        )
        self.runtime = BodhanRuntime.load(snap)
        self.device = device
        if num_threads:
            import torch

            torch.set_num_threads(num_threads)
        self.cache = open_cache(cache, readonly=not allow_compute)
        self.recognizer = Recognizer(
            self.runtime,
            self.cache,
            dtype=dtype,
            device=device,
            max_new_tokens=max_new_tokens,
            loop_stop=loop_stop,
            allow_compute=allow_compute,
        )
        self._layout_model: Any = None
        log.info("harness on %s (snapshot %s)", device, snap)

    @classmethod
    def from_parts(
        cls, runtime: BodhanRuntime, recognizer: Recognizer, config: HarnessConfig | None = None, device: str = "cpu"
    ) -> Harness:
        """A harness over an already loaded runtime and recognizer (sharing one recognizer between configs, or a
        stand-in recognizer in tests)."""
        self = cls.__new__(cls)
        self.config = HarnessConfig() if config is None else config
        self.runtime, self.recognizer, self.cache, self.device = runtime, recognizer, recognizer.cache, device
        self._layout_model = None
        return self

    # -- lifecycle ------------------------------------------------------------------------------------------- #
    def close(self) -> None:
        """Persist the read cache and release both models."""
        self.recognizer.close()
        if self._layout_model is not None:
            self._layout_model.close()
            self._layout_model = None

    def __enter__(self) -> Harness:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- layout ---------------------------------------------------------------------------------------------- #
    def detect_layout(self, image: ImageInput) -> dict[str, Any]:
        """Layout of a page (blocks, labels, reading order) as a Bodhan layout record."""
        img, name = load_image(image)
        return self._layout(img, name).as_record()

    def _layout(self, img: Image, name: str, layout: LayoutInput | None = None) -> Any:
        rt = self.runtime
        if layout is None:
            if self._layout_model is None:
                self._layout_model = rt.layout_model(self.device)
            page = rt.detect(self._layout_model, img, name)
            record = page.as_record()
        elif isinstance(layout, dict):
            record = layout
        elif isinstance(layout, (str, os.PathLike)):
            record = json.loads(Path(layout).read_text(encoding="utf-8"))
        else:
            record = layout.as_record()
        # Normalised through the record form (boxes to 0.1 px), so a detected layout and a saved one read the same.
        return rt.PageResult.from_record(record)

    # -- reading --------------------------------------------------------------------------------------------- #
    def read(
        self,
        image: ImageInput,
        *,
        layout: LayoutInput | None = None,
        config: HarnessConfig | None = None,
        name: str | None = None,
    ) -> PageResult:
        """Read one page.

        Args:
            image: a path, a PIL image or an RGB ``uint8`` array.
            layout: a layout to use instead of detecting one: a Bodhan layout record (dict), a path to one as JSON,
                or a Bodhan ``PageResult``. The layout must be for this image at its full resolution.
            config: a different config for this call.
            name: the page's image name in the result (default: the file name).
        """
        t = time.time()
        cfg = self.config if config is None else config
        img, default_name = load_image(image)
        page = self._layout(img, name or default_name, layout)
        if (page.width, page.height) != img.size:
            raise ValueError(f"layout is for a {page.width}x{page.height} page, the image is {img.width}x{img.height}")
        out = run_page(img, page, cfg, self.recognizer, self.runtime)
        self.cache.flush()
        blocks = [
            BlockResult(
                order=b.order,
                label=b.label,
                type=b.type,
                bbox=tuple(b.bbox_xyxy),
                conf=b.conf,
                text=b.text or "",
                trace=out.traces.get(b.order),
            )
            for b in out.blocks
        ]
        return PageResult(
            image=page.image,
            width=page.width,
            height=page.height,
            text=out.markdown,
            blocks=blocks,
            config=cfg,
            flags=out.flags or {},
            harvested_dates=out.harvested_dates or {},
            recognizer_seconds=sum(recognizer_seconds(tr) for tr in out.traces.values()),
            wall_seconds=time.time() - t,
        )

    def read_many(self, images: Iterable[ImageInput], **kwargs: Any) -> Iterator[PageResult]:
        """Read pages one after another (models load once; the cache is flushed after every page)."""
        for image in images:
            yield self.read(image, **kwargs)


def load_image(image: ImageInput) -> tuple[Image, str]:
    """(PIL RGB image, name) from a path, a PIL image or an array."""
    from PIL import Image as PILImage

    if isinstance(image, (str, os.PathLike)):
        limit = PILImage.MAX_IMAGE_PIXELS
        PILImage.MAX_IMAGE_PIXELS = None  # large scans legitimately exceed PIL's decompression-bomb guard
        try:
            with PILImage.open(image) as im:
                return im.convert("RGB"), os.path.basename(os.fspath(image))
        finally:
            PILImage.MAX_IMAGE_PIXELS = limit
    if isinstance(image, PILImage.Image):
        filename = getattr(image, "filename", "")
        return image.convert("RGB"), os.path.basename(filename) if filename else "page"
    return PILImage.fromarray(image).convert("RGB"), "page"
