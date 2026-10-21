"""Adapter to the Bodhan ``indic-ocr`` snapshot: the only module that touches Bodhan's own code.

The model's inference code ships inside its Hugging Face repository (``idp_*.py`` next to the weights) under the
Bodhan AI Open Model License, and the repository is gated. This package does not vendor that code. It resolves a
snapshot, a local directory or a download at a pinned revision, and imports the snapshot's modules from there.
Bodhan's modules import each other by top-level name (``from idp_types import ...``), so the snapshot directory is
put on ``sys.path``. Only one snapshot can be loaded per process.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ModelUnavailableError

log = logging.getLogger(__name__)

DEFAULT_REPO = "bodhan-ai/indic-ocr"
PINNED_REVISION = "cd50d301d0e17e8ecb32fc49c8ccbd7914dcfc25"
"""The snapshot revision the harness is validated against."""

ATTRIBUTION = "Built with IndicOCR from Bodhan AI / AI4Bharat."
"""The attribution notice the Bodhan AI Open Model License (Exhibit A) requires wherever the model is made
available to others."""

_MODULES = ("idp_contract", "idp_types", "idp_blocks", "idp_crops", "idp_reconstruct", "idp_offline", "idp_recognizer")
_REQUIRED = ("idp_types.py", "idp_offline.py", "weights/ocr/config.json", "weights/layout/config.json")


def resolve_snapshot(
    model: str | os.PathLike[str] | None = None,
    revision: str | None = PINNED_REVISION,
    token: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    local_files_only: bool = False,
) -> Path:
    """Path to a Bodhan snapshot: ``model`` if it is a directory, else a Hugging Face download of repo ``model``.

    The repository is gated: accept its license on Hugging Face first, then pass ``token`` or set ``HF_TOKEN``
    (or log in with ``hf auth login``). The token is passed through to ``huggingface_hub`` and never logged.
    """
    model = DEFAULT_REPO if model is None else model
    path = Path(model).expanduser()
    looks_local = isinstance(model, os.PathLike) or str(model).startswith((".", "~", os.sep)) or str(model).count("/") > 1
    if path.is_dir():
        snap = path
    elif looks_local:
        raise ModelUnavailableError(f"no snapshot directory at {path}")
    else:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

        try:
            snap = Path(
                snapshot_download(
                    str(model),
                    revision=revision,
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            )
        except (HfHubHTTPError, LocalEntryNotFoundError, OSError, ValueError) as e:
            raise ModelUnavailableError(
                f"cannot get {model}@{revision}: {type(e).__name__}. The repository is gated: accept its license on "
                "https://huggingface.co/bodhan-ai/indic-ocr, then set HF_TOKEN or run `hf auth login`."
            ) from e
    missing = [p for p in _REQUIRED if not (snap / p).exists()]
    if missing:
        raise ModelUnavailableError(f"{snap} is not a Bodhan indic-ocr snapshot (missing {', '.join(missing)})")
    return snap


@dataclass(frozen=True)
class BodhanRuntime:
    """The parts of Bodhan's inference code the harness uses, imported from one snapshot.

    Binding them by name at load time makes a snapshot with a different API fail here, with one clear error,
    instead of deep inside a page.
    """

    snapshot: Path
    # types and settings (idp_types)
    Block: Any
    PageResult: Any
    CropConfig: Any
    DedupConfig: Any
    LayoutConfig: Any
    RecognizerConfig: Any
    # prompts and block taxonomy (idp_contract)
    TEXT_PROMPT: str
    DROP_TYPES: frozenset
    is_transcribed: Any
    prompt_for: Any
    # page geometry and assembly
    area_clamp: Any
    resolve_nested_equations: Any
    reconstruct: Any
    # the two models
    IndicDocLayout: Any
    HfRecognizer: Any

    @classmethod
    def load(cls, snapshot: str | os.PathLike[str]) -> BodhanRuntime:
        snap = Path(snapshot).resolve()
        loaded = sys.modules.get("idp_types")
        if loaded is not None and Path(loaded.__file__ or "").resolve().parent != snap:
            raise ModelUnavailableError(
                f"Bodhan modules from {Path(loaded.__file__ or '').parent} are already imported; one snapshot per process"
            )
        if str(snap) not in sys.path:
            sys.path.insert(0, str(snap))
        m = {name: importlib.import_module(name) for name in _MODULES}
        try:
            return cls(
                snapshot=snap,
                Block=m["idp_types"].Block,
                PageResult=m["idp_types"].PageResult,
                CropConfig=m["idp_types"].CropConfig,
                DedupConfig=m["idp_types"].DedupConfig,
                LayoutConfig=m["idp_types"].LayoutConfig,
                RecognizerConfig=m["idp_types"].RecognizerConfig,
                TEXT_PROMPT=m["idp_contract"].TEXT_PROMPT,
                DROP_TYPES=m["idp_contract"].DROP_TYPES,
                is_transcribed=m["idp_contract"].is_transcribed,
                prompt_for=m["idp_contract"].prompt_for,
                area_clamp=m["idp_crops"].area_clamp,
                resolve_nested_equations=m["idp_blocks"].resolve_nested_equations,
                reconstruct=m["idp_reconstruct"].reconstruct,
                IndicDocLayout=m["idp_offline"].IndicDocLayout,
                HfRecognizer=m["idp_recognizer"].HfRecognizer,
            )
        except AttributeError as e:
            raise ModelUnavailableError(f"{snap} has an incompatible Bodhan API ({e}); use revision {PINNED_REVISION}") from e

    # -- the two models ------------------------------------------------------------------------------------- #
    def layout_model(self, device: str = "cpu") -> Any:
        """IndicDocLayout with the model card's settings."""
        return self.IndicDocLayout(
            ckpt=str(self.snapshot / "weights" / "layout"),
            config=self.LayoutConfig(device=device),
            dedup=self.DedupConfig(),
        )

    def recognizer_backend(self, max_new_tokens: int, device: str = "cpu") -> Any:
        """The model card's ``HfRecognizer`` (same weights, prompt template and EOS fix), in its checkpoint dtype."""
        return self.HfRecognizer(
            ckpt=str(self.snapshot / "weights" / "ocr"),
            config=self.RecognizerConfig(max_tokens=max_new_tokens),
            device=device,
        )

    def detect(self, layout_model: Any, image: Any, name: str) -> Any:
        """Layout of a PIL page image -> Bodhan ``PageResult`` (as ``IndicDocLayout.detect`` does for a path)."""
        return self.PageResult(image=name, width=image.width, height=image.height, blocks=layout_model.backend.detect(image))
