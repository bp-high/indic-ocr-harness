"""Training-free OCR harness around the open-weights Bodhan ``indic-ocr`` model.

Same two models and weights as the model card; the harness decides what the recognizer is shown (zoom, seam
strips, shifted views) and how its answers are assembled (verification against the model card's read, banned-token
repairs, votes, date harvesting). Built with IndicOCR from Bodhan AI / AI4Bharat.

>>> from indic_ocr_harness import Harness
>>> with Harness() as h:
...     print(h.read("page.jpg").text)
"""

from ._version import __version__
from .api import Harness, load_image
from .config import PLAIN, HarnessConfig, VoteConfig, ZoomConfig
from .errors import CacheMissError, HarnessError, IncompatibleEnvironmentError, ModelUnavailableError
from .model.bodhan import ATTRIBUTION, DEFAULT_REPO, PINNED_REVISION
from .model.cache import JsonFileCache, ReadCache
from .results import BlockResult, PageResult

__all__ = [
    "ATTRIBUTION",
    "DEFAULT_REPO",
    "PINNED_REVISION",
    "PLAIN",
    "BlockResult",
    "CacheMissError",
    "Harness",
    "HarnessConfig",
    "HarnessError",
    "IncompatibleEnvironmentError",
    "JsonFileCache",
    "ModelUnavailableError",
    "PageResult",
    "ReadCache",
    "VoteConfig",
    "ZoomConfig",
    "__version__",
    "load_image",
]
