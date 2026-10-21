"""Content-addressed store of recognizer reads.

Every read is keyed on (crop pixels, prompt, token budget, dtype tag, banned-token set), so configs that share crops
share work, an interrupted corpus run resumes, and a cached page can be re-read without the model.

A record: ``text``, ``ntok`` (tokens generated, EOS included), ``lp`` (mean token log-prob), ``hit`` (hit the
token cap or looped), ``loop``, ``sec``, and per token ``ids``, ``tlp`` (log-prob), ``tmargin`` (top-1 minus top-2
log-prob) and ``tent`` (entropy).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any

import numpy as np

Record = dict[str, Any]


def read_key(img: Any, prompt: str, max_new: int, tag: str = "") -> str:
    """Cache key of one read: SHA-1 of the crop pixels, shape, prompt, token budget and the tag.

    ``tag`` holds the dtype (``"|float32"``) and any banned-token set (``"|ban=latex"``); reads in the checkpoint's
    own dtype without a ban have an empty tag.
    """
    a = np.asarray(img)
    payload = a.tobytes() + str(a.shape).encode() + prompt.encode() + str(max_new).encode() + tag.encode()
    return hashlib.sha1(payload).hexdigest()


class ReadCache(MutableMapping[str, Record]):
    """In-memory read cache. Subclasses persist it; ``flush()`` is called after every recognizer batch."""

    def __init__(self, data: dict[str, Record] | None = None) -> None:
        self._data: dict[str, Record] = dict(data or {})

    def __getitem__(self, key: str) -> Record:
        return self._data[key]

    def __setitem__(self, key: str, value: Record) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def flush(self) -> None:
        """Persist pending writes (no-op in memory)."""


class JsonFileCache(ReadCache):
    """A read cache in one JSON file (``{key: record}``), rewritten atomically on flush.

    Simple and diff-friendly; suited to tens of thousands of reads. ``readonly=True`` never writes (reading from an
    existing cache only).
    """

    def __init__(self, path: str | os.PathLike[str], readonly: bool = False) -> None:
        self.path = Path(path)
        self.readonly = readonly
        self._dirty = False
        self._lock = threading.Lock()
        data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        super().__init__(data)

    def __setitem__(self, key: str, value: Record) -> None:
        super().__setitem__(key, value)
        self._dirty = True

    def flush(self) -> None:
        if self.readonly or not self._dirty:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
            self._dirty = False


def open_cache(spec: str | os.PathLike[str] | ReadCache | None, readonly: bool = False) -> ReadCache:
    """A cache from a path (``.json`` file), an existing cache, or None (in memory)."""
    if spec is None:
        return ReadCache()
    if isinstance(spec, ReadCache):
        return spec
    return JsonFileCache(spec, readonly=readonly)
