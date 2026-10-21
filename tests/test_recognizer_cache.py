"""Read cache and recognizer bookkeeping (no model: the cache answers or the recognizer is scripted)."""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from fakes import ScriptedRecognizer, fake_runtime
from indic_ocr_harness.errors import CacheMissError
from indic_ocr_harness.model.cache import JsonFileCache, ReadCache, open_cache, read_key
from indic_ocr_harness.model.recognizer import Recognizer


def test_key_layout():
    """The key is the sha1 of pixels, shape, prompt and budget, then the dtype tag and ban; existing cache files
    depend on it staying stable."""
    img, prompt = np.zeros((3, 4, 3), np.uint8), "p"
    old = hashlib.sha1(img.tobytes() + str(img.shape).encode() + prompt.encode() + str(2048).encode()).hexdigest()
    assert read_key(img, prompt, 2048) == old
    assert read_key(Image.fromarray(img), prompt, 2048) == old
    rec = Recognizer(fake_runtime(), ReadCache(), dtype=None)
    assert rec.key(img, prompt, 2048) == old
    assert rec.key(img, prompt, 2048, "latex") != old
    assert Recognizer(fake_runtime(), ReadCache()).key(img, prompt, 2048) == read_key(img, prompt, 2048, "|float32")


def test_lookup_reuses_a_full_budget_read_that_ended_early():
    img = Image.new("RGB", (10, 10))
    rec = Recognizer(fake_runtime(), ReadCache())
    rec.cache[rec.key(img, "p", 2048)] = {"text": "x", "ntok": 30, "lp": -0.1, "hit": False, "sec": 1.0}
    assert rec.lookup(img, "p", 64)[1]["text"] == "x"
    rec2 = Recognizer(fake_runtime(), ReadCache())
    rec2.cache[rec2.key(img, "p", 2048)] = {"text": "x", "ntok": 300, "lp": -0.1, "hit": False, "sec": 1.0}
    assert rec2.lookup(img, "p", 64)[1] is None  # it ran past the smaller budget: not the same read


def test_cache_only_without_model():
    img = Image.new("RGB", (10, 10))
    rec = Recognizer(fake_runtime(), ReadCache(), allow_compute=False)
    rec.cache[rec.key(img, "p", 2048)] = {"text": "cached", "ntok": 3, "lp": -0.1, "hit": False, "sec": 1.0}
    out = rec.read([(img, "p")])
    assert out[0]["text"] == "cached" and out[0]["cached"] is True
    with pytest.raises(CacheMissError):
        rec.read([(Image.new("RGB", (11, 10)), "p")])


def test_batches_by_budget_and_ban():
    rec = ScriptedRecognizer(lambda prompt, size, ban: f"{prompt}:{size[0]}:{ban}")
    imgs = [Image.new("RGB", (10 + i, 10)) for i in range(3)]
    out = rec.read([(imgs[0], "a"), (imgs[1], "a"), (imgs[0], "a")], max_new=[64, 128, 64], ban=[None, None, "latex"])
    assert [o["text"] for o in out] == ["a:10:None", "a:11:None", "a:10:latex"]
    assert [o["cached"] for o in out] == [False, False, False]
    again = rec.read([(imgs[0], "a")], max_new=[64])
    assert again[0]["cached"] is True and len(rec.calls) == 3


def test_json_file_cache_round_trip(tmp_path):
    p = tmp_path / "reads.json"
    c = JsonFileCache(p)
    c["k"] = {"text": "दिनांक", "ntok": 2}
    c.flush()
    assert json.loads(p.read_text(encoding="utf-8")) == {"k": {"text": "दिनांक", "ntok": 2}}
    assert JsonFileCache(p)["k"]["text"] == "दिनांक"
    ro = JsonFileCache(p, readonly=True)
    ro["k2"] = {}
    ro.flush()
    assert "k2" not in json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(open_cache(None), ReadCache) and isinstance(open_cache(p), JsonFileCache)


def test_dtype_validation():
    with pytest.raises(ValueError, match="dtype"):
        Recognizer(fake_runtime(), ReadCache(), dtype="int8")
