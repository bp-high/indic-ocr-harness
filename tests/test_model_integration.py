"""Checks against the real Bodhan model. Skipped unless INDIC_OCR_HARNESS_MODEL names a local snapshot directory.

INDIC_OCR_HARNESS_MODEL=/path/to/bodhan-indic-ocr pytest -m model
"""

import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not os.environ.get("INDIC_OCR_HARNESS_MODEL"), reason="INDIC_OCR_HARNESS_MODEL not set"),
]


@pytest.fixture(scope="module")
def harness():
    from indic_ocr_harness import Harness

    with Harness(model=os.environ["INDIC_OCR_HARNESS_MODEL"], num_threads=os.cpu_count()) as h:
        yield h


def synthetic_page() -> Image.Image:
    img = Image.new("RGB", (900, 600), "white")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(["Meeting held on 14-3-26", "Total amount 4510", "Next meeting 23-6-25"]):
        d.text((80, 80 + 60 * i), line, fill="black", font_size=36)
    return img


def test_runtime_and_tokenizer(harness):
    rt = harness.runtime
    assert rt.TEXT_PROMPT.startswith("Transcribe") and rt.CropConfig().max_px == 1536**2
    harness.recognizer.backend  # loads the model and runs the tokenizer round-trip check  # noqa: B018


def test_read_records_token_statistics(harness):
    crop = synthetic_page().crop((60, 60, 860, 140))
    r = harness.recognizer.read([(crop, harness.runtime.TEXT_PROMPT)], max_new=64)[0]
    assert r["ntok"] > 0 and len(r["tlp"]) == len(r["tmargin"]) == len(r["tent"]) == r["ntok"]
    assert "14-3-26" in r["text"].replace(" ", "")
    again = harness.recognizer.read([(crop, harness.runtime.TEXT_PROMPT)], max_new=64)[0]
    assert again["cached"] and again["text"] == r["text"]


def test_read_page_end_to_end(harness):
    page = harness.read(np.asarray(synthetic_page()))
    assert "4510" in page.text
    assert page.to_record()["config"] == harness.config.describe() and page.blocks
    layout = harness.detect_layout(synthetic_page())
    assert harness.read(synthetic_page(), layout=layout).text == page.text
