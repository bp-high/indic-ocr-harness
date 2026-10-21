"""The page pipeline end to end on a synthetic page, with a scripted recognizer instead of the model."""

import cv2
import numpy as np
import pytest

from fakes import TABLE_PROMPT, TEXT_PROMPT, ScriptedRecognizer, fake_runtime
from indic_ocr_harness import Harness, HarnessConfig, PageResult

W, H = 1200, 1600


def page_image() -> np.ndarray:
    img = np.full((H, W, 3), 255, np.uint8)
    # a ruled table at y 700-1100 with 5 rules and 3 columns, and a handwritten-like note under the grid
    for y in range(700, 1101, 100):
        cv2.line(img, (150, y), (1050, y), (0, 0, 0), 3)
    for x in (150, 450, 750, 1050):
        cv2.line(img, (x, 700), (x, 1100), (0, 0, 0), 3)
    for i in range(3):
        cv2.putText(img, f"note line {i}", (180, 1180 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2)
    return img


def layout(blocks):
    return {"image": "synthetic.png", "width": W, "height": H, "blocks": blocks}


TEXT_BLOCK = {"order": 1, "label": "text", "type": "Text", "bbox_xyxy": [100, 200, 500, 400], "conf": 0.9}
HEADER = {"order": 0, "label": "header", "type": "PageHeader", "bbox_xyxy": [100, 40, 700, 120], "conf": 0.9}
TABLE = {"order": 2, "label": "table", "type": "Table", "bbox_xyxy": [120, 640, 1080, 1340], "conf": 0.9}
TABLE_HTML = "<table><tr><td>1</td><td>राम</td><td>500</td></tr><tr><td>2</td><td>श्याम</td><td>700</td></tr></table>"


def make(answer, config=None, lp=-0.1):
    rec = ScriptedRecognizer(answer, lp=lp)
    return Harness.from_parts(fake_runtime(), rec, config), rec


def test_model_card_reads_each_transcribed_block_once():
    h, rec = make(lambda p, s, b: "बैठक हुई" if p == TEXT_PROMPT else TABLE_HTML, HarnessConfig.plain())
    page = h.read(page_image(), layout=layout([HEADER, TEXT_BLOCK, TABLE]))
    assert isinstance(page, PageResult)
    assert len(rec.calls) == 2  # the header is not transcribed by the model card
    assert page.text == "बैठक हुई\n\n" + TABLE_HTML
    assert page.flags == {} and page.harvested_dates == {}
    assert [b.text for b in page.blocks] == ["", "बैठक हुई", TABLE_HTML]
    assert page.blocks[1].trace["mode"] == "whole"


def test_post_processing_dates_and_table_text():
    def answer(prompt, size, ban):
        if prompt == TABLE_PROMPT:
            return TABLE_HTML
        if size[1] < 150:  # the header strip
            return "दिनांक 14-3-26 बैठक"
        if size[1] < 400:  # text block, or the note under the table grid
            return r"बैठक में $\frac{12}{7}$ हुई" if size[0] < 450 else "कुल जमा राशि का विवरण नीचे दिया गया है"
        return "?"

    h, _ = make(answer)
    page = h.read(page_image(), layout=layout([HEADER, TEXT_BLOCK, TABLE]))
    assert page.text.startswith("14-3-26\n")  # a header date the body lacks is added on top
    assert page.harvested_dates == {"header": ["14-3-26"], "footer": []}
    assert "12/7" in page.text and "$" not in page.text  # LaTeX rewritten as plain text
    table = page.blocks[2]
    parts = table.trace["parts"]
    assert parts[0]["role"] == "table" and any(p.get("role") == "below grid" for p in parts)
    assert table.text.startswith(TABLE_HTML)
    rec_dict = page.to_record()
    assert rec_dict["config"] == HarnessConfig().describe()
    assert rec_dict["blocks"][0]["harness"]["mode"] == "harvest_dates"


def test_vote_applies_only_edits_both_views_agree_on():
    first = {}

    def answer(prompt, size, ban):
        first.setdefault("size", size)  # the block's own crop is read first; the vote's views are other crops
        return "बश्ती बाई 4510" if size == first["size"] else "बस्ती बाई 4510"

    h, _ = make(answer, lp=-0.5)  # below the vote trigger (-0.3)
    page = h.read(np.full((H, W, 3), 255, np.uint8), layout=layout([TEXT_BLOCK]))
    assert page.text == "बस्ती बाई 4510"
    assert page.blocks[0].trace["mode"] == "vote" and page.blocks[0].trace["applied"] == 1
    h4, _ = make(answer, HarnessConfig(vote=None), lp=-0.5)
    first.clear()
    assert h4.read(np.full((H, W, 3), 255, np.uint8), layout=layout([TEXT_BLOCK])).text == "बश्ती बाई 4510"


def test_repair_splices_only_flagged_spans():
    def answer(prompt, size, ban):
        return "दिनांक 14-3-26 को बेठक हुई" if ban == "latex" else "दिनांक $14x=3x-6$ को बैठक हुई"

    h, rec = make(answer)
    page = h.read(np.full((H, W, 3), 255, np.uint8), layout=layout([TEXT_BLOCK]))
    assert page.text == "दिनांक 14-3-26 को बैठक हुई"  # the letter flip elsewhere in the re-read does not leak in
    assert any(b == "latex" for _, _, b in rec.calls)


def test_layout_must_match_the_image():
    h, _ = make(lambda p, s, b: "x")
    with pytest.raises(ValueError, match="layout is for"):
        h.read(np.full((100, 100, 3), 255, np.uint8), layout=layout([TEXT_BLOCK]))
