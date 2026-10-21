"""Read records, token budgets, coverage and the uncertainty sidecar."""

import types

from indic_ocr_harness.pipeline.flags import page_flags
from indic_ocr_harness.pipeline.page import covered_frac, token_budget
from indic_ocr_harness.pipeline.records import block_lp, final_reads, recognizer_seconds


def r(n, lp=-0.1, hit=False, sec=1.0):
    return {"ntok": n, "lp": lp, "hit": hit, "sec": sec}


def test_final_reads():
    info = {"mode": "segments", "parts": [{"mode": "strips", "pieces": [
        r(10), {"split_after_hit": [r(3), r(4)], "first_read": r(2048, hit=True)},
        {"repair": "latex", "accepted": True, "reread": r(7), "first_read": r(9)},  # spliced: the first read is the backbone
        {"band_only_reread": r(70), "first_read": r(5)}],
        "header_band_read": r(99)}]}  # fmt: skip
    assert [x["ntok"] for x in final_reads(info)] == [10, 3, 4, 9, 70]
    vote = {"mode": "vote", "original": info, "views": [{"view": "pad", "info": {"parts": [{"pieces": [r(5)]}]}}]}
    assert [x["ntok"] for x in final_reads(vote)] == [10, 3, 4, 9, 70]
    sv = {"mode": "split_verify", "kept": "card", "split": info, "card": r(11)}
    assert [x["ntok"] for x in final_reads(sv)] == [11]
    assert final_reads({"role": "below grid", "kept": False, "pieces": [r(5)]}) == []
    assert abs(block_lp({"pieces": [r(10, -0.2), r(30, -0.6)]}) - (-0.5)) < 1e-9
    assert block_lp({"pieces": [r(10, None)]}) is None
    assert recognizer_seconds(vote) == 10.0  # every read recorded, kept or not (9 in the original, 1 in a view)


def test_token_budget():
    assert token_budget(32 * 32, 2048) == 64  # floor
    assert token_budget(200 * 100, 2048) == 256
    assert token_budget(1536 * 1536, 2048) == 2048  # whole blocks keep the cap


def test_covered_frac():
    b = types.SimpleNamespace(bbox_xyxy=[0, 0, 100, 100])
    assert covered_frac(b, []) == 0.0
    assert covered_frac(b, [types.SimpleNamespace(bbox_xyxy=[0, 0, 50, 100])]) == 0.5
    assert covered_frac(b, [types.SimpleNamespace(bbox_xyxy=[-10, -10, 200, 200])]) == 1.0


def test_flags():
    rows = "".join(f"<tr><td>{i}</td><td>नाम</td><td>500</td></tr>" for i in range(1, 7))
    rows += '<tr><td colspan="2">कुल</td><td>3000</td></tr><tr><td>7</td><td>500</td></tr>'
    blk = types.SimpleNamespace(order=3, type="Table", text=f"<table>{rows}</table>")
    assert page_flags([blk], blk.text, cell_counts=True)["cell_count_outliers"] == [[3, 3, [7]]]
    assert "cell_count_outliers" not in page_flags([blk], blk.text)
    serial = "<table>" + "".join(f"<tr><td>{i}</td><td>x</td></tr>" for i in (1, 2, 3, 5, 6, 7)) + "</table>"
    sb = types.SimpleNamespace(order=1, type="Table", text=serial)
    assert page_flags([sb], serial)["serial_breaks"] == [[1, [[3, 5]]]]
    traces = {1: {"pieces": [{"ntok": 5, "degenerate": True}], "zoom_rejected": "lower confidence than the card read"}}
    f = page_flags([sb], "दिनांक 1/2/2020, 3/4/2021, 5/6/2020 और 7/8/2012", traces)
    assert f["degenerate"] == [1] and f["zoom_rejected"] == [[1, "lower confidence than the card read"]]
    assert f["date_outliers"] == [2012]
