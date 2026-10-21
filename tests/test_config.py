"""The config: the default must stay exactly as pinned here; a change to it changes every page's output."""

import dataclasses

import pytest

from indic_ocr_harness.config import PLAIN, HarnessConfig, VoteConfig, ZoomConfig

DEFAULT = {
    "harness": True, "script": "deva", "token_budget": True, "guard": True, "degeneracy_check": True,
    "zoom": {"min_line": 32, "max_zoom": 2.0, "min_change": 1.2, "floor": True, "verify_margin": 0.1,
             "verify_len": None, "latex_ok": True},
    "strip_max_aspect": 8.0, "header_band": True, "split_verify": True,
    "table_text": True, "table_text_gated": True, "table_dup_aksharas": True,
    "latex_ban": True, "script_ban": True, "digit_ban": True,
    "vote": {"max_lp": -0.3, "pad": 24, "shift": 8, "max_blocks": 4},
    "dedupe": True, "latex_plain": True, "harvest_dates": True, "harvest_year_window": False,
    "flags": True, "flags_cell_counts": True,
}  # fmt: skip


def test_default_is_pinned():
    assert HarnessConfig().describe() == DEFAULT


def test_plain_turns_everything_off():
    p = HarnessConfig.plain()
    assert p is PLAIN and p.harness is False and p.zoom is None and p.vote is None
    d = p.describe()
    assert d["script"] is None and not any(v for k, v in d.items() if isinstance(v, bool))


def test_replace_copies():
    base = HarnessConfig()
    c = base.replace(vote=None)
    assert c.vote is None and base.vote == VoteConfig()
    assert c.replace(script="beng").script == "beng" and c.script == "deva"
    assert base.replace() == base
    z = base.replace(zoom=ZoomConfig(verify_len=(0.6, 1.8))).describe()["zoom"]
    assert z["verify_len"] == (0.6, 1.8) and z["verify_margin"] == 0.1


def test_validation():
    with pytest.raises(ValueError, match="script"):
        HarnessConfig(script="latn")
    with pytest.raises(ValueError, match="script_ban"):
        HarnessConfig(script=None)
    HarnessConfig(script=None, script_ban=False)
    with pytest.raises(ValueError, match="max_blocks"):
        HarnessConfig(vote=VoteConfig(max_blocks=-1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        HarnessConfig().vote = None  # type: ignore[misc]
