"""Command line (no model needed)."""

import json

import pytest

from indic_ocr_harness import HarnessConfig, __version__
from indic_ocr_harness.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and __version__ in capsys.readouterr().out


def test_show_config(capsys):
    assert main(["show-config"]) == 0
    assert json.loads(capsys.readouterr().out) == HarnessConfig().describe()


def test_read_reports_a_missing_snapshot(tmp_path, caplog):
    img = tmp_path / "p.png"
    img.write_bytes(b"")
    assert main(["read", str(img), "--model", str(tmp_path / "nope")]) == 1
    assert "no snapshot directory" in caplog.text


def test_read_rejects_unknown_option(tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["read", str(tmp_path / "p.png"), "--no-such-option"])
    assert e.value.code == 2
