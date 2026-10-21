"""tools/release.py: version and changelog bookkeeping of the release workflow."""

import datetime
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("release", Path(__file__).resolve().parent.parent / "tools" / "release.py")
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)

CHANGELOG = """# Changelog

## 0.2.0 (unreleased)

- New thing.

## 0.1.0 (2026-10-21)

First release.
"""


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src/indic_ocr_harness").mkdir(parents=True)
    (tmp_path / "src/indic_ocr_harness/_version.py").write_text('__version__ = "0.1.0"\n')
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG)
    return tmp_path


def test_the_real_repo_is_consistent():
    v = release.read_version()
    assert release.FINAL.match(v) and release.release_date(v) is not None


def test_notes(repo):
    assert release.notes("0.2.0", repo) == "- New thing.\n"
    assert release.notes("0.1.0", repo) == "First release.\n"
    with pytest.raises(release.ReleaseError):
        release.notes("0.3.0", repo)


def test_prepare_dates_the_section_and_sets_the_version(repo):
    release.prepare("0.2.0", today=datetime.date(2026, 11, 2), root=repo)
    assert release.read_version(repo) == "0.2.0"
    assert release.release_date("0.2.0", repo) == "2026-11-02"
    with pytest.raises(release.ReleaseError, match="add a"):
        release.prepare("0.3.0", root=repo)
    with pytest.raises(release.ReleaseError, match=r"X\.Y\.Z"):
        release.prepare("0.3", root=repo)


def test_ci_version_for_tags(repo):
    assert release.ci_version("refs/tags/v0.1.0", 7, repo) == "0.1.0"
    with pytest.raises(release.ReleaseError, match="does not match"):
        release.ci_version("refs/tags/v0.2.0", 7, repo)
    release.write_version("0.2.0", repo)
    with pytest.raises(release.ReleaseError, match="dated section"):
        release.ci_version("refs/tags/v0.2.0", 7, repo)  # still "unreleased"


def test_ci_version_stamps_dev_builds(repo):
    assert release.ci_version("refs/heads/main", 42, repo) == "0.1.0.dev42"
    assert release.read_version(repo) == "0.1.0.dev42"
