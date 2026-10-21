"""Release helper: version and changelog bookkeeping for local releases and the release workflow.

  python tools/release.py prepare 0.2.0    set the version, date its CHANGELOG section, print the git commands
  python tools/release.py version          print the version in src/indic_ocr_harness/_version.py
  python tools/release.py ci-version --ref <git ref> --run-number <n>
                                           release workflow: for a tag vX.Y.Z, check it matches the package version
                                           and has a dated CHANGELOG section; otherwise stamp a dev version
                                           X.Y.Z.dev<n> into the checkout (TestPyPI test builds). Prints version=...
  python tools/release.py notes 0.2.0      print the CHANGELOG section of a version (GitHub release notes)

The CHANGELOG has one ``## X.Y.Z (unreleased)`` or ``## X.Y.Z (YYYY-MM-DD)`` heading per version, newest first.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = Path("src/indic_ocr_harness/_version.py")
CHANGELOG = Path("CHANGELOG.md")
FINAL = re.compile(r"^\d+\.\d+\.\d+$")
VERSION_LINE = re.compile(r'^__version__ = "([^"]+)"$', re.M)


class ReleaseError(Exception):
    pass


def read_version(root: Path = ROOT) -> str:
    m = VERSION_LINE.search((root / VERSION_FILE).read_text(encoding="utf-8"))
    if not m:
        raise ReleaseError(f"no __version__ line in {VERSION_FILE}")
    return m[1]


def write_version(version: str, root: Path = ROOT) -> None:
    p = root / VERSION_FILE
    p.write_text(VERSION_LINE.sub(f'__version__ = "{version}"', p.read_text(encoding="utf-8")), encoding="utf-8")


def _heading(version: str) -> re.Pattern[str]:
    return re.compile(rf"^## {re.escape(version)} \((unreleased|\d{{4}}-\d{{2}}-\d{{2}})\)[ \t]*$", re.M)


def release_date(version: str, root: Path = ROOT) -> str | None:
    """The CHANGELOG date of a version, ``"unreleased"``, or None if it has no section."""
    m = _heading(version).search((root / CHANGELOG).read_text(encoding="utf-8"))
    return m[1] if m else None


def notes(version: str, root: Path = ROOT) -> str:
    """The body of a version's CHANGELOG section."""
    text = (root / CHANGELOG).read_text(encoding="utf-8")
    m = _heading(version).search(text)
    if not m:
        raise ReleaseError(f"CHANGELOG.md has no section for {version}")
    nxt = re.compile(r"^## ", re.M).search(text, m.end())
    return text[m.end() : nxt.start() if nxt else len(text)].strip() + "\n"


def prepare(version: str, today: datetime.date | None = None, root: Path = ROOT) -> None:
    """Set the package version and date its CHANGELOG section (which must exist, dated or unreleased)."""
    if not FINAL.match(version):
        raise ReleaseError(f"release versions are X.Y.Z, got {version!r}")
    date = release_date(version, root)
    if date is None:
        raise ReleaseError(f'add a "## {version} (unreleased)" section to CHANGELOG.md first')
    if date == "unreleased":
        p = root / CHANGELOG
        day = (today or datetime.date.today()).isoformat()
        p.write_text(_heading(version).sub(f"## {version} ({day})", p.read_text(encoding="utf-8"), count=1), encoding="utf-8")
    write_version(version, root)


def ci_version(ref: str, run_number: int, root: Path = ROOT) -> str:
    """The version a release-workflow run builds (see the module docstring)."""
    current = read_version(root)
    if ref.startswith("refs/tags/"):
        tag = ref.removeprefix("refs/tags/")
        if tag != f"v{current}":
            raise ReleaseError(f"tag {tag} does not match the package version {current}")
        if not FINAL.match(current):
            raise ReleaseError(f"tagged releases are X.Y.Z, got {current}")
        date = release_date(current, root)
        if date in (None, "unreleased"):
            raise ReleaseError(f"CHANGELOG.md needs a dated section for {current} (run tools/release.py prepare {current})")
        return current
    dev = f"{current}.dev{run_number}"
    write_version(dev, root)
    return dev


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version")
    p = sub.add_parser("prepare")
    p.add_argument("version")
    c = sub.add_parser("ci-version")
    c.add_argument("--ref", required=True)
    c.add_argument("--run-number", type=int, required=True)
    n = sub.add_parser("notes")
    n.add_argument("version")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "version":
            print(read_version())
        elif a.cmd == "prepare":
            prepare(a.version)
            print(f"Prepared {a.version}. Review the diff, then:")
            print(f'  git commit -am "release: {a.version}"')
            print(f'  git tag -a v{a.version} -m "indic-ocr-harness {a.version}"')
            print(f"  git push origin main v{a.version}")
        elif a.cmd == "ci-version":
            print(f"version={ci_version(a.ref, a.run_number)}")
        else:
            sys.stdout.write(notes(a.version))
    except ReleaseError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
