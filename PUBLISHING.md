# Publishing

The package is built with setuptools from `pyproject.toml`. It uses the src layout, and the version lives in
`src/indic_ocr_harness/_version.py`.

`.github/workflows/release.yml` publishes releases, and `tools/release.py` keeps the version and the changelog
consistent.

- **GitHub releases work now.** A `vX.Y.Z` tag publishes a GitHub release with the wheel, the sdist, `SHA256SUMS`
  and the changelog section as notes.
- **PyPI is opt-in.** After the one-time setup below, set the repository variable `PUBLISH_TO_PYPI` to `true`. The
  same tag then also publishes to TestPyPI and PyPI, using
  [trusted publishing](https://docs.pypi.org/trusted-publishers/), so no API token is stored anywhere.

## Releasing X.Y.Z

1. **Write the changelog.** In `CHANGELOG.md`, put the changes under a `## X.Y.Z (unreleased)` heading at the top.
2. **Prepare the release.** This sets `_version.py` and dates that heading:

   ```bash
   python tools/release.py prepare X.Y.Z
   ```

3. **Check locally.**

   ```bash
   pip install -e ".[dev]"
   ruff check . && ruff format --check . && pytest
   INDIC_OCR_HARNESS_MODEL=/path/to/bodhan-snapshot pytest -m model   # needs the Bodhan snapshot
   ```

4. **Commit, tag and push.** Put the tag on a commit that is on `main`:

   ```bash
   git commit -am "release: X.Y.Z"
   git tag -a vX.Y.Z -m "indic-ocr-harness X.Y.Z"
   git push origin main vX.Y.Z
   ```

5. **The tag starts `release.yml`.** It:
   1. runs CI (lint, tests on Python 3.11–3.13, the package-contents check);
   2. checks the tag: it must equal `v` + `__version__`, sit on `main`, and have a dated changelog section;
   3. builds the sdist and wheel reproducibly (file times from the commit) and runs `twine check`;
   4. installs the wheel in a clean environment and runs the CLI;
   5. on a public repository, attests build provenance (`gh attestation verify <file> -R bp-high/indic-ocr-harness`);
   6. with `PUBLISH_TO_PYPI=true` only:
      1. publishes to TestPyPI;
      2. installs the package back from there and compares it with the build;
      3. waits for approval on the `pypi` environment;
      4. publishes to PyPI;
   7. creates the GitHub release `vX.Y.Z`: the changelog section and install instructions as notes, with the wheel,
      the sdist and `SHA256SUMS` attached.
6. **Smoke-test the release:**

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install https://github.com/bp-high/indic-ocr-harness/releases/download/vX.Y.Z/indic_ocr_harness-X.Y.Z-py3-none-any.whl
   # or, once on PyPI: pip install indic-ocr-harness==X.Y.Z
   indic-ocr-harness --version
   ```

   Release assets of a private repository need a GitHub login. To download them, use
   `gh release download vX.Y.Z -R bp-high/indic-ocr-harness`.

## Test run

Go to Actions → Release → Run workflow, on `main`.
- **Always:** the run does CI, builds `X.Y.Z.dev<run number>` and install-tests the wheel.
- **With `PUBLISH_TO_PYPI=true`:** it also publishes that dev build to TestPyPI, installs it back and checks it is
  byte-identical to the build.

A manual run never publishes to PyPI or creates a GitHub release. Run one after the PyPI setup, and whenever you
change the workflow.

## Turning on PyPI (one-time setup)

1. **Name.**
   - Check that `indic-ocr-harness` is free on [PyPI](https://pypi.org/project/indic-ocr-harness/) and
     [TestPyPI](https://test.pypi.org/project/indic-ocr-harness/).
   - To rename, change `name` in `pyproject.toml`, the package directory `src/indic_ocr_harness`, the script entry,
     the project URLs and the imports in tests and docs.
   - Don't use "Bodhan" in the name: the Bodhan license (§8.4) grants no trademark use beyond the attribution notice.
2. **Trusted publishers.** On [pypi.org](https://pypi.org/manage/account/publishing/) and
   [test.pypi.org](https://test.pypi.org/manage/account/publishing/), add a "pending publisher" with:

   | Field | PyPI | TestPyPI |
   |---|---|---|
   | PyPI project name | `indic-ocr-harness` | `indic-ocr-harness` |
   | Owner | `bp-high` | `bp-high` |
   | Repository name | `indic-ocr-harness` | `indic-ocr-harness` |
   | Workflow name | `release.yml` | `release.yml` |
   | Environment name | `pypi` | `testpypi` |

3. **GitHub environments.** Under Settings → Environments, create `testpypi` and `pypi`.
   - On `pypi`, add yourself as a required reviewer. Every PyPI release then waits for your approval after the
     TestPyPI install test.
   - On both, limit deployments to protected branches and tags, and add the tag rule `v*`.
4. **Switch it on.** Under Settings → Secrets and variables → Actions → Variables, add `PUBLISH_TO_PYPI` = `true`.
5. **Test it.** Do a manual test run (above).

**Recommended:** protect release tags. Under Settings → Rules → New tag ruleset, target `v*` and restrict creation and
deletion to admins.

If the repository moves (for example to an organisation), update the owner in both trusted publishers and the URLs
in `pyproject.toml` and `README.md`.

## When something fails

- **Before anything is published** (CI, the tag checks, build): fix the problem on `main` and move the tag.

  ```bash
  git tag -d vX.Y.Z && git push origin :vX.Y.Z
  ```

  Then tag again. If the version was already uploaded to TestPyPI, bump the patch version instead, because TestPyPI
  never accepts the same file name twice.
- **After PyPI** (the GitHub release job): use "Re-run failed jobs". Uploads use `skip-existing`, so a re-run never
  uploads twice.
- **`invalid-publisher` from PyPI or TestPyPI:** the trusted publisher does not match. Check the owner, repository,
  workflow file name and environment name against the table above.
- **A broken release:**
  - On PyPI, don't delete it. Yank it (Manage → Release → Options → Yank) and release a fixed patch version.
  - On GitHub, mark the release as a pre-release or edit its notes, then release the fix.

## Versioning

[Semantic versioning](https://semver.org/). Before 1.0:
- a minor release may change the API or the default config's output (`tests/test_config.py` pins the default);
- a patch release fixes bugs, packaging or speed without changing the default config's output.

## What a release must not contain

- Bodhan weights, tokenizer files or `idp_*.py` code. They are downloaded from the gated repository under its own
  license.
- Page images, transcriptions or outputs from real documents. They may contain personal data.
- Tokens. Hugging Face access comes from `HF_TOKEN` or the stored login at run time. PyPI uses trusted publishing.

The wheel holds only the package; the sdist adds `tests/`, `LICENSE`, `NOTICE`, `README.md` and `pyproject.toml`.
CI fails the build if either contains model weights, images, read caches or Bodhan code.
