# Publishing

The package is built with setuptools from `pyproject.toml`. It uses the src layout, and the version lives in
`src/indic_ocr_harness/_version.py`.

`.github/workflows/release.yml` publishes it to PyPI with
[trusted publishing](https://docs.pypi.org/trusted-publishers/), so no API token is stored anywhere.
`tools/release.py` keeps the version and the changelog consistent.

## One-time setup

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
   - On `pypi`, add yourself as a required reviewer. Every real release then waits for your approval after the
     TestPyPI install test.
   - On both, limit deployments to protected branches and tags. Add the tag rule `v*`.
4. **Tag protection (recommended).** Under Settings → Rules → New tag ruleset, target `v*` and restrict creation
   and deletion to admins.
5. **Test the pipeline.** Run it once manually; this publishes nothing real (see below).

If the repository moves (for example to an organisation), update the owner in both trusted publishers and the URLs
in `pyproject.toml` and `README.md`.

## Test run: a dev build to TestPyPI

Go to Actions → Release → Run workflow, on `main`. The workflow:

1. runs CI;
2. builds `X.Y.Z.dev<run number>`;
3. install-tests the wheel;
4. publishes it to TestPyPI;
5. installs it back from TestPyPI and checks it is byte-identical to the build.

It never touches PyPI or GitHub releases. Use it after the one-time setup, and whenever you change the workflow.

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
   5. on a public repository, attests build provenance;
   6. publishes to TestPyPI, then installs the package back from there and compares it with the build;
   7. waits for approval on the `pypi` environment, then publishes to PyPI;
   8. creates the GitHub release `vX.Y.Z`, with the changelog section as notes and the files attached.
6. **Smoke-test from PyPI:**

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install indic-ocr-harness==X.Y.Z
   indic-ocr-harness --version
   ```

## When something fails

- **Before PyPI** (CI, the tag checks, build, TestPyPI): nothing is released.
  - Fix the problem on `main`.
  - Move the tag: `git tag -d vX.Y.Z && git push origin :vX.Y.Z`, then tag again.
  - If the release was already uploaded to TestPyPI, bump the patch version instead, because TestPyPI never
    accepts the same file name twice.
- **After PyPI** (the GitHub release job): use "Re-run failed jobs". Uploads use `skip-existing`, so a re-run never
  uploads twice.
- **`invalid-publisher` from PyPI or TestPyPI:** the trusted publisher does not match. Check the owner, repository,
  workflow file name and environment name against the table above.
- **A broken release:** don't delete it from PyPI. Yank it on pypi.org (Manage → Release → Options → Yank) and
  release a fixed patch version.

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
