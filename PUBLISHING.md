# Publishing

The package is built with setuptools from `pyproject.toml` (src layout, version in
`src/indic_ocr_harness/_version.py`). It is released to PyPI by GitHub Actions with
[trusted publishing](https://docs.pypi.org/trusted-publishers/), so no API token is stored anywhere.

## Before the first release

1. **Name.**
   - Check that `indic-ocr-harness` is free on PyPI and TestPyPI.
   - To rename, change `name` in `pyproject.toml`, the package directory `src/indic_ocr_harness`, the script entry
     and the imports in tests and docs.
   - Don't use "Bodhan" in the name: the Bodhan license (§8.4) grants no trademark use beyond the attribution notice.
2. **Repository.** The repository is https://github.com/bp-high/indic-ocr-harness (`main`); Actions run CI on every push
   and pull request.
3. **Trusted publishers.** Add one on pypi.org and one on test.pypi.org: account → Publishing → "Add a new pending
   publisher". Use:
   - owner `bp-high`, repository `indic-ocr-harness`;
   - workflow: `release.yml`;
   - environment: `pypi` and `testpypi` respectively.
4. **GitHub environments.** In the repository settings, create the environments `pypi` and `testpypi`. Protect
   `pypi` with required reviewers if you want a manual gate.

## Release steps

1. Update `src/indic_ocr_harness/_version.py` and `CHANGELOG.md`.
2. Check locally:

   ```bash
   pip install -e ".[dev]"
   ruff check . && ruff format --check . && pytest
   python -m build && twine check dist/*
   ```

3. Run the model-gated checks on a machine with the Bodhan snapshot:

   ```bash
   INDIC_OCR_HARNESS_MODEL=/path/to/snapshot pytest -m model
   ```

4. Commit, tag and push:

   ```bash
   git tag -a v0.1.0 -m "indic-ocr-harness 0.1.0" && git push origin v0.1.0
   ```

5. `release.yml` then:
   1. runs the tests;
   2. builds the sdist and wheel;
   3. checks that the tag matches `__version__`;
   4. publishes to TestPyPI;
   5. after the `pypi` environment gate, publishes to PyPI;
   6. attaches the files to a GitHub release.
6. Smoke-test the published package from TestPyPI:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ indic-ocr-harness==0.1.0
   indic-ocr-harness show-config
   ```

## Versioning

[Semantic versioning](https://semver.org/). Before 1.0:
- a minor release may change the API or the default config's output (`tests/test_config.py` pins the default);
- a patch release fixes bugs, packaging or speed without changing the default config's output.

## What a release must not contain

- Bodhan weights, tokenizer files or `idp_*.py` code. They are downloaded from the gated repository under its own
  license.
- Page images, transcriptions or outputs from real documents. They may contain personal data.
- Tokens. Hugging Face access comes from `HF_TOKEN` or the stored login at run time. PyPI uses trusted publishing.

`python -m build` packs only `src/`, `LICENSE`, `NOTICE`, `README.md` and `pyproject.toml`. Check the listing with
`tar tzf dist/*.tar.gz` and `unzip -l dist/*.whl` before the first upload.
