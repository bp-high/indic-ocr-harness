# Changelog

## 0.1.0 (2026-10-21)

First release.

- **Config.** One typed `HarnessConfig` with every component as a field; `HarnessConfig.plain()` turns every
  component off, for comparison with the model card's pipeline.
- **Python API.** `Harness.read` returns a `PageResult` with:
  - per-block traces;
  - the uncertainty sidecar (`flags`);
  - the header and footer dates that were added.
- **CLI.** `indic-ocr-harness read | show-config | download`. Runs are resumable through a persistent read cache.
- **Bodhan model.**
  - The snapshot is resolved at the pinned revision `cd50d301`. The model's code and weights are not vendored.
  - The tokenizer is checked at load time and refused if it does not round-trip Indic text
    (transformers 5.19 and later).
