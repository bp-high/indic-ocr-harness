# Architecture

```
src/indic_ocr_harness/
  api.py              Harness: the facade (load models lazily, read pages, layouts in and out)
  config.py           HarnessConfig / ZoomConfig / VoteConfig; the default config and PLAIN
  results.py          PageResult / BlockResult (+ JSON records)
  cli.py              indic-ocr-harness read | show-config | download
  errors.py
  model/              everything that touches the Bodhan model
    bodhan.py         snapshot resolution (pinned revision, gated repo) + BodhanRuntime, the only adapter to Bodhan's code
    recognizer.py     Recognizer: batched greedy reads with log-probs, banned tokens, loop stop, teacher-forced scoring
    cache.py          content-addressed read cache (memory, JSON file)
  pipeline/           the harness algorithm
    page.py           PageRun: one page through one config; one method per component
    records.py        read traces: leaves, final reads, confidence
    flags.py          the uncertainty sidecar
  text/               model-free text rules (loops, degeneracy, LaTeX, scripts, alignment, tables, dates)
  image/              model-free image operations (line height, seam strips, grid extent)
```

## Layers and dependencies

`text/` and `image/` are pure functions over strings and numpy arrays. They import nothing from the rest of the
package and nothing heavy: numpy, OpenCV and rapidfuzz only. `pipeline/` composes them with a `Recognizer`.
`model/` owns torch, transformers and Bodhan's code. Importing the package does not import torch. The recognizer
loads on the first read the cache cannot answer, and the layout model on the first page without a given layout.

```
api ──> pipeline ──> text, image
  │        └──────> model.recognizer (interface: read, score, max_new)
  └──> model.bodhan, model.cache
```

## Design rules

- **Two models, unchanged.** The harness never fine-tunes, re-prompts beyond the model card's prompts, or swaps a
  model. It changes the crops it shows the recognizer and the way it assembles the answers.
- **Trigger → candidate → verify → accept or roll back.** No component changes a block without evidence of a
  problem. Every candidate is checked against the model card's own read of the same crop, and the fallback is always
  that read.
- **Bodhan's code is not vendored.** `BodhanRuntime.load` imports the `idp_*` modules that ship in the model
  repository, at a pinned revision. It binds the names the harness uses explicitly, so an incompatible snapshot fails
  at load time with one error. Bodhan's code is the model card's reference for crops, prompts, layout clean-up and
  page assembly, so `HarnessConfig.plain()` is the model card by construction.
- **Reproducible by construction.**
  - Configs are frozen dataclasses. `tests/test_config.py` pins the default config field by field.
  - The read cache key is stable, so a cached run re-reads without the model (`allow_compute=False`).
  - Layouts are normalised through their record form, so a detected layout and a saved one read the same.
  - The recognizer is loaded in the checkpoint dtype and then cast.
- **Everything recorded.** Each block's `trace` records what was tried and why it was kept or rejected. The
  record includes the token count, the mean log-prob and whether a read hit the token cap.
- **No global side effects.** Logging goes through `logging.getLogger("indic_ocr_harness")`; there are no prints. PIL's
  decompression-bomb limit is lifted only while a page is opened. `num_threads` is the only process-wide setting, and
  it is opt-in. The one deliberate exception: Bodhan's modules go on `sys.path`, so one snapshot per process.

## The read cache

Key: SHA-1 over the crop pixels, the shape, the prompt, the token budget, the dtype tag and the banned-token set.
- A read of a crop with a smaller budget reuses a full-budget read of the same crop that ended within that budget.
- Teacher-forced scores are cached under `score:` keys.
- `loop_stop` is not in the key. A stopped read is recorded as a loop that ran to the cap, and the harness's
  fallbacks treat the two identically.

`JsonFileCache` rewrites its file atomically after every recognizer batch. It suits tens of thousands of reads.
Implement `ReadCache` (a `MutableMapping` with `flush()`) for another store.

## Adding a component

1. Put the model-free logic in `text/` or `image/`, with tests.
2. Add a config field with a docstring. Its default is off until the component has been shown to help.
3. Call the component from `PageRun` behind the field, and record its decision in the block trace.
4. Turning a component on by default changes every page's output: update the pin in `tests/test_config.py` and
   note it in `CHANGELOG.md`.
