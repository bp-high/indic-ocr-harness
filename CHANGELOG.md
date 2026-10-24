# Changelog

## 0.2.0 (unreleased)

The default configuration changes how text in the wrong script is fixed. Pages whose reads contain no letters of
another Indic script give the same output as 0.1.0.

- **Fixed letter map** (`script_map_fixed`, default on).
  - A few code points 0x80 apart are not the same letter. They are now mapped to the letter they stand for:
    - Bengali and Oriya য়/ୟ → य (was य़);
    - ৎ → त् (was ॎ);
    - Assamese ৰ → र and ৱ/ୱ → व (were signs);
    - Gurmukhi tippi → ं and addak → a doubled consonant (were signs);
    - Gurmukhi iri/ura with a vowel sign → one independent vowel.
  - Two-part vowel signs are composed before mapping.
  - Signs with no Devanagari counterpart (currency, fractions) are kept as written.
  - New functions: `text.map_script` and `text.map_script_spans`. `to_script` keeps the 1:1 map.
- **More script re-reads kept** (`script_reread_gate`, default on).
  - Sometimes the recognizer writes a Hindi line as words of another language (Bengali words, not Hindi in Bengali
    letters). No letter map fixes that, and the re-read with the other scripts banned differs from the mapped read
    by more than half, so 0.1.0 dropped it.
  - Such a re-read is now also kept when two things hold (`text.script_reread_plausible`):
    - its length in letters is 0.5 to 2 times the first read's;
    - it adds at most 5 Latin letters, since a re-read that escapes into English is dropped.
  - The recognizer still chooses between the re-read and the letter map.
  - No extra reads: only a scoring pass for each re-read now kept.
- Set both fields to False for the 0.1.0 behaviour.

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
