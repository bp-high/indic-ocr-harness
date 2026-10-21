# indic-ocr-harness

[![CI](https://github.com/bp-high/indic-ocr-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/bp-high/indic-ocr-harness/actions/workflows/ci.yml)

A training-free OCR harness around the open-weights [Bodhan `indic-ocr`](https://huggingface.co/bodhan-ai/indic-ocr)
model, for handwritten and scanned Indic records (Devanagari first).

The harness uses the model card's two models and weights unchanged: IndicDocLayout for layout and reading order, and
the 0.8B IndicBlockOCR recognizer. It changes what the recognizer is shown and how the answers are put together. Each
component follows the same pattern: a trigger, a candidate read, a check, and then it keeps the candidate or rolls
back to the model card's own read.

| Component | Trigger → candidate | Kept only if |
|---|---|---|
| Zoom | text lines under 32 px → read the crop enlarged (≤ 2×) | its token log-prob is within 0.1 of the card read's |
| Seam strips | a block the pixel budget would shrink below 32 px lines → strips cut at minimum-ink seams | it recovers clearly more text, numbers or rows |
| Guard | a read hits the token cap, loops, or is degenerate → split in two and re-read | — (junk is dropped) |
| Banned-token repair | LaTeX junk, another script's letters, or a stray digit script → re-read with those tokens banned | only the flagged spans are spliced in |
| Table text | free text above or below a table's grid → read with the text prompt | it passes the quality and duplicate gates |
| Vote | the page's 4 lowest-confidence blocks → two shifted views | an edit is applied where both views agree |
| Header dates | header/footer blocks the card skips → read | only dates the page lacks are added |
| Post-processing | LaTeX → plain text, script mapping, repeated passages cut | — |
| Flags | degenerate reads, rejected zooms, serial-number breaks, odd dates, odd cell counts | sidecar only; the text is unchanged |

## Install

Requires Python ≥ 3.11. Install torch first, for your hardware. The CPU wheels are much smaller:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install indic-ocr-harness
```

`transformers` is pinned to 5.7.0. Version 5.19 and later swap the recognizer's tokenizer for one that garbles Indic
text. The harness checks this when it loads the model and refuses to run with the wrong tokenizer.

**Model access.** The Bodhan repository on Hugging Face is gated.
1. Accept its license at https://huggingface.co/bodhan-ai/indic-ocr.
2. Give the harness a token: set `HF_TOKEN` or run `hf auth login`.
3. Download the model (≈ 1.8 GB) with `indic-ocr-harness download`. The harness pins the snapshot revision it is
   validated against.

## Use

```python
from indic_ocr_harness import Harness

with Harness() as h:  # default config, float32 on CPU
    page = h.read("page.jpg")  # a path, a PIL image or an RGB array
    print(page.text)  # reading-order Markdown; tables as HTML
    print(page.flags)  # e.g. {"serial_breaks": [[3, [[7, 9]]]]}
    for block in page.blocks:
        print(block.order, block.type, block.text[:40], block.trace["mode"] if block.trace else "")
```

```bash
indic-ocr-harness read page.jpg                                     # Markdown to stdout
indic-ocr-harness read scans/*.jpg --out outputs.json --cache reads.json -v
indic-ocr-harness read scans/*.jpg --out-dir out/ --script beng
indic-ocr-harness read page.jpg --plain                             # the model card's pipeline, for comparison
indic-ocr-harness show-config                                       # the default config as JSON
```

- **Corpus runs.** `--cache` stores every recognizer read, so an interrupted run resumes and configs that share
  crops share work. `--out` skips pages it already holds.
- **Options.**
  - `Harness(config=..., model=<local snapshot dir>, device="cuda", dtype="bfloat16", loop_stop=True, cache="reads.json", num_threads=4)`
  - `h.read(image, layout=<layout record or JSON>)` reuses or corrects a layout.
  - `h.detect_layout(image)` returns the layout on its own.
- **Configs.**
  - `HarnessConfig().replace(vote=None)` turns a component off.
  - `HarnessConfig.plain()` turns every component off: Bodhan's own pipeline through the harness's reader, for
    comparison (trailing loops trimmed to one copy).
  - `replace(script="beng")` switches the page script to Bengali; Gurmukhi, Gujarati and Oriya are also supported.
  - Every field is documented in [`config.py`](src/indic_ocr_harness/config.py).

### Speed

The harness costs about twice the model card's recognizer time on CPU. A dense handwritten page takes minutes on 4
cores.
- **`loop_stop`** (on by default) stops reads that are already looping, which saves most of the tokens a loop would
  spend before the token cap.
- **dtype.** `bfloat16` is 1.3–1.5× faster on CPUs with AMX and is the natural choice on GPUs. Its reads differ
  slightly from float32, so don't mix dtypes in one comparison.
- **Batching.** Crops are batched up to 8 per call. vLLM is not supported.

## Licenses and attribution

- **This package** is licensed under Apache-2.0 ([LICENSE](LICENSE)). It contains none of Bodhan's code or weights.
  It downloads them from Hugging Face and imports the inference code that ships with the model.
- **The Bodhan model** (weights, tokenizer and inference code) is under the
  [Bodhan AI Open Model License](https://huggingface.co/bodhan-ai/indic-ocr). If you make the model or a derivative
  available to others:
  - you must show the notice **"Built with IndicOCR from Bodhan AI / AI4Bharat."** (`indic_ocr_harness.ATTRIBUTION`);
  - hosting it as an API for third parties needs Bodhan AI's written approval (§3). Nonprofit, public-sector and
    academic users are exempt (§13);
  - the prohibited uses of §10 apply to everyone.

  Read the license itself before deploying.
- See [NOTICE](NOTICE).

## Limitations

- **Text.** The harness was developed on Devanagari handwriting and scans. Other scripts work through `script=`,
  but have been tested less.
- **Images.** Pages are read as stored: no EXIF rotation, no dewarping, no orientation detection.
- **Accuracy.** The remaining error is mostly recognition inside blocks (handwritten names, table cells). A
  training-free harness cannot fix that; fine-tuning the recognizer can.
- **Validation.** Only CPU and transformers were validated.

## Development

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest                       # model-free; the model-gated tests run when INDIC_OCR_HARNESS_MODEL is set
ruff check . && ruff format --check .
```

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains the package layout and the design rules.
- [PUBLISHING.md](PUBLISHING.md) covers releases.
