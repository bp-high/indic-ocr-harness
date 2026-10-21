"""The recognizer: Bodhan's ``HfRecognizer`` plus token log-probs, banned tokens, an online loop stop, teacher-forced
scoring and the read cache.

Decoding is the model card's: greedy, the card's prompt template, EOS taken from the tokenizer, batches left-padded.
The additions below change no read unless asked for:

- per-token log-prob, top-1 margin and entropy of every read (the confidence signals the harness gates on);
- ``ban``: a read may forbid every vocabulary token containing given characters (LaTeX, another script's letters,
  one digit script); the log-probs recorded stay those of the unconstrained model;
- ``loop_stop``: every 16 tokens, a row whose output already ends in a long loop is stopped and recorded exactly as a
  loop that ran to the cap (``hit`` and ``loop`` set, tail trimmed), so the harness's fallbacks run unchanged. It
  saves time only; the cache key does not include it;
- ``score``: teacher-forced log-probs of candidate texts for a crop, so the recognizer can arbitrate between
  readings.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import re
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..errors import CacheMissError, IncompatibleEnvironmentError
from ..text.loops import loop_tail, looped, trim_loop
from ..text.scripts import ban_chars
from .cache import ReadCache, Record, read_key

if TYPE_CHECKING:
    from PIL.Image import Image

    from .bodhan import BodhanRuntime

log = logging.getLogger(__name__)

DTYPES = ("float32", "bfloat16", "float16")
_TOKENIZER_PROBE = "दिनांक 20-5-25 को ग्राम संगठन की बैठक हुई। বাজার 4510"


class Recognizer:
    """Reads crops with the Bodhan recognizer; the model is loaded on the first read the cache cannot answer.

    Args:
        runtime: the loaded Bodhan snapshot.
        cache: where reads are stored and looked up.
        dtype: ``"float32"`` (default), ``"bfloat16"`` (faster on CPUs with AMX and on
            GPUs) or None for the checkpoint's own dtype (bf16, without a dtype tag in the cache key). Different
            dtypes give slightly different reads: never mix them in one comparison.
        device: torch device for the recognizer.
        max_new_tokens: the token cap of a read (the model card's 2,048).
        loop_stop: stop reads that are already looping (speed only).
        batch_size, batch_pixels: at most this many crops, and this many crop pixels, per generate() call.
        allow_compute: False makes a cache miss raise ``CacheMissError`` (cache-only runs, without the model).
    """

    def __init__(
        self,
        runtime: BodhanRuntime,
        cache: ReadCache,
        dtype: str | None = "float32",
        device: str = "cpu",
        max_new_tokens: int = 2048,
        loop_stop: bool = True,
        batch_size: int = 8,
        batch_pixels: int = 5_000_000,
        allow_compute: bool = True,
    ) -> None:
        if dtype is not None and dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES} or None, got {dtype!r}")
        self.rt = runtime
        self.cache = cache
        self.dtype = dtype
        self.device = device
        self.max_new = max_new_tokens
        self.loop_stop = loop_stop
        self.batch_size, self.batch_pixels = batch_size, batch_pixels
        self.allow_compute = allow_compute
        self.tag = f"|{dtype}" if dtype else ""
        self._backend: Any = None
        self._bans: dict[str, Any] = {}
        self._vocab: list[str] | None = None
        self._special: set[int] = set()
        self.stats = {"reads": 0, "cached": 0, "tokens": 0, "seconds": 0.0}

    # -- model ----------------------------------------------------------------------------------------------- #
    @property
    def backend(self) -> Any:
        if self._backend is None:
            if not self.allow_compute:
                raise CacheMissError("a read is not in the cache and allow_compute is False")
            if importlib.util.find_spec("torchvision") is None:
                raise IncompatibleEnvironmentError("the recognizer's image processor needs torchvision (install it with torch)")
            import torch

            t = time.time()
            rec = self.rt.recognizer_backend(self.max_new, self.device)
            if self.dtype:
                # Loaded in the checkpoint dtype and then cast.
                rec.model.to(getattr(torch, self.dtype))
            _check_tokenizer(rec.processor.tokenizer)
            self._backend = rec
            log.info("recognizer loaded (%s, %s) in %.0fs", self.dtype or "checkpoint dtype", self.device, time.time() - t)
        return self._backend

    def close(self) -> None:
        self.cache.flush()
        self._backend = None

    # -- cache ----------------------------------------------------------------------------------------------- #
    def key(self, img: Any, prompt: str, max_new: int | None = None, ban: str | None = None) -> str:
        return read_key(img, prompt, self.max_new if max_new is None else max_new, self.tag + (f"|ban={ban}" if ban else ""))

    def lookup(self, img: Any, prompt: str, budget: int, ban: str | None = None) -> tuple[str, Record | None]:
        """(key, cached read or None). A full-budget read that ended within ``budget`` tokens is the same read."""
        k = self.key(img, prompt, budget, ban)
        if k not in self.cache and budget < self.max_new:
            full = self.cache.get(self.key(img, prompt, self.max_new, ban))
            if full and not full["hit"] and (full["ntok"] or 0) <= budget:
                self.cache[k] = full
        return k, self.cache.get(k)

    # -- reading --------------------------------------------------------------------------------------------- #
    def read(
        self,
        items: Sequence[tuple[Image, str]],
        max_new: int | Sequence[int] | None = None,
        ban: str | Sequence[str | None] | None = None,
    ) -> list[Record]:
        """Read crops.

        Args:
            items: (PIL image, prompt) pairs.
            max_new: a token budget, or one per item (default: the cap).
            ban: None, a ban name (``'latex'``, ``'script:deva'``, ``'digits:ascii'``, ``'+'``-joined), or one per item.

        Returns one record per item (see ``cache``), with ``cached`` set. Items are batched only with items of the
        same budget and ban.
        """
        budgets = list(max_new) if isinstance(max_new, Sequence) else [max_new or self.max_new] * len(items)
        bans = list(ban) if isinstance(ban, (list, tuple)) else [ban] * len(items)
        keys = [self.lookup(im, p, b, bn)[0] for (im, p), b, bn in zip(items, budgets, bans, strict=True)]
        todo = sorted(
            {k: i for i, k in enumerate(keys) if k not in self.cache}.values(),
            key=lambda i: (budgets[i], bans[i] or "", items[i][0].size[0] * items[i][0].size[1]),
        )
        chunk: list[int] = []
        for i in [*todo, None]:
            px = sum(items[j][0].size[0] * items[j][0].size[1] for j in chunk)
            if chunk and (
                i is None
                or len(chunk) == self.batch_size
                or budgets[i] != budgets[chunk[0]]
                or bans[i] != bans[chunk[0]]
                or px + items[i][0].size[0] * items[i][0].size[1] > self.batch_pixels
            ):
                self._run(chunk, items, keys, budgets[chunk[0]], bans[chunk[0]])
                chunk = []
            if i is not None:
                chunk.append(i)
        self.stats["reads"] += len(items)
        self.stats["cached"] += len(items) - len(todo)
        return [dict(self.cache[k], cached=(i not in todo)) for i, k in enumerate(keys)]

    def _run(self, idx: list[int], items: Sequence[tuple[Image, str]], keys: list[str], max_new: int, ban: str | None) -> None:
        rec = self.backend
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

        banned = self._ban_ids(ban) if ban else None

        class GreedyLogProb(LogitsProcessor):
            """Records log p(chosen token), the top-1 margin and the entropy per step, without output_scores (which
            keeps the full 248k-way distribution per step). With a ban, banned tokens are removed before the argmax
            and the log-prob recorded is the chosen token's under the unconstrained distribution."""

            def __init__(self) -> None:
                self.steps: list[Any] = []
                self.margins: list[Any] = []
                self.ents: list[Any] = []

            def __call__(self, input_ids: Any, scores: Any) -> Any:
                raw = torch.log_softmax(scores.float(), -1)
                top2 = raw.topk(2, -1).values
                self.margins.append(top2[:, 0] - top2[:, 1])
                self.ents.append(-(raw.exp() * raw).sum(-1))
                if banned is None:
                    self.steps.append(raw.max(-1).values)
                    return scores
                scores[:, banned] = -float("inf")
                self.steps.append(raw.gather(-1, scores.argmax(-1, keepdim=True))[:, 0])
                return scores

        t = time.time()
        inputs = rec.processor(
            text=[rec._prompt(items[i][1]) for i in idx], images=[items[i][0] for i in idx], padding=True, return_tensors="pt"
        ).to(rec.model.device)
        plen = inputs["input_ids"].shape[-1]
        glp = GreedyLogProb()
        stops: dict[int, int] = {}
        criteria = None
        if self.loop_stop:
            tok = rec.processor.tokenizer

            class LoopStop(StoppingCriteria):
                """Every 16 tokens, stop each row whose output already ends in a long loop (``loop_tail``)."""

                def __call__(self, input_ids: Any, scores: Any, **kw: Any) -> Any:
                    g = input_ids[:, plen:]
                    done = torch.zeros(g.shape[0], dtype=torch.bool, device=input_ids.device)
                    for r in range(g.shape[0]):
                        if r in stops:
                            done[r] = True
                        elif (
                            g.shape[1] % 16 == 0
                            and g.shape[1] >= 64
                            and loop_tail(tok.decode(g[r, -300:].tolist(), skip_special_tokens=True))
                        ):
                            stops[r] = g.shape[1]
                            done[r] = True
                    return done

            criteria = StoppingCriteriaList([LoopStop()])
        with torch.inference_mode():
            seqs = rec.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                use_cache=True,
                eos_token_id=rec.eos_token_id,
                pad_token_id=rec.pad_token_id,
                logits_processor=LogitsProcessorList([glp]),
                stopping_criteria=criteria,
            )
        gen = seqs[:, plen:].cpu()

        def stack(xs: list[Any]) -> Any:
            return torch.stack(xs, 1).cpu() if xs else torch.zeros(len(idx), 0)

        lps, mgs, ents = stack(glp.steps), stack(glp.margins), stack(glp.ents)
        texts = rec.processor.batch_decode(gen, skip_special_tokens=True)
        dt = time.time() - t
        for r, i in enumerate(idx):
            ids = gen[r].tolist()
            n = ids.index(rec.eos_token_id) + 1 if rec.eos_token_id in ids else len(ids)
            if r in stops:  # stopped in a loop: recorded like a loop that ran to the cap
                n = min(n, stops[r])
                texts[r] = rec.processor.tokenizer.decode(ids[:n], skip_special_tokens=True)
            lp = lps[r, :n]
            text = texts[r].strip()
            loop = looped(text) or r in stops
            entry: Record = {
                "text": trim_loop(text) if loop else text,
                "ntok": n,
                "lp": round(float(lp.mean()) if len(lp) else 0.0, 4),
                "hit": rec.eos_token_id not in ids[:n] or loop,
                "sec": round(dt / len(idx), 2),
                "loop": loop,
                "ids": ids[:n],
                "tlp": [round(float(x), 3) for x in lp],
                "tmargin": [round(float(x), 3) for x in mgs[r, :n]],
                "tent": [round(float(x), 3) for x in ents[r, :n]],
            }
            if r in stops:
                entry["stopped_at"] = stops[r]
            if ban:
                entry["ban"] = ban
            self.cache[keys[i]] = entry
        self.cache.flush()
        ntok = max(self.cache[keys[i]]["ntok"] for i in idx)
        self.stats["tokens"] += sum(self.cache[keys[i]]["ntok"] for i in idx)
        self.stats["seconds"] += dt
        log.debug("batch of %d%s: %.1fs, max %d tokens", len(idx), f" ban {ban}" if ban else "", dt, ntok)

    def _ban_ids(self, name: str) -> Any:
        """Token ids whose decoded text contains any banned character ('+'-joined ban names); special tokens are
        never banned."""
        if name not in self._bans:
            import torch

            tok = self.backend.processor.tokenizer
            if self._vocab is None:
                self._vocab = [tok.decode([i]) for i in range(len(tok))]
                self._special = set(tok.all_special_ids)
            chars = set().union(*(ban_chars(p) for p in name.split("+")))
            ids = [i for i, s in enumerate(self._vocab) if i not in self._special and any(c in chars for c in s)]
            self._bans[name] = torch.tensor(ids, dtype=torch.long, device=self.backend.model.device)
            log.debug("ban %s: %d tokens", name, len(ids))
        return self._bans[name]

    # -- scoring --------------------------------------------------------------------------------------------- #
    def score(self, img: Image, prompt: str, texts: Sequence[str]) -> list[tuple[float, list[float]]]:
        """Teacher-forced log-probs of candidate transcriptions of one crop -> [(mean token log-prob including EOS,
        per-token log-probs)], one forward pass, no generation. Cached under a ``score:`` key.

        The vocabulary projection is applied to the answer positions only, 64 at a time: full-sequence logits for a
        2,000-token table would take gigabytes.
        """
        k = (
            "score:"
            + hashlib.sha1(
                (read_key(img, prompt, 0, self.tag) + json.dumps(list(texts), ensure_ascii=False)).encode()
            ).hexdigest()
        )
        if k in self.cache:
            return [(float(m), list(t)) for m, t in self.cache[k]["scores"]]
        rec = self.backend
        import torch

        tok = rec.processor.tokenizer
        p = rec._prompt(prompt)
        eos = tok.convert_ids_to_tokens(rec.eos_token_id)
        inputs = rec.processor(
            text=[p + t + eos for t in texts], images=[img] * len(texts), padding=True, return_tensors="pt"
        ).to(rec.model.device)
        p_ids = rec.processor(text=[p], images=[img], return_tensors="pt")["input_ids"][0].tolist()
        head = rec.model.get_output_embeddings()
        out: list[tuple[float, list[float]]] = []
        with torch.inference_mode():
            hidden = rec.model.model(**inputs).last_hidden_state  # after the final norm
            for r in range(len(texts)):
                ids, mask = inputs["input_ids"][r], inputs["attention_mask"][r]
                seq = ids[mask.bool()].tolist()
                if seq[: len(p_ids)] != p_ids:
                    raise RuntimeError("prompt tokens are not a prefix of prompt+text; cannot score")
                n = len(seq) - len(p_ids)  # answer tokens incl. EOS; rows are left- or right-padded
                end = int(mask.nonzero().max()) + 1
                h, tgt = hidden[r, end - n - 1 : end - 1], ids[end - n : end]
                tl = torch.cat(
                    [
                        torch.log_softmax(head(h[c : c + 64]).float(), -1).gather(-1, tgt[c : c + 64, None])[:, 0]
                        for c in range(0, n, 64)
                    ]
                )
                out.append((round(float(tl.mean()), 4), [round(float(x), 3) for x in tl]))
        self.cache[k] = {"scores": [[m, t] for m, t in out]}
        self.cache.flush()
        return out


def _check_tokenizer(tok: Any) -> None:
    """Fail fast if the installed transformers swaps the Sarvam tokenizer (transformers 5.19 substitutes a
    Qwen3_5Tokenizer that mis-encodes the vocabulary: garbled prompts, '▁' left in the output)."""
    ids = tok(_TOKENIZER_PROBE, add_special_tokens=False)["input_ids"]
    back = tok.decode(ids, skip_special_tokens=True)
    if re.sub(r"\s+", " ", back).strip() != _TOKENIZER_PROBE or "▁" in back:
        import transformers

        raise IncompatibleEnvironmentError(
            f"the recognizer's tokenizer ({type(tok).__name__}) does not round-trip Indic text under "
            f"transformers {transformers.__version__}; install transformers==5.7.0"
        )
