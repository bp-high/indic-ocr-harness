"""Command line: ``indic-ocr-harness read | show-config | download``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import HarnessConfig
from .model.bodhan import ATTRIBUTION, DEFAULT_REPO, PINNED_REVISION
from .model.recognizer import DTYPES

log = logging.getLogger("indic_ocr_harness")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="indic-ocr-harness",
        description="Training-free OCR harness around the open-weights Bodhan indic-ocr model.",
        epilog=ATTRIBUTION,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    model = argparse.ArgumentParser(add_help=False)
    model.add_argument("--model", default=None, help=f"snapshot directory or Hugging Face repo (default {DEFAULT_REPO})")
    model.add_argument("--revision", default=PINNED_REVISION, help="snapshot revision (default: the pinned one)")
    model.add_argument("--model-cache-dir", default=None, help="Hugging Face cache directory")
    model.add_argument("-v", "--verbose", action="count", default=0, help="-v: progress, -vv: every recognizer batch")

    r = sub.add_parser("read", parents=[model], help="read page images")
    r.add_argument("images", nargs="+", type=Path, help="page images")
    r.add_argument("--plain", action="store_true", help="every harness component off: the model card's pipeline")
    r.add_argument("--script", default=None, help="the pages' script, if not Devanagari (beng, guru, gujr, orya)")
    r.add_argument("--device", default="cpu")
    r.add_argument("--dtype", default="float32", choices=[*DTYPES, "checkpoint"], help="recognizer dtype (default float32)")
    r.add_argument("--no-loop-stop", action="store_true", help="let looping reads run to the token cap")
    r.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    r.add_argument("--cache", type=Path, default=None, help="read cache (.json); makes runs resumable")
    out = r.add_mutually_exclusive_group()
    out.add_argument("--out", type=Path, default=None, help="write all pages to one JSON file {page id: record}; resumes")
    out.add_argument("--out-dir", type=Path, default=None, help="write <page id>.md and <page id>.json per page")

    sub.add_parser("show-config", help="print the default config as JSON")

    sub.add_parser("download", parents=[model], help="download the Bodhan snapshot (gated: accept its license first)")
    return p


def _logging(verbose: int) -> None:
    level = logging.WARNING if not verbose else logging.INFO if verbose == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)


def _cmd_show_config(args: argparse.Namespace) -> int:
    print(json.dumps(HarnessConfig().describe(), indent=1))
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from .model.bodhan import resolve_snapshot

    print(resolve_snapshot(args.model, revision=args.revision, cache_dir=args.model_cache_dir))
    return 0


def _cmd_read(args: argparse.Namespace) -> int:
    from .api import Harness

    cfg = HarnessConfig.plain() if args.plain else HarnessConfig()
    if args.script:
        cfg = cfg.replace(script=args.script)
    done: dict = {}
    if args.out and args.out.exists():
        done = json.loads(args.out.read_text(encoding="utf-8"))
    ids = [img.stem for img in args.images]
    if len(set(ids)) != len(ids):
        log.error("page ids (file stems) must be unique")
        return 2
    todo = [(i, img) for i, img in zip(ids, args.images, strict=True) if i not in done]
    if len(todo) < len(ids):
        log.warning("%d of %d pages already in %s", len(ids) - len(todo), len(ids), args.out)
    failed = 0
    with Harness(
        cfg,
        model=args.model,
        revision=args.revision,
        model_cache_dir=args.model_cache_dir,
        device=args.device,
        dtype=None if args.dtype == "checkpoint" else args.dtype,
        loop_stop=not args.no_loop_stop,
        cache=args.cache,
        num_threads=args.threads,
    ) as h:
        for pid, img in todo:
            try:
                page = h.read(img)
            except Exception:
                log.exception("%s failed", img)
                failed += 1
                continue
            log.info("%s: %.0fs wall, %.0fs recognizer", pid, page.wall_seconds, page.recognizer_seconds)
            if args.out:
                done[pid] = page.to_record()
                args.out.parent.mkdir(parents=True, exist_ok=True)
                tmp = args.out.with_name(args.out.name + ".tmp")
                tmp.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
                tmp.replace(args.out)
            elif args.out_dir:
                args.out_dir.mkdir(parents=True, exist_ok=True)
                (args.out_dir / f"{pid}.md").write_text(page.text + "\n", encoding="utf-8")
                (args.out_dir / f"{pid}.json").write_text(page.to_json(indent=1), encoding="utf-8")
            else:
                if len(todo) > 1:
                    print(f"<!-- page: {pid} -->")
                print(page.text)
                sys.stdout.flush()
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _logging(getattr(args, "verbose", 0))
    from .errors import HarnessError

    try:
        return {"read": _cmd_read, "show-config": _cmd_show_config, "download": _cmd_download}[args.cmd](args)
    except HarnessError as e:
        log.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
