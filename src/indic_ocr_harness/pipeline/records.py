"""Read records: what the harness keeps about each recognizer read of a block, and how they are aggregated.

A block's trace (``BlockResult.trace``) is a tree of plain dicts: leaves are reads (``piece``), inner nodes record a
decision (a split after a cap hit, a repair, a verified zoom, a vote). ``final_reads`` walks it to the reads whose
text ended up in the output, which is what confidence is measured on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

Node = dict[str, Any]


def piece(r: dict[str, Any], arr: np.ndarray) -> Node:
    """The trace leaf for one read of a crop of shape ``arr.shape``."""
    p = {
        "h": int(arr.shape[0]),
        "w": int(arr.shape[1]),
        "ntok": r["ntok"],
        "lp": r["lp"],
        "hit": r["hit"],
        "sec": r["sec"],
        "cached": r.get("cached", False),
        "seeded": "seeded" in r,
    }
    if r.get("ban"):
        p["ban"] = r["ban"]
    return p


def final_reads(node: Any) -> list[Node]:
    """The reads whose text ended up in a block's output.

    A split read counts as its halves, a repaired read as the first read (the repair splices spans into it), a
    verified split as the version kept, a vote as the original read; header-band reads only drop repeated rows and
    do not count, nor does an outside-grid region that was read but not kept.
    """
    if isinstance(node, list):
        return [x for v in node for x in final_reads(v)]
    if not isinstance(node, dict):
        return []
    if "ntok" in node:
        return [node]
    if node.get("kept") is False:
        return []
    if "split_after_hit" in node:
        return final_reads(node["split_after_hit"])
    if "band_only_reread" in node:
        return final_reads(node["band_only_reread"])
    if "repair" in node:
        return final_reads(node["first_read"])
    if node.get("mode") == "vote":
        return final_reads(node["original"])
    if node.get("mode") == "split_verify":
        return final_reads(node["split"] if node["kept"] == "split" else node["card"])
    return final_reads(node.get("parts", [])) + final_reads(node.get("pieces", []))


def mean_lp(reads: list[Node]) -> float | None:
    """Token-weighted mean log-prob of reads; None if there are none or any lacks a log-prob."""
    if not reads or any(r.get("lp") is None for r in reads):
        return None
    n = sum(r["ntok"] for r in reads)
    return sum(r["lp"] * r["ntok"] for r in reads) / n if n else None


def block_lp(trace: Any) -> float | None:
    """Token-weighted mean log-prob of the reads in a block's output."""
    return mean_lp(final_reads(trace))


def recognizer_seconds(node: Any) -> float:
    """Recognizer seconds of every read recorded anywhere in a trace."""
    if isinstance(node, dict):
        own = node.get("sec", 0.0) if "ntok" in node else 0.0
        return own + sum(recognizer_seconds(v) for v in node.values() if isinstance(v, (dict, list)))
    if isinstance(node, list):
        return sum(recognizer_seconds(v) for v in node)
    return 0.0
