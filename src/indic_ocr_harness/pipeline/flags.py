"""The uncertainty sidecar: page-level flags for review queues. Flags never change the text."""

from __future__ import annotations

import collections
import json
import re
from typing import Any

from ..text.dates import canon_date, dates_in
from ..text.scripts import DEVA_TO_ASCII_DIGITS
from ..text.tables import TR, row_text


def page_flags(blocks: list[Any], md: str, traces: dict[int, Any] | None = None, cell_counts: bool = False) -> dict[str, list]:
    """Flags of one page (``traces``: block order -> trace); only non-empty ones are returned.

    - ``degenerate``: blocks with a degenerate read that could not be split away (dropped lines)
    - ``zoom_rejected``: [block, reason] where a zoomed read lost to the card read
    - ``serial_breaks``: [block, [[n, next], ...]] where a register's serial numbers do not increase by one
    - ``date_outliers``: years more than 3 from the page's median date year
    - ``cell_count_outliers`` (``cell_counts``): [block, usual cell count, [row indices]] for table rows whose cell
      count (colspans added) differs from the table's usual count
    """
    flags: dict[str, list] = {
        "degenerate": [],
        "zoom_rejected": [],
        "serial_breaks": [],
        "date_outliers": [],
        "cell_count_outliers": [],
    }
    for b in blocks:
        js = json.dumps((traces or {}).get(b.order) or {})
        if '"degenerate": true' in js:
            flags["degenerate"].append(b.order)
        for w in re.findall(r'"zoom_rejected": "([^"]+)"', js):
            flags["zoom_rejected"].append([b.order, w])
        if b.type == "Table" and b.text:
            first = [re.sub(r"\D", "", row_text(r).split(" ")[0].translate(DEVA_TO_ASCII_DIGITS)) for r in TR.findall(b.text)]
            nums = [int(x) for x in first if x and len(x) <= 4]
            if len(nums) >= 5:
                breaks = [i for i in range(1, len(nums)) if nums[i] != nums[i - 1] + 1]
                if breaks and len(breaks) <= len(nums) // 2:
                    flags["serial_breaks"].append([b.order, [nums[i - 1 : i + 1] for i in breaks][:5]])
            if cell_counts:
                counts = [
                    sum(int(c or 1) for c in re.findall(r'<t[dh](?:\s+colspan="(\d+)")?[^>]*>', r)) for r in TR.findall(b.text)
                ]
                if len(counts) >= 5:
                    mode = collections.Counter(counts).most_common(1)[0][0]
                    odd = [i for i, c in enumerate(counts) if c != mode]
                    if odd and len(odd) <= len(counts) // 3:
                        flags["cell_count_outliers"].append([b.order, mode, odd[:10]])
    ys = [y for _, _, y in (x for x in (canon_date(d) for d in dates_in(md, raw=True)) if x)]
    if len(ys) >= 3:
        med = sorted(ys)[len(ys) // 2]
        flags["date_outliers"] = sorted({y for y in ys if abs(y - med) > 3})
    return {k: v for k, v in flags.items() if v}
