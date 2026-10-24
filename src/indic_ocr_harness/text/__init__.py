"""Model-free text rules: loops, degeneracy, LaTeX, scripts, alignment, tables and dates."""

from .align import agreed_edits, merge_spans, splice, word_spans
from .dates import canon_date, dates_in, harvest_filter
from .degeneracy import degenerate, strip_degenerate_lines
from .latex import LATEX_RE, latex_spans, latex_to_plain, suspect_latex
from .loops import cut_macro_repeat, loop_tail, looped, trim_loop
from .scripts import (
    ban_chars,
    count_numbers,
    foreign_letters,
    map_script,
    map_script_spans,
    minority_digits,
    script_reread_plausible,
    to_script,
)
from .tables import TR, contained_in, drop_repeated_rows, row_text

__all__ = [
    "LATEX_RE",
    "TR",
    "agreed_edits",
    "ban_chars",
    "canon_date",
    "contained_in",
    "count_numbers",
    "cut_macro_repeat",
    "dates_in",
    "degenerate",
    "drop_repeated_rows",
    "foreign_letters",
    "harvest_filter",
    "latex_spans",
    "latex_to_plain",
    "loop_tail",
    "looped",
    "map_script",
    "map_script_spans",
    "merge_spans",
    "minority_digits",
    "row_text",
    "script_reread_plausible",
    "splice",
    "strip_degenerate_lines",
    "suspect_latex",
    "to_script",
    "trim_loop",
    "word_spans",
]
