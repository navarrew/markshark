"""
Protection against "formula injection" in CSV and Excel files that MarkShark writes.

Why this exists
---------------
Student names come from roster files that MarkShark did not create (an LMS
export, a colleague's spreadsheet, ...).  If a name is ``=HYPERLINK("http://x")``
and we write it unchanged into a CSV, Excel treats that cell as a formula the
moment a teacher opens the file, and the formula can leak data or launch
programs.  This is a well-known attack on grade files ("CSV injection").

Two different fixes are needed because the two file types behave differently:

* **CSV** has no cell types, so Excel guesses.  Any text starting with
  ``=  +  -  @`` (or a tab / carriage return) may be run as a formula.  The
  standard defence is to put a single quote in front of the text, which Excel
  shows as plain text.  See ``neutralize_csv_cell``.

* **XLSX** does have cell types.  openpyxl (the library that writes our
  reports) marks a text value that starts with ``=`` as a formula, but text
  starting with ``+ - @`` is stored as ordinary text and is not executed.  So
  for XLSX the only dangerous case is ``=``, and we can fix it exactly by
  telling openpyxl "this is text".  See ``force_formulas_to_text``.

Only use ``force_formulas_to_text`` on workbooks that contain no *intentional*
formulas.  MarkShark's reports contain none (all numbers are computed in
Python).  If a future report adds real formulas, apply the fix to the specific
text cells instead of the whole workbook.

Example inputs and outputs (also checked in tests/test_safe_output.py)::

    neutralize_csv_cell('=1+1')        -> "'=1+1"
    neutralize_csv_cell('@SUM(A1)')    -> "'@SUM(A1)"
    neutralize_csv_cell('Smith')       -> 'Smith'
    neutralize_csv_cell('-12.5')       -> '-12.5'   (a plain negative number)
    neutralize_csv_cell(42)            -> 42        (non-text is never changed)
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping

# Characters that make Excel / LibreOffice / Google Sheets treat a CSV cell as a
# formula.  Tab and carriage return are included because some spreadsheet
# programs strip them and then look at the next character.
_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")


def _looks_like_plain_number(text: str) -> bool:
    """True for text such as ``-5`` or ``+3.2`` that is just a number.

    Scores and IDs must keep working, so a leading ``-`` or ``+`` on a real
    number is not treated as an attack.  ``float()`` also accepts words like
    ``nan`` and ``inf``, but those never start with a trigger character, so the
    caller never asks about them.
    """
    try:
        float(text)
    except ValueError:
        return False
    return True


def neutralize_csv_cell(value: Any) -> Any:
    """Return *value* made safe to write into a CSV file.

    Only text is touched.  Numbers, None and everything else are returned as-is
    so numeric columns stay numeric.  Text that is already protected (starts
    with ``'``) is not triggered, so applying this twice does nothing extra.
    """
    if not isinstance(value, str):
        return value
    # A single character such as "-" or "=" cannot form a formula; leaving it
    # alone keeps blank-answer markers and similar placeholders unchanged.
    if len(value) < 2:
        return value
    if value.startswith(_FORMULA_TRIGGERS) and not _looks_like_plain_number(value):
        return "'" + value
    return value


def neutralize_csv_row(row: Iterable[Any]) -> List[Any]:
    """Apply ``neutralize_csv_cell`` to every cell of a list-style CSV row."""
    return [neutralize_csv_cell(cell) for cell in row]


def neutralize_csv_dict(row: Mapping[str, Any]) -> dict:
    """Apply ``neutralize_csv_cell`` to every value of a dict-style CSV row.

    Column names (the keys) are left alone.
    """
    return {key: neutralize_csv_cell(val) for key, val in row.items()}


def neutralize_csv_dataframe(df):
    """Return a copy of a pandas DataFrame that is safe to pass to ``to_csv``.

    Only text columns are changed; numeric columns keep their type so number
    formatting options such as ``float_format`` still work.  The original
    DataFrame is not modified.
    """
    safe = df.copy()
    for column in safe.columns:
        # Skip numeric (b, i, u, f, c), datetime (M) and timedelta (m) columns.
        # Checking "not a number" rather than "== object" matters because newer
        # pandas versions store text in a dedicated string type, not "object".
        if safe[column].dtype.kind not in "biufcmM":
            safe[column] = safe[column].map(neutralize_csv_cell)
    return safe


def force_formulas_to_text(workbook) -> int:
    """Turn every openpyxl "formula" cell in *workbook* back into plain text.

    Call this just before ``workbook.save()``.  Returns the number of cells
    changed, which is handy for tests and log messages.  The visible text is
    unchanged (``=1+1`` still shows as ``=1+1``); it simply no longer runs.
    """
    changed = 0
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                # openpyxl sets data_type "f" for any string starting with "=".
                if cell.data_type == "f":
                    cell.data_type = "s"
                    changed += 1
    return changed
