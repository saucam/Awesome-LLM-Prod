#!/usr/bin/env python3
"""Align the pipes of every markdown table in README.md.

awesome-lint (remark-lint's table-pipe-alignment rule) requires all pipes
in a column to line up. Run with no arguments to rewrite README.md in
place; run with --check to exit non-zero if the file is not already
formatted (used by CI).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
SEPARATOR_CELL_RE = re.compile(r"^[\s:-]+$")


def format_table(lines):
    """Return the table lines rewritten with aligned, padded pipes."""
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    ncols = max(len(row) for row in rows)
    widths = [3] * ncols
    for row in rows:
        if all(SEPARATOR_CELL_RE.match(cell) for cell in row):
            continue
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    formatted = []
    for row in rows:
        is_separator = all(SEPARATOR_CELL_RE.match(cell) for cell in row)
        cells = []
        for index in range(ncols):
            cell = row[index] if index < len(row) else ""
            if is_separator:
                cells.append("-" * widths[index])
            else:
                cells.append(cell.ljust(widths[index]))
        formatted.append("| " + " | ".join(cells) + " |")
    return formatted


def main() -> int:
    check_only = "--check" in sys.argv
    readme = Path(__file__).resolve().parent.parent / "README.md"
    original = readme.read_text()

    output, table = [], []
    for line in original.splitlines():
        if TABLE_LINE_RE.match(line):
            table.append(line)
            continue
        if table:
            output.extend(format_table(table))
            table = []
        output.append(line)
    if table:
        output.extend(format_table(table))

    result = "\n".join(output) + "\n"
    if result == original:
        print("Tables already formatted.")
        return 0
    if check_only:
        print("Tables are not formatted. Run: python3 scripts/format_tables.py")
        return 1
    readme.write_text(result)
    print("Tables reformatted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
