#!/usr/bin/env python3
"""Lint the README project tables.

Checks, for every markdown table in README.md:
  1. Every project row has exactly 4 columns (Name | Support | Tags | Description).
  2. The project link parses as [Name](url).
  3. Rows are in alphabetical order (case-insensitive, by project name).

Exits non-zero with a report of every violation, so CI fails on malformed
rows like a missing column or an entry inserted out of order.
"""

import re
import sys
from pathlib import Path

ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
LINK_RE = re.compile(r"^\[(?P<name>[^\]]+)\]\((?P<url>[^)\s]+)\)$")
SEPARATOR_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def main() -> int:
    readme = Path(__file__).resolve().parent.parent / "README.md"
    errors = []
    prev_name = None
    in_table = False

    for lineno, line in enumerate(readme.read_text().splitlines(), start=1):
        if SEPARATOR_RE.match(line):
            in_table = True
            prev_name = None
            continue

        match = ROW_RE.match(line)
        if not match:
            in_table = False
            continue
        if not in_table:  # header row
            continue

        cells = [cell.strip() for cell in match.group(1).split("|")]
        if len(cells) != 4:
            errors.append(
                f"line {lineno}: expected 4 columns, found {len(cells)}: {line.strip()[:80]}"
            )
            continue

        link = LINK_RE.match(cells[0])
        if not link:
            errors.append(f"line {lineno}: first cell is not a [Name](url) link: {cells[0][:80]}")
            continue

        if any(not cell for cell in cells):
            errors.append(f"line {lineno}: empty cell in row for {link.group('name')}")

        name = link.group("name").lower()
        if prev_name is not None and name < prev_name:
            errors.append(
                f"line {lineno}: '{link.group('name')}' is out of alphabetical order"
                f" (comes before '{prev_name}')"
            )
        prev_name = name

    if errors:
        print(f"README lint failed with {len(errors)} error(s):\n")
        for error in errors:
            print(f"  {error}")
        return 1

    print("README lint passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
