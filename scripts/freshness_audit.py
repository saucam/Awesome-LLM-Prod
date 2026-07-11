#!/usr/bin/env python3
"""Audit every GitHub-hosted entry in README.md for staleness.

For each unique github.com repo link, queries the GitHub API and flags:
  - repos that no longer exist (404)
  - repos that were renamed/moved (the link only works via redirect)
  - archived repos
  - stale repos (no push in STALE_DAYS days)

Writes a markdown report to the path given by --output ONLY if problems
were found (so CI can condition issue-creation on the file's existence),
and always prints the report to stdout. Exits 0 either way; this is an
audit, not a gate.

Requires GITHUB_TOKEN in the environment for a usable rate limit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

STALE_DAYS = 180
REPO_RE = re.compile(r"https://github\.com/([\w.-]+)/([\w.-]+)")


def api_get(path: str, token: str | None):
    request = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "awesome-llm-prod-freshness-audit",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="write report here if problems found")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    readme = Path(__file__).resolve().parent.parent / "README.md"
    repos = sorted(set(match.group(1, 2) for match in REPO_RE.finditer(readme.read_text())))

    problems = []
    checked = 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=STALE_DAYS)

    for owner, name in repos:
        slug = f"{owner}/{name}"
        try:
            repo = api_get(f"/repos/{slug}", token)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                problems.append(f"| {slug} | **gone** | repo returns 404 |")
            else:
                problems.append(f"| {slug} | error | API returned HTTP {error.code} |")
            continue
        checked += 1

        full_name = repo.get("full_name", slug)
        if full_name.lower() != slug.lower():
            problems.append(
                f"| {slug} | **moved** | now lives at `{full_name}` — update the link |"
            )
        if repo.get("archived"):
            problems.append(f"| {slug} | **archived** | repo is archived on GitHub |")

        pushed_at = repo.get("pushed_at")
        if pushed_at:
            pushed = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
            if pushed < cutoff:
                days = (datetime.now(timezone.utc) - pushed).days
                problems.append(
                    f"| {slug} | **stale** | last push {days} days ago ({pushed.date()}) |"
                )

    lines = [
        f"Freshness audit of {len(repos)} GitHub entries ({checked} reachable).",
        "",
    ]
    if problems:
        lines += [
            "| Repo | Status | Detail |",
            "|------|--------|--------|",
            *problems,
            "",
            f"_Flags: moved link, archived, 404, or no push in {STALE_DAYS} days._",
        ]
    else:
        lines.append("No problems found — every entry is live, unmoved, and recently pushed.")

    report = "\n".join(lines)
    print(report)

    if problems and args.output:
        args.output.write_text(report + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
