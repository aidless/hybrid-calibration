#!/usr/bin/env python3
"""Offline check: recompute metrics and fail if the prose drifts.

No model download, no retraining, no network. Reads committed predictions,
historical JSON aggregates, and embedding matrices.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checks.render_observed import GITHUB_ABOUT, STATUS_LINE, render

README_START = "<!-- README-OBSERVED:START -->"
README_END = "<!-- README-OBSERVED:END -->"
DEEP_START = "<!-- DEEP-OBSERVED:START -->"
DEEP_END = "<!-- DEEP-OBSERVED:END -->"
PROBLEM_START = "<!-- PROBLEM:START -->"
PROBLEM_END = "<!-- PROBLEM:END -->"


def _between(text: str, start: str, end: str, label: str) -> str:
    if start not in text or end not in text:
        raise SystemExit(f"{label} is missing {start} / {end}")
    body = text.split(start, 1)[1].split(end, 1)[0]
    return body.strip("\n") + "\n"


def _replace(text: str, start: str, end: str, body: str) -> str:
    pre, rest = text.split(start, 1)
    _, post = rest.split(end, 1)
    return pre + start + "\n" + body.strip("\n") + "\n" + end + post


def _diff(name: str, expected: str, actual: str) -> str:
    return "".join(difflib.unified_diff(
        actual.splitlines(keepends=True),
        expected.splitlines(keepends=True),
        fromfile=f"{name} (committed)",
        tofile=f"{name} (recomputed)",
    ))


def _problem_lines(readme: str) -> list[str]:
    body = _between(readme, PROBLEM_START, PROBLEM_END, "README problem")
    lines = [ln for ln in body.splitlines() if ln.strip()]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite the observed blocks in place")
    args = parser.parse_args()

    if len(GITHUB_ABOUT) > 350:
        print(f"GitHub About text is {len(GITHUB_ABOUT)} characters; the field limit is 350")
        return 1

    blocks = render(ROOT)
    readme_path = ROOT / "README.md"
    deep_path = ROOT / "DEEP_DIVE.md"
    readme = readme_path.read_text(encoding="utf-8")
    deep = deep_path.read_text(encoding="utf-8")

    if args.write:
        readme_path.write_text(
            _replace(readme, README_START, README_END, blocks["readme"]),
            encoding="utf-8",
        )
        deep_path.write_text(
            _replace(deep, DEEP_START, DEEP_END, blocks["deep"]),
            encoding="utf-8",
        )
        print("wrote observed blocks")
        return 0

    failures = []
    got_readme = _between(readme, README_START, README_END, "README")
    if got_readme != blocks["readme"]:
        failures.append(_diff("README observed block", blocks["readme"], got_readme))
    got_deep = _between(deep, DEEP_START, DEEP_END, "DEEP_DIVE")
    if got_deep != blocks["deep"]:
        failures.append(_diff("DEEP_DIVE observed block", blocks["deep"], got_deep))

    problem = _problem_lines(readme)
    if len(problem) != 3:
        failures.append(f"problem statement has {len(problem)} non-empty lines, want 3\n")
    if STATUS_LINE not in readme.splitlines()[0:15] and STATUS_LINE not in readme:
        failures.append("README is missing the status line\n")
    if STATUS_LINE not in readme:
        failures.append("README status line missing\n")
    if GITHUB_ABOUT not in readme:
        failures.append("README is missing the GitHub About replacement text\n")
    if "NOT YET MEASURED" not in readme:
        failures.append("README does not say NOT YET MEASURED\n")
    if "768-dimensional GPT-2" in readme or "768-d GPT-2" in readme:
        failures.append("README still states 768-d GPT-2 as the study that was run\n")

    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    if "make check" not in ci:
        failures.append("ci.yml does not run make check\n")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    if "check:" not in makefile or "check_offline.py" not in makefile:
        failures.append("Makefile check target does not run checks/check_offline.py\n")

    modes = (ROOT / "FAILURE_MODES.md").read_text(encoding="utf-8")
    for heading in ("## Leakage", "## ECE binning", "## Small-n", "## Calibration on the test split"):
        if heading not in modes:
            failures.append(f"FAILURE_MODES.md missing {heading}\n")

    if failures:
        print("".join(failures))
        print(f"RESULT: FAIL ({len(failures)} checks)")
        return 1
    print("RESULT: PASS")
    print(f"GitHub About ({len(GITHUB_ABOUT)} chars) matches README")
    return 0


if __name__ == "__main__":
    sys.exit(main())
