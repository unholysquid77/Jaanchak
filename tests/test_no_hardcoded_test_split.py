"""No experiment script may reach for the test split without being told to.

The standing rule on this project is that every measurement is a validation
measurement; the test split is read once, for the final table, as an explicit act.
``run_baselines.py`` -- the only script that reports an accuracy -- used to sample
``splits["test"]`` unconditionally, so a 100-episode parameter-type table would have
burned the test set while looking exactly like every other validation run.

This is a source-level test rather than a behavioural one on purpose: the failure
mode is a literal, it is invisible in the output (a test-split accuracy is a
plausible number), and nothing short of reading the data would catch it at runtime.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "experiments").glob("*.py"))

# A literal subscript with the string "test" in it. ``splits[args.split]`` and
# ``splits[split_name]`` are fine -- the point is that the choice is a parameter.
HARDCODED = re.compile(r"""\[\s*["']test["']\s*\]""")


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_no_literal_test_split_subscript(path: Path) -> None:
    offenders = [
        f"{path.name}:{i}: {line.strip()}"
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if HARDCODED.search(line) and not line.lstrip().startswith("#")
    ]
    assert not offenders, (
        "test split read from a literal; make it a --split argument instead:\n"
        + "\n".join(offenders)
    )


def test_run_baselines_defaults_to_validation() -> None:
    """The safe split must be the default, so forgetting the flag cannot leak test."""
    src = (ROOT / "experiments" / "run_baselines.py").read_text()
    assert '"--split", default="val"' in src, (
        "run_baselines.py must default --split to val"
    )
