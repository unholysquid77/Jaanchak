"""No driver script may assign a bash special variable.

`GROUPS` holds the caller's group IDs and is special: bash silently ignores an
assignment to it, so `GROUPS="full_ft ln_only_ft ..."` followed by `$GROUPS` expands
to the primary gid. scripts/run_vit_2b.sh did exactly that and passed
`--baselines 1002` to run_baselines.py, which died on KeyError four minutes into
loading the backbone. `set -u` cannot catch it -- the variable is always set -- and
neither can `bash -n`, so it is asserted here.

The list is the special variables a driver script might plausibly reach for as an
ordinary name. PATH, HOME, IFS and friends are left out: shadowing those is sometimes
deliberate, and a script that does it is not making this mistake by accident.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.sh"))

SPECIAL = (
    "GROUPS",
    "RANDOM",
    "SECONDS",
    "LINENO",
    "UID",
    "EUID",
    "PPID",
    "BASHPID",
    "FUNCNAME",
    "PIPESTATUS",
    "REPLY",
    "OPTARG",
    "OPTIND",
    "SHLVL",
    "PWD",
    "OLDPWD",
)
ASSIGN = re.compile(
    r"^\s*(?:local\s+|export\s+|readonly\s+|declare\s+-\w+\s+)?(" + "|".join(SPECIAL) + r")="
)


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_no_special_variable_assignment(path: Path) -> None:
    offenders = [
        f"{path.name}:{i}: {line.strip()}"
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if ASSIGN.match(line) and not line.lstrip().startswith("#")
    ]
    assert not offenders, (
        "assignment to a bash special variable; bash may ignore it silently:\n"
        + "\n".join(offenders)
    )


def test_the_guard_actually_matches_the_bug_it_was_written_for(tmp_path: Path) -> None:
    """Without this, a typo in SPECIAL or ASSIGN would make the test vacuously pass."""
    bad = tmp_path / "bad.sh"
    bad.write_text('#!/usr/bin/env bash\nGROUPS="full_ft frozen_proto"\necho "$GROUPS"\n')
    with pytest.raises(AssertionError, match="special variable"):
        test_no_special_variable_assignment(bad)
