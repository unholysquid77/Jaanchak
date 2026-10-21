"""Guards on what a committed result file promises about itself.

These exist because of a specific wrong turn: reading
``manifest.config.inner_lr`` from a k=20 measurement gave 1e-04, which is
*attention's* tuned optimum, and for several minutes it looked like the --tuned
path was measuring every baseline at the wrong learning rate -- i.e. that every
headline number in the project was suspect.

It was not. ``manifest.config`` is ``vars(args)``, so its ``inner_lr`` is whatever
the CLI default was (1e-04), echoed identically into every baseline's file. The
learning rate actually used is the TOP-LEVEL ``inner_lr``, written from
``runner.inner_lr`` after the tuned setting is applied. In results/vit_2b_eurosat
all eight baselines say ``manifest.config.inner_lr == 1e-04`` while their
top-level values range over 1e-06 .. 0.03.

So: the top-level field is authoritative, the nested one is a default, and the
misleading one is nested where a reader naturally looks for configuration. These
tests pin that distinction rather than relying on anyone remembering it.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _result_files() -> list[str]:
    return sorted(glob.glob(str(ROOT / "results" / "*" / "*.json")))


@pytest.fixture(scope="module")
def results() -> list[dict]:
    files = _result_files()
    if not files:
        pytest.skip("no committed result files")
    out = []
    for f in files:
        try:
            j = json.load(open(f))
        except json.JSONDecodeError:
            continue
        if "episodes" in j and "baseline" in j:
            out.append({"path": f, "json": j})
    if not out:
        pytest.skip("no per-episode result files")
    return out


def test_every_result_records_the_effective_learning_rate(results) -> None:
    """Top-level inner_lr is the number that was actually used -- it must exist."""
    missing = [r["path"] for r in results if "inner_lr" not in r["json"]]
    assert not missing, f"{len(missing)} result files do not record inner_lr"


def test_every_result_records_whether_it_was_tuned(results) -> None:
    """Without this flag an untuned run is indistinguishable from a tuned one."""
    missing = [r["path"] for r in results if "tuned" not in r["json"]]
    assert not missing, f"{len(missing)} result files do not record `tuned`"


def test_manifest_config_inner_lr_is_not_the_effective_rate(results) -> None:
    """Pin the trap itself, so a future reader meets it as a test and not a scare.

    If this ever fails -- if nested and effective agree everywhere -- then either
    nothing in the corpus was tuned away from the default, or the manifest was
    changed to record the effective rate. Both are fine outcomes, but both mean
    this file's docstring is stale and should be re-read before being trusted.
    """
    disagreements = 0
    for r in results:
        j = r["json"]
        nested = j.get("manifest", {}).get("config", {}).get("inner_lr")
        if nested is not None and "inner_lr" in j and nested != j["inner_lr"]:
            disagreements += 1
    assert disagreements > 0, (
        "manifest.config.inner_lr now agrees with the effective inner_lr in every "
        "committed result. Re-read this module's docstring: the hazard it describes "
        "may be gone, in which case say so, rather than leaving a stale warning."
    )


def test_no_committed_result_was_measured_on_the_test_split(results) -> None:
    """The test split is one run at the very end, after every decision is frozen.

    A result file claiming split='test' means that discipline was broken, which
    would be invisible otherwise -- run_baselines.py defaults to val, so the only
    way in is an explicit flag.
    """
    offenders = [
        r["path"] for r in results
        if r["json"].get("manifest", {}).get("config", {}).get("split") == "test"
    ]
    assert not offenders, f"results measured on the test split: {offenders}"


# The split fractions every driver passes. NOT make_splits' default of
# (0.6, 0.2, 0.2), which no run in this project has ever used -- a distinction that
# put wrong numbers in the paper's Methods section until 2026-10-21, because the
# count was taken with the default while every result came from these.
DRIVER_FRACTIONS = {"eurosat": "0.2 0.1 0.7", "isic": "0.1 0.4 0.5"}


def test_every_driver_uses_the_documented_split_fractions() -> None:
    """Pin the fractions, because the paper quotes split sizes derived from them.

    A driver that passes something else -- or omits --fractions and silently takes
    make_splits' default -- produces results whose split sizes do not match what the
    paper says, and nothing else in the pipeline would notice: the manifest records
    the fractions, but no reader compares them against the prose.

    The fractions differ per dataset on purpose (ISIC's smallest classes cannot
    spare a large training share and still leave 5-way episodes), so this checks
    each dataset against its own value rather than demanding they agree.
    """
    scripts = sorted((ROOT / "scripts").glob("run_*.sh"))
    if not scripts:
        pytest.skip("no driver scripts")
    offenders: list[str] = []
    seen = 0
    for sc in scripts:
        text = sc.read_text()
        for ds, frac in DRIVER_FRACTIONS.items():
            for m in re.finditer(rf"--dataset\s+{ds}\s+--fractions\s+([\d. ]+?)(?=[\"\'\n]|--)", text):
                seen += 1
                got = " ".join(m.group(1).split())
                if got != frac:
                    offenders.append(f"{sc.name}: {ds} uses {got!r}, expected {frac!r}")
    assert seen, "no --dataset/--fractions pairs found -- the regex or the drivers changed"
    assert not offenders, "drivers disagree on split fractions: " + "; ".join(offenders)


def test_no_driver_relies_on_the_default_fractions() -> None:
    """Every --dataset that reaches a split must carry --fractions beside it.

    make_splits' default is (0.6, 0.2, 0.2). Nothing in this project uses it and it
    matches no number in the paper, so a driver reaching it silently produces splits
    incomparable with every other result -- and the manifest would record the
    default without anyone noticing.

    Three attempts' worth of scoping, recorded because each wrong version was a
    plausible-looking test that checked the wrong thing:

    * A physical-line scan cries wolf, because drivers write one argument per line
      with trailing backslashes.
    * Scoping to literal ``run_baselines.py ... --dataset`` finds *nothing*: the
      drivers hold the arguments in a shell variable
      (``EURO_DATA="--dataset eurosat --fractions 0.2 0.1 0.7"``) and expand it at
      the call site, so the literal flag only ever appears in the assignment.
    * merge_sweeps.py also takes --dataset, to label its output, and never splits --
      so it must be exempt rather than flagged.

    What is left is the check that actually holds: wherever ``--dataset`` appears in
    a driver, either ``--fractions`` appears in the same logical line, or that line
    is a merge_sweeps call.
    """
    scripts = sorted((ROOT / "scripts").glob("run_*.sh"))
    if not scripts:
        pytest.skip("no driver scripts")
    offenders: list[str] = []
    checked = 0
    for sc in scripts:
        text = re.sub(r"\\\s*\n\s*", " ", sc.read_text())
        for line in text.splitlines():
            if "--dataset" not in line:
                continue
            if "merge_sweeps" in line:
                continue
            checked += 1
            if "--fractions" not in line:
                offenders.append(f"{sc.name}: {line.strip()[:110]}")
    assert checked, (
        "no --dataset found outside merge_sweeps calls -- the drivers or this "
        "test's assumptions changed"
    )
    assert not offenders, (
        "--dataset without --fractions (silently defaulting to 0.6/0.2/0.2): "
        + "; ".join(offenders)
    )
