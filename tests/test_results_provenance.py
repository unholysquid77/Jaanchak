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
