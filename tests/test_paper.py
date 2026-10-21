"""Structural checks on the LaTeX draft.

These exist because this project's recurring failure mode is *a key that omits an
axis*: two different things land in one slot and produce a plausible wrong number.
A dangling ``\\cite`` is the same bug in the bibliography -- it compiles to a silent
``[?]`` that a reader takes for a real reference, and the first draft shipped with
one (``tschandl2018`` cited, never defined) plus three fabricated claims about which
checkpoint the comparators use.

There is no texlive in CI, so these are the cheap checks that do not need one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TEX = Path(__file__).resolve().parents[1] / "paper" / "arc.tex"


@pytest.fixture(scope="module")
def tex() -> str:
    if not TEX.exists():
        pytest.skip(f"{TEX} not present")
    return TEX.read_text()


def _cited(tex: str) -> set[str]:
    keys: set[str] = set()
    for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
        keys.update(k.strip() for k in m.group(1).split(",") if k.strip())
    return keys


def _defined(tex: str) -> set[str]:
    return set(re.findall(r"\\bibitem\{([^}]*)\}", tex))


def test_every_citation_resolves(tex: str) -> None:
    """A cited key with no bibitem renders as [?] -- invisible unless you look."""
    assert not _cited(tex) - _defined(tex)


def test_no_unused_bibliography_entries(tex: str) -> None:
    """An uncited entry is usually a citation that got dropped during an edit."""
    assert not _defined(tex) - _cited(tex)


def test_every_ref_has_a_label(tex: str) -> None:
    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    refs = set(re.findall(r"\\(?:ref|autoref|pageref)\{([^}]*)\}", tex))
    assert not refs - labels


def test_braces_balance(tex: str) -> None:
    assert tex.count("{") == tex.count("}")


def test_figure_data_files_exist(tex: str) -> None:
    """Every table the tex reads must be one make_figures.py actually emits."""
    figdata = TEX.parent / "figdata"
    wanted = set(re.findall(r"figdata/([A-Za-z0-9_]+\.dat)", tex))
    assert wanted, "no figdata tables referenced -- the regex or the tex changed"
    missing = sorted(n for n in wanted if not (figdata / n).exists())
    assert not missing, f"tex reads tables that do not exist: {missing}"


def test_every_quoted_margin_comes_from_a_generated_table(tex: str) -> None:
    """The project rule is: never transcribe a number into the tex. This enforces it.

    Every ``\\dd{value}{ci}`` in the paper is a claimed margin. Each one must appear,
    to the digit, in one of the tables make_figures.py emits from results/. A number
    typed by hand -- or left behind when its underlying run was re-measured -- fails
    here rather than surviving into a submission.

    It also catches the specific mistake this project keeps making: quoting a point
    estimate from one draw count beside a confidence interval from another. The k=5
    gap was briefly written as 2.72 +/- 0.64, pairing a seed-0 mean with a
    three-draw interval; the pair (2.72, 0.64) exists in no table, so this fails on
    it while either honest pair passes.
    """
    figdata = TEX.parent / "figdata"
    cols = {"key_numbers.dat": (2, 3), "shots.dat": (4, 5),
            "mech_gains.dat": (2, 3), "weight_decay.dat": (3, 4)}
    allowed: set[tuple[float, float]] = set()
    for name, (vi, ci) in cols.items():
        f = figdata / name
        if not f.exists():
            continue
        for line in f.read_text().splitlines()[1:]:
            tok = line.split()
            allowed.add((abs(float(tok[vi])), float(tok[ci])))
    if not allowed:
        pytest.skip("no generated tables to check against")
    quoted = [(float(a), float(b))
              for a, b in re.findall(r"\\dd\{(-?[\d.]+)\}\{([\d.]+)\}", tex)]
    assert quoted, "no \\dd margins found -- the macro or the regex changed"
    ungenerated = [q for q in quoted if (abs(q[0]), q[1]) not in allowed]
    assert not ungenerated, (
        f"margins in the paper that no generated table contains: {ungenerated}. "
        f"Either the number was typed by hand, or its run was re-measured and the "
        f"tex was not regenerated, or a point estimate and an interval come from "
        f"different draw counts."
    )


def test_comparators_are_not_claimed_to_share_our_checkpoint(tex: str) -> None:
    """Regression guard on a specific false claim.

    The draft asserted ViT-B/16 augreg_in21k was "the checkpoint SSF, LN-Tune and DIPA
    report on". None of the three reports on BSCD-FSL with our protocol, and DIPA uses
    ViT-S/16 with DINO -- a different size and a different pretraining objective. The
    claim is easy to reintroduce while tightening prose, so pin it.
    """
    flat = " ".join(tex.split())
    assert "checkpoint SSF, LN-Tune and DIPA report on" not in flat
