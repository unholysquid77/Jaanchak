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


def test_comparators_are_not_claimed_to_share_our_checkpoint(tex: str) -> None:
    """Regression guard on a specific false claim.

    The draft asserted ViT-B/16 augreg_in21k was "the checkpoint SSF, LN-Tune and DIPA
    report on". None of the three reports on BSCD-FSL with our protocol, and DIPA uses
    ViT-S/16 with DINO -- a different size and a different pretraining objective. The
    claim is easy to reintroduce while tightening prose, so pin it.
    """
    flat = " ".join(tex.split())
    assert "checkpoint SSF, LN-Tune and DIPA report on" not in flat
