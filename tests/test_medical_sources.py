"""Parsing tests for the medical benchmark loaders.

Built against synthetic manifests that reproduce the real file formats, so the CSV
handling, class mapping and -- most importantly -- the provenance grouping are verified
without a 2.6 GB download. The grouping is the part worth testing: both corpora contain
many images per underlying subject, and getting it wrong inflates accuracy invisibly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arc.data.base import make_splits
from arc.data.medical import CHESTX_CLASSES, ISIC_CLASSES, chestx, isic


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (120, 80, 40)).save(path)


@pytest.fixture
def isic_root(tmp_path: Path) -> Path:
    """Three lesions, two images each -- the leakage structure HAM10000 really has."""
    img_dir = tmp_path / "ISIC2018_Task3_Training_Input"
    rows, meta = [], []
    for lesion in range(3):
        for view in range(2):
            image_id = f"ISIC_{lesion:04d}{view}"
            _write_image(img_dir / f"{image_id}.jpg")
            onehot = ["0.0"] * len(ISIC_CLASSES)
            onehot[lesion] = "1.0"
            rows.append(f"{image_id}," + ",".join(onehot))
            meta.append(f"HAM_{lesion:04d},{image_id}")

    (tmp_path / "ISIC2018_Task3_Training_GroundTruth.csv").write_text(
        "image," + ",".join(ISIC_CLASSES) + "\n" + "\n".join(rows) + "\n"
    )
    (tmp_path / "HAM10000_metadata.csv").write_text(
        "lesion_id,image_id\n" + "\n".join(meta) + "\n"
    )
    return tmp_path


def test_isic_reads_one_hot_labels(isic_root: Path) -> None:
    src = isic(isic_root, size=32, metadata=isic_root / "HAM10000_metadata.csv")
    assert len(src) == 6
    assert src.class_names == ISIC_CLASSES
    assert sorted(np.unique(src.labels)) == [0, 1, 2]
    assert src.load(0).shape == (3, 32, 32)


def test_isic_groups_by_lesion_when_metadata_is_supplied(isic_root: Path) -> None:
    src = isic(isic_root, size=32, metadata=isic_root / "HAM10000_metadata.csv")
    assert len(np.unique(src.groups)) == 3, "six images should collapse to three lesions"


def test_isic_without_metadata_falls_back_to_per_image_groups(isic_root: Path, caplog) -> None:
    """The fallback must be loud: silently treating each image as its own lesion is
    exactly how leakage gets into a published number."""
    with caplog.at_level("WARNING"):
        src = isic(isic_root, size=32)
    assert len(np.unique(src.groups)) == 6
    assert any("lesion" in r.message.lower() for r in caplog.records)


def test_isic_lesion_grouping_actually_prevents_split_leakage(isic_root: Path) -> None:
    """The property that matters: no lesion may appear in two splits."""
    src = isic(isic_root, size=32, metadata=isic_root / "HAM10000_metadata.csv")
    splits = make_splits(src, mode="group_disjoint", fractions=(0.34, 0.33, 0.33), seed=0)
    seen: dict[str, str] = {}
    for name, split in splits.items():
        for g in src.groups[split.indices]:
            assert seen.setdefault(g, name) == name, f"lesion {g} spans splits"


def test_isic_skips_rows_whose_image_is_missing(isic_root: Path) -> None:
    (isic_root / "ISIC2018_Task3_Training_Input" / "ISIC_00000.jpg").unlink()
    src = isic(isic_root, size=32, metadata=isic_root / "HAM10000_metadata.csv")
    assert len(src) == 5


@pytest.fixture
def chestx_root(tmp_path: Path) -> Path:
    """Covers the three filters: multi-label, out-of-vocabulary, and nested image dirs."""
    rows = [
        # Image Index, Finding Labels, Patient ID  -- kept
        ("00000001_000.png", "Cardiomegaly", "1"),
        ("00000001_001.png", "Cardiomegaly", "1"),   # same patient, second study
        ("00000002_000.png", "Effusion", "2"),
        ("00000003_000.png", "Nodule", "3"),
        # dropped
        ("00000004_000.png", "Cardiomegaly|Effusion", "4"),  # multi-label
        ("00000005_000.png", "No Finding", "5"),             # not a kept class
        ("00000006_000.png", "Hernia", "6"),                 # outside the 7 classes
    ]
    # Images live in per-tarball subdirectories, as the NIH release ships them.
    for i, (name, _, _) in enumerate(rows):
        _write_image(tmp_path / f"images_{i % 2 + 1:03d}" / "images" / name)

    header = "Image Index,Finding Labels,Patient ID\n"
    (tmp_path / "Data_Entry_2017.csv").write_text(
        header + "\n".join(f"{a},{b},{c}" for a, b, c in rows) + "\n"
    )
    return tmp_path


def test_chestx_keeps_only_single_label_images_in_the_seven_classes(chestx_root: Path) -> None:
    src = chestx(chestx_root, size=32)
    assert len(src) == 4, "multi-label, No Finding and Hernia rows must be dropped"
    assert src.class_names == CHESTX_CLASSES
    assert set(np.unique(src.labels)) == {
        CHESTX_CLASSES.index("Cardiomegaly"),
        CHESTX_CLASSES.index("Effusion"),
        CHESTX_CLASSES.index("Nodule"),
    }


def test_chestx_finds_images_in_nested_tarball_directories(chestx_root: Path) -> None:
    src = chestx(chestx_root, size=32)
    assert all(p.exists() for p in src.paths)
    assert src.load(0).shape == (3, 32, 32)


def test_chestx_groups_by_patient(chestx_root: Path) -> None:
    src = chestx(chestx_root, size=32)
    assert len(np.unique(src.groups)) == 3, "two studies of patient 1 must share a group"


def test_chestx_reports_a_clear_error_when_the_manifest_is_absent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Data_Entry_2017"):
        chestx(tmp_path)


def test_csv_reader_tolerates_a_byte_order_mark(tmp_path: Path) -> None:
    """Several of these manifests ship with a BOM, which otherwise corrupts the first
    column name and surfaces as a baffling KeyError on 'image'."""
    from arc.data.pathlist import read_csv

    path = tmp_path / "bom.csv"
    path.write_bytes("﻿image,MEL\nISIC_0,1.0\n".encode("utf-8"))
    cols, rows = read_csv(path)
    assert "image" in cols and rows == [["ISIC_0", "1.0"]]
