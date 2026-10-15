"""The hard members of the cross-domain few-shot benchmark.

EuroSAT, which the pipeline has been developed against, is the easy end: a frozen
ImageNet backbone already reaches ~84.6% at 5-shot, so every adaptation strategy lands
within about three points of every other and there is little room to distinguish them.

ISIC and ChestX are the opposite. Published 5-shot accuracy sits around the 40s and the
mid-20s respectively -- ChestX is barely above the 20% chance floor for 5-way. Those are
the regimes where pretrained features are genuinely inadequate, adaptation decisions
actually bite, and a controller has room to win or lose visibly. They are the right
place to test ARC's thesis; EuroSAT becomes the "does not break on easy domains" control.

Both label via CSV rather than directory layout, and both have a real provenance
structure that must be respected when splitting -- see the notes on each loader.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from arc.data.pathlist import PathListSource, read_csv
from arc.data.sources import build_transform

log = logging.getLogger(__name__)

ISIC_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# The standard subset: these seven findings, with "No Finding" and every multi-label
# image dropped so each episode class is unambiguous.
CHESTX_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumothorax",
]


def isic(
    root: str | Path,
    size: int = 224,
    metadata: str | Path | None = None,
) -> PathListSource:
    """ISIC 2018 Task 3 -- dermoscopic skin lesions, 7 classes.

    Expects the official layout::

        root/ISIC2018_Task3_Training_Input/ISIC_0024306.jpg
        root/ISIC2018_Task3_Training_GroundTruth.csv    (image,MEL,NV,BCC,AKIEC,BKL,DF,VASC)

    Parameters
    ----------
    metadata
        Path to ``HAM10000_metadata.csv``. **Strongly recommended.** The ISIC 2018
        training set is HAM10000, which contains multiple dermoscopic images of the
        *same physical lesion* -- roughly 10k images drawn from ~7.5k lesions. The
        ground-truth CSV does not record this; only the metadata file carries
        ``lesion_id``. Splitting by image therefore puts two views of one lesion on
        opposite sides of the split and inflates accuracy for reasons that have nothing
        to do with the method. This is the same leakage mode flagged for the forensic
        data in docs/DATA_REQUEST.md, and here it is known to be present rather than
        merely suspected. Without this file every image is treated as its own lesion and
        a warning is emitted.
    """
    root = Path(root)
    image_dir = next(
        (p for p in root.rglob("ISIC2018_Task3_Training_Input") if p.is_dir()), None
    )
    if image_dir is None:
        raise FileNotFoundError(f"ISIC2018_Task3_Training_Input/ not found under {root}")

    gt = next(root.rglob("ISIC2018_Task3_Training_GroundTruth.csv"), None)
    if gt is None:
        raise FileNotFoundError(
            f"ISIC2018_Task3_Training_GroundTruth.csv not found under {root}"
        )

    cols, rows = read_csv(gt)
    absent_cols = [c for c in ISIC_CLASSES if c not in cols]
    if absent_cols:
        raise ValueError(f"{gt} is missing expected class columns {absent_cols}")

    lesion_of: dict[str, str] = {}
    if metadata is not None:
        mcols, mrows = read_csv(Path(metadata))
        for r in mrows:
            lesion_of[r[mcols["image_id"]]] = r[mcols["lesion_id"]]
    else:
        log.warning(
            "ISIC: no HAM10000_metadata.csv supplied -- every image treated as a "
            "distinct lesion. HAM10000 holds several images per lesion, so "
            "group_disjoint splitting cannot prevent leakage without it."
        )

    paths, labels, groups, skipped = [], [], [], 0
    for r in rows:
        image_id = r[cols["image"]]
        path = image_dir / f"{image_id}.jpg"
        if not path.exists():
            skipped += 1
            continue
        onehot = [float(r[cols[c]]) for c in ISIC_CLASSES]
        paths.append(path)
        labels.append(int(np.argmax(onehot)))
        groups.append(lesion_of.get(image_id, image_id))

    if skipped:
        log.warning("ISIC: %d ground-truth rows had no matching image file", skipped)

    return PathListSource(
        paths, labels, groups, ISIC_CLASSES, build_transform(size), name="isic2018"
    )


def chestx(root: str | Path, size: int = 224) -> PathListSource:
    """NIH ChestX-ray14, restricted to the standard 7-class single-label subset.

    Expects::

        root/Data_Entry_2017.csv
        root/images/00000001_000.png

    The NIH release ships images across ``images_001/`` .. ``images_012/``, each with
    its own ``images/`` subdirectory, so the loader indexes by filename rather than
    assuming a flat tree.

    Images are grouped by **patient**. The corpus holds many studies per patient, so
    splitting by image would place two x-rays of the same person on opposite sides.
    """
    root = Path(root)
    entry = next(root.rglob("Data_Entry_2017*.csv"), None)
    if entry is None:
        raise FileNotFoundError(
            f"Data_Entry_2017.csv not found under {root}. ChestX-ray14 has no stable "
            f"public direct link -- see scripts/download_data.sh for the manual and "
            f"Kaggle routes."
        )

    index: dict[str, Path] = {}
    for p in root.rglob("*.png"):
        index.setdefault(p.name, p)
    if not index:
        raise FileNotFoundError(f"no .png images found under {root}")

    cols, rows = read_csv(entry)
    keep = set(CHESTX_CLASSES)

    paths, labels, groups, multi, absent = [], [], [], 0, 0
    for r in rows:
        finding = r[cols["Finding Labels"]]
        if "|" in finding:
            multi += 1
            continue
        if finding not in keep:
            continue
        path = index.get(r[cols["Image Index"]])
        if path is None:
            absent += 1
            continue
        paths.append(path)
        labels.append(CHESTX_CLASSES.index(finding))
        groups.append(f"patient{r[cols['Patient ID']]}")

    log.info(
        "ChestX: kept %d single-label images in %d classes "
        "(dropped %d multi-label, %d with no image file)",
        len(paths),
        len(CHESTX_CLASSES),
        multi,
        absent,
    )
    return PathListSource(
        paths, labels, groups, CHESTX_CLASSES, build_transform(size), name="chestx"
    )
