"""ISIC 2018 and NIH ChestX-ray14 -- the two hard BSCD-FSL targets.

These are where ARC should actually be tested. EuroSAT has a frozen-backbone 5-shot
accuracy of 84.6%, which leaves every method inside three points of every other and no
room for a controller to demonstrate anything. Published 5-shot numbers on ISIC and
ChestX are around 45% and 25% respectively.

Both corpora have the repeated-view structure that ``docs/DATA_REQUEST.md`` warns
about, and in both cases the provenance key lives in a side file rather than in the
image names:

* ISIC: ``HAM10000_metadata.csv`` carries ``lesion_id``. Measured on the real corpus,
  10,015 images come from 7,470 lesions and 4,501 images (45%) share a lesion with at
  least one other. Splitting by image puts nearly half the corpus on both sides.
* ChestX: ``Data_Entry_2017.csv`` carries ``Patient ID``, and the release contains many
  studies per patient.

.. note::
   Reconstructed on 2026-10-15; see the note in ``arc/data/base.py``. The parsing here
   is pinned by ``tests/test_medical_sources.py`` against synthetic manifests in the
   real formats, and the ISIC path was additionally verified against the downloaded
   corpus (per-class counts match the published HAM10000 distribution exactly --
   ``docs/EXPERIMENT_LOG.md`` section 5.3).
"""

from __future__ import annotations

import logging
from pathlib import Path

from arc.data.pathlist import PathListSource, read_csv
from arc.data.sources import IMAGE_SUFFIXES, build_transform

log = logging.getLogger("arc.data")

# Column order of ISIC2018_Task3_Training_GroundTruth.csv, which is also the order the
# challenge reports in. Kept as the canonical class order so a label index means the
# same thing in every table we produce.
ISIC_CLASSES: tuple[str, ...] = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")

# The seven findings BSCD-FSL uses from ChestX-ray14. 'No Finding' is excluded (it is
# the absence of a class, not a class) and so are the seven rarer findings, which is
# the published protocol -- deviating from it would make the numbers incomparable.
CHESTX_CLASSES: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumothorax",
)


def _index_images(root: Path) -> dict[str, Path]:
    """Map filename -> path, searching recursively.

    Both releases ship in more than one layout: ChestX arrives as twelve tarballs that
    extract to ``images_001/images/...``, but a Kaggle download or a manual re-extract
    can be flat. Indexing by basename makes the loader indifferent to which one is on
    disk, which is what ``scripts/download_data.sh`` promises.
    """
    out: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            out.setdefault(path.name, path)
    return out


def isic(
    root: str | Path,
    size: int = 224,
    metadata: str | Path | None = None,
) -> PathListSource:
    """ISIC 2018 Task 3 (= HAM10000): 7 dermoscopy classes, ~10k images.

    Parameters
    ----------
    metadata
        Path to ``HAM10000_metadata.csv``. Supplying it is what makes the split
        leakage-safe; ``scripts/download_data.sh isic`` fetches it from the Harvard
        Dataverse record because the ISIC S3 bucket does not serve it.

        Omitting it is legal but **logs a warning**, because the fallback -- one group
        per image -- is precisely the silent-leakage case. Loud by design and pinned by
        a test.

    Rows whose image is absent from disk are skipped rather than failing the load: a
    partial or resumed download is common on a 2.6 GB fetch, and the count is logged so
    a half-present corpus cannot be mistaken for a complete one.
    """
    root = Path(root)
    gt_path = root / "ISIC2018_Task3_Training_GroundTruth.csv"
    if not gt_path.exists():
        # The release nests the CSV inside a directory of the same name as the zip.
        nested = root / "ISIC2018_Task3_Training_GroundTruth" / gt_path.name
        if nested.exists():
            gt_path = nested
        else:
            raise FileNotFoundError(
                f"no ISIC2018_Task3_Training_GroundTruth.csv under {root}. "
                f"Run: bash scripts/download_data.sh isic"
            )

    image_dir = root / "ISIC2018_Task3_Training_Input"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"no ISIC2018_Task3_Training_Input directory under {root}")
    index = _index_images(image_dir)

    header, rows = read_csv(gt_path)
    lower = [h.lower() for h in header]
    if "image" not in lower:
        raise ValueError(f"{gt_path} has no 'image' column; found {header}")
    id_col = lower.index("image")
    try:
        class_cols = [header.index(c) for c in ISIC_CLASSES]
    except ValueError as exc:
        raise ValueError(
            f"{gt_path} is missing an expected class column: {exc}. Expected all of "
            f"{ISIC_CLASSES}, found {header}"
        ) from None

    lesion_of: dict[str, str] = {}
    if metadata is not None:
        meta_header, meta_rows = read_csv(metadata)
        try:
            li, ii = meta_header.index("lesion_id"), meta_header.index("image_id")
        except ValueError:
            raise ValueError(
                f"{metadata} has no lesion_id/image_id columns (found {meta_header}). "
                f"Dataverse ingests the CSV into a .tab; refetch with "
                f"'?format=original'."
            ) from None
        lesion_of = {row[ii]: row[li] for row in meta_rows if len(row) > max(li, ii)}
    else:
        log.warning(
            "no HAM10000 metadata supplied: every ISIC image will be treated as its "
            "own lesion. The real corpus has 10,015 images from 7,470 lesions, so 45%% "
            "of images share a lesion with another and a split built this way leaks "
            "across train/test. Pass metadata=<HAM10000_metadata.csv>."
        )

    paths, labels, groups = [], [], []
    missing_images = missing_lesions = 0
    for row in rows:
        if len(row) <= max(class_cols):
            continue
        image_id = row[id_col].strip()
        path = index.get(f"{image_id}.jpg") or index.get(f"{image_id}.png")
        if path is None:
            missing_images += 1
            continue
        onehot = [float(row[c]) for c in class_cols]
        paths.append(path)
        labels.append(int(max(range(len(onehot)), key=onehot.__getitem__)))
        if lesion_of:
            lesion = lesion_of.get(image_id)
            if lesion is None:
                missing_lesions += 1
                lesion = image_id
            groups.append(lesion)
        else:
            groups.append(image_id)

    if not paths:
        raise FileNotFoundError(
            f"{gt_path} lists {len(rows)} rows but none of their images are present "
            f"under {image_dir}"
        )
    if missing_images:
        log.warning("%d of %d ISIC rows skipped: image not on disk", missing_images, len(rows))
    if missing_lesions:
        log.warning("%d ISIC images absent from the lesion metadata; grouped alone", missing_lesions)

    source = PathListSource(
        paths, labels, groups, ISIC_CLASSES, build_transform(size), name="isic2018"
    )
    log.info("%s", source.describe())
    return source


def chestx(root: str | Path, size: int = 224) -> PathListSource:
    """NIH ChestX-ray14, restricted to the BSCD-FSL protocol.

    Keeps only images with a *single* finding drawn from :data:`CHESTX_CLASSES`.
    Multi-label rows are dropped because a 5-way single-label episode cannot represent
    them, and keeping one of their labels arbitrarily would put genuinely ambiguous
    images in as clean exemplars.

    Grouped by ``Patient ID``: the release contains many studies per patient, and two
    radiographs of one chest a week apart are not independent samples.
    """
    root = Path(root)
    manifest = root / "Data_Entry_2017.csv"
    if not manifest.exists():
        found = next(root.rglob("Data_Entry_2017*.csv"), None) if root.is_dir() else None
        if found is None:
            raise FileNotFoundError(
                f"no Data_Entry_2017.csv under {root}. ChestX cannot be fetched "
                f"automatically -- see scripts/download_data.sh chestx for the two "
                f"working routes."
            )
        manifest = found

    header, rows = read_csv(manifest)
    for column in ("Image Index", "Finding Labels", "Patient ID"):
        if column not in header:
            raise ValueError(f"{manifest} has no {column!r} column; found {header}")
    ci, cl, cp = (header.index(c) for c in ("Image Index", "Finding Labels", "Patient ID"))

    index = _index_images(root)
    keep = {name: i for i, name in enumerate(CHESTX_CLASSES)}

    paths, labels, groups = [], [], []
    dropped_multilabel = dropped_class = dropped_missing = 0
    for row in rows:
        if len(row) <= max(ci, cl, cp):
            continue
        findings = row[cl].strip()
        if "|" in findings:
            dropped_multilabel += 1
            continue
        if findings not in keep:
            dropped_class += 1
            continue
        path = index.get(row[ci].strip())
        if path is None:
            dropped_missing += 1
            continue
        paths.append(path)
        labels.append(keep[findings])
        groups.append(f"patient_{row[cp].strip()}")

    if not paths:
        raise FileNotFoundError(
            f"{manifest} yielded no usable images: {dropped_missing} had no file under "
            f"{root}, {dropped_multilabel} were multi-label, {dropped_class} were "
            f"outside the seven protocol classes."
        )
    log.info(
        "ChestX: kept %d of %d rows (dropped %d multi-label, %d out-of-protocol, "
        "%d image missing)",
        len(paths), len(rows), dropped_multilabel, dropped_class, dropped_missing,
    )

    source = PathListSource(
        paths, labels, groups, CHESTX_CLASSES, build_transform(size), name="chestx"
    )
    log.info("%s", source.describe())
    return source
