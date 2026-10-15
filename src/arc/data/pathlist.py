"""A corpus defined by explicit ``(path, label, group)`` lists.

Every dataset in the project reduces to this. A loader's only job is to turn whatever
manifest a release ships with -- a one-hot CSV, a pipe-delimited label column, a
directory tree -- into three parallel lists, and then hand them here. That keeps the
format-specific parsing in one small function per corpus and the episodic machinery
entirely format-agnostic.

The ``group`` is the identity of the *physical object* an image depicts: a lesion, a
patient, a fiber. It is the only defence against the dominant failure mode in
small-sample imaging studies -- multiple views of one object landing on both sides of a
split, so the model recognises the object rather than its class. See
``docs/DATA_REQUEST.md``; on ISIC this affects 45% of the corpus.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("arc.data")


def read_csv(path: str | Path) -> tuple[list[str], list[list[str]]]:
    """Read a CSV into ``(column_names, rows)``.

    Opened with ``utf-8-sig`` because several of these manifests ship with a byte-order
    mark, which otherwise becomes part of the first column's name and surfaces much
    later as a baffling ``KeyError: 'image'``. Pinned by a test.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        rows = [row for row in reader if row]
    return [h.strip() for h in header], rows


def load_image(path: str | Path) -> Image.Image:
    """Open an image as RGB.

    The explicit conversion matters: ChestX ships 8-bit greyscale PNGs (and a handful of
    4-channel ones), and a pretrained ResNet needs three channels. Without this the
    corpus loads fine and fails at the first forward pass.
    """
    with Image.open(path) as img:
        return img.convert("RGB")


class PathListSource:
    """An in-memory corpus: parallel path / label / group arrays, loaded on demand.

    Implements :class:`~arc.data.base.FewShotSource`. Images are read from disk per
    access rather than cached: an episode touches at most a few hundred images, the
    full corpora run to 112k, and the box has 61 GiB of RAM to spend on better things.

    Parameters
    ----------
    paths
        One filesystem path per image.
    labels
        Integer class index per image, indexing into ``class_names``.
    groups
        Source-object identity per image. ``None`` means "every image is its own
        object", which is the correct reading only when that is actually true.
    class_names
        Class index -> human-readable name, for tables and figures.
    transform
        PIL -> tensor. Required; there is no sensible default resolution.
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        labels: Sequence[int],
        groups: Sequence[object] | None,
        class_names: Sequence[str],
        transform: Callable[[Image.Image], torch.Tensor],
        name: str = "pathlist",
    ) -> None:
        if not (len(paths) == len(labels) and (groups is None or len(groups) == len(paths))):
            raise ValueError(
                f"paths/labels/groups length mismatch: {len(paths)}, {len(labels)}, "
                f"{'-' if groups is None else len(groups)}"
            )
        self.paths: list[Path] = [Path(p) for p in paths]
        self.labels: np.ndarray = np.asarray(labels, dtype=np.int64)
        # Strings, not integers: a group id is an opaque identity ('HAM_0001'), and
        # coercing it to int would quietly collide unrelated specimens.
        self.groups: np.ndarray = np.asarray(
            [str(g) for g in groups] if groups is not None else [str(p) for p in self.paths],
            dtype=object,
        )
        self.class_names: tuple[str, ...] = tuple(class_names)
        self.transform = transform
        self.name = name

        if len(self.labels) and self.labels.max() >= len(self.class_names):
            raise ValueError(
                f"label {int(self.labels.max())} has no entry in class_names "
                f"({len(self.class_names)} names)"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, index: int) -> torch.Tensor:
        return self.transform(load_image(self.paths[index]))

    # -- provenance reporting ------------------------------------------------

    def class_counts(self) -> dict[str, int]:
        counts = np.bincount(self.labels, minlength=len(self.class_names))
        return {name: int(c) for name, c in zip(self.class_names, counts)}

    def group_summary(self) -> dict[str, int]:
        """How much the corpus is actually inflated by repeated views.

        Logged at load time for every corpus. ``images_sharing_a_group`` is the number
        that belongs in the paper's dataset table instead of the raw image count.
        """
        _, counts = np.unique(self.groups, return_counts=True)
        return {
            "images": len(self.paths),
            "groups": int(counts.size),
            "images_sharing_a_group": int(counts[counts > 1].sum()),
            "largest_group": int(counts.max()) if counts.size else 0,
        }

    def describe(self) -> str:
        g = self.group_summary()
        return (
            f"{self.name}: {g['images']} images, {len(self.class_names)} classes, "
            f"{g['groups']} source objects "
            f"({g['images_sharing_a_group']} images share an object)"
        )
