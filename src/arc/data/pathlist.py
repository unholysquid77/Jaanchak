"""A corpus defined by an explicit list of images, labels and provenance groups.

Needed by any dataset whose labels live in a manifest rather than in the directory
layout. That is both medical few-shot benchmarks, and is also the shape the forensic
trace-evidence data will take if it arrives with a CSV (see docs/DATA_REQUEST.md).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image

from arc.data.sources import build_transform

log = logging.getLogger(__name__)


class PathListSource:
    """Satisfies :class:`~arc.data.base.FewShotSource` from explicit lists."""

    def __init__(
        self,
        paths: Sequence[Path],
        labels: Sequence[int],
        groups: Sequence,
        class_names: Sequence[str],
        transform: Callable | None = None,
        name: str = "corpus",
    ) -> None:
        if not (len(paths) == len(labels) == len(groups)):
            raise ValueError(
                f"length mismatch: {len(paths)} paths, {len(labels)} labels, "
                f"{len(groups)} groups"
            )
        if not paths:
            raise ValueError(f"{name}: no images")

        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.groups = np.asarray(groups, dtype=object)
        self.class_names = list(class_names)
        self.transform = transform or build_transform()
        self.name = name

        n_groups = len(np.unique(self.groups))
        log.info(
            "%s: %d images, %d classes, %d groups (%.1f images per group)",
            name,
            len(paths),
            len(self.class_names),
            n_groups,
            len(paths) / max(n_groups, 1),
        )

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as img:
            return self.transform(img.convert("RGB"))


def read_csv(path: Path) -> tuple[dict[str, int], list[list[str]]]:
    """Return a column-name index and the data rows.

    ``utf-8-sig`` because several of these manifests ship with a BOM, which otherwise
    corrupts the first column name and produces a confusing KeyError.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    return {name: i for i, name in enumerate(rows[0])}, rows[1:]
