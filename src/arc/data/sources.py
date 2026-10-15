"""Concrete corpora.

:class:`ImageFolderSource` is the workhorse: any corpus laid out as
``root/<class_name>/<image>`` is usable immediately, which covers the forensic
fiber/glass/paint sets if and when they arrive, LUWA, CropDisease, ISIC, and most
microscopy data in the wild.

:func:`eurosat` is the one corpus wired end-to-end with auto-download, so the pipeline
can be exercised on a real cross-domain target from a cold checkout with no manual steps.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_transform(size: int = 224, train: bool = False) -> Callable:
    """ImageNet-normalised transforms.

    Normalisation must match the pretrained backbone's statistics -- the frozen
    reference branch is only a meaningful baseline if it sees inputs distributed the
    way its pretraining did.
    """
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize(int(size * 1.14)),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ImageFolderSource:
    """A corpus of ``root/<class_name>/<image>``.

    Parameters
    ----------
    group_pattern
        Regex with one capture group, applied to each image's *filename*, yielding the
        provenance key for leakage-safe splitting. For a corpus where ``fiber07_c3.png``
        means "crop 3 of physical fiber 7", pass ``r"(.+?)_c\\d+"``.

        Leaving this ``None`` asserts that every image is an independent specimen. That
        is frequently false for trace evidence built from crops of a few objects, so the
        default emits a warning rather than passing silently -- a wrong assumption here
        inflates every number downstream.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        group_pattern: str | None = None,
        extensions: set[str] = IMAGE_EXTS,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.root} is not a directory")

        self.transform = transform or build_transform()
        class_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        if not class_dirs:
            raise ValueError(f"{self.root} contains no class subdirectories")

        self.class_names: list[str] = [p.name for p in class_dirs]
        paths: list[Path] = []
        labels: list[int] = []
        for label, d in enumerate(class_dirs):
            found = sorted(p for p in d.rglob("*") if p.suffix.lower() in extensions)
            if not found:
                log.warning("class %r has no images", d.name)
            paths.extend(found)
            labels.extend([label] * len(found))

        if not paths:
            raise ValueError(f"no images found under {self.root}")

        self.paths = paths
        self.labels = np.asarray(labels, dtype=np.int64)
        self.groups = self._infer_groups(paths, group_pattern)

        log.info(
            "%s: %d images, %d classes, %d specimen groups",
            self.root.name,
            len(paths),
            len(self.class_names),
            len(np.unique(self.groups)),
        )

    def _infer_groups(self, paths: Sequence[Path], pattern: str | None) -> np.ndarray:
        if pattern is None:
            log.warning(
                "%s: no group_pattern given -- treating every image as an independent "
                "specimen. If this corpus contains multiple crops/views of the same "
                "physical object, results will be leakage-inflated.",
                self.root.name,
            )
            return np.arange(len(paths))

        rx = re.compile(pattern)
        groups, unmatched = [], 0
        for p in paths:
            m = rx.match(p.stem)
            if m:
                groups.append(f"{p.parent.name}/{m.group(1)}")
            else:
                unmatched += 1
                groups.append(f"{p.parent.name}/{p.stem}")
        if unmatched:
            log.warning(
                "%s: group_pattern %r did not match %d/%d filenames; those were treated "
                "as singleton specimens",
                self.root.name,
                pattern,
                unmatched,
                len(paths),
            )
        return np.asarray(groups, dtype=object)

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as img:
            return self.transform(img.convert("RGB"))


class TensorSource:
    """In-memory corpus. Used by tests and by small cached datasets."""

    def __init__(
        self,
        images: torch.Tensor,
        labels: np.ndarray,
        groups: np.ndarray | None = None,
        class_names: Sequence[str] | None = None,
    ) -> None:
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64)
        self.groups = np.arange(len(labels)) if groups is None else np.asarray(groups)
        n_classes = int(self.labels.max()) + 1 if len(self.labels) else 0
        self.class_names = list(class_names or [str(i) for i in range(n_classes)])

    def __len__(self) -> int:
        return len(self.labels)

    def load(self, index: int) -> torch.Tensor:
        return self.images[index]


def eurosat(root: str | Path, size: int = 224, download: bool = True) -> ImageFolderSource:
    """EuroSAT -- the satellite member of the BSCD-FSL cross-domain benchmark.

    Auto-downloads (~90 MB, 27k images, 10 classes) and then re-exposes the extracted
    tree through :class:`ImageFolderSource`, so it travels the exact same code path as
    every other corpus. Each tile is an independent sample, so per-image groups are
    correct here.
    """
    from torchvision.datasets import EuroSAT

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    EuroSAT(root=str(root), download=download)  # extraction only; we use the tree

    candidates = [p for p in root.rglob("2750") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"EuroSAT extracted tree not found under {root}")
    return ImageFolderSource(candidates[0], transform=build_transform(size))
