"""Image-folder corpora and the transform the whole project shares.

``eurosat`` is the cross-domain few-shot anchor that every result so far rests on.
``ImageFolderSource`` is what the forensic datasets will arrive as, if they arrive:
one directory per class, with the source object recoverable from the filename.

.. note::
   Reconstructed on 2026-10-15; see the note in ``arc/data/base.py``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from torchvision import transforms

from arc.data.pathlist import PathListSource

log = logging.getLogger("arc.data")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ImageNet statistics, because every backbone here is ImageNet-pretrained and the
# frozen reference twin must be fed exactly what it was trained on. Normalising with
# target-domain statistics instead would move the pretrained network off its operating
# point and make the "all gates at 0 reproduces the pretrained network" anchor
# meaningless.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(size: int = 224) -> Callable[[Image.Image], torch.Tensor]:
    """Deterministic evaluation transform: resize to ``size x size``, tensor, normalise.

    No random augmentation here. Augmentation belongs to the inner loop
    (``arc.train.augment``), on the GPU, applied to an already-batched episode -- doing
    it per image on the CPU would make the dataloader the bottleneck on a 20-thread box
    feeding one 3080, and would also put randomness inside the "identical episodes for
    every method" guarantee, where it does not belong.

    Resize is to an exact square rather than ``Resize(size)`` + ``CenterCrop(size)``:
    the corpora here are square already (EuroSAT 64x64, ChestX 1024x1024) or are
    conventionally squashed rather than cropped (ISIC 600x450, where a centre crop
    discards lesion border that the diagnosis depends on).
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _scan_classes(root: Path, classes: Sequence[str] | None) -> list[str]:
    if classes is not None:
        return list(classes)
    found = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not found:
        raise FileNotFoundError(
            f"{root} contains no class subdirectories. An image-folder corpus must be "
            f"one directory per class."
        )
    return found


def ImageFolderSource(
    root: str | Path,
    transform: Callable[[Image.Image], torch.Tensor] | None = None,
    group_pattern: str | None = None,
    classes: Sequence[str] | None = None,
    name: str | None = None,
) -> PathListSource:
    """Corpus from ``root/<class>/<image>``.

    Parameters
    ----------
    group_pattern
        Regex applied to each filename *stem*; capture group 1 is the source-object id.
        For ``cotton_f01_c00.png`` with ``'(.+?)_c\\d+'`` the group is ``cotton_f01``,
        so all crops of fiber 1 move together through a split. Without it every image
        is treated as its own object -- correct only when that is actually true, which
        for a microscopy corpus it usually is not. ``docs/DATA_REQUEST.md`` is the ask
        that gets this answered for the forensic sets.

        A stem that does not match is a hard error, not a silent per-image group: a
        pattern that matches nothing would reinstate exactly the leakage it was passed
        to prevent, and would do so invisibly.

    Named like a class because it was one before the loaders were unified; every call
    site constructs it positionally, so it stays spelled this way.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")
    transform = transform or build_transform()
    class_names = _scan_classes(root, classes)
    matcher = re.compile(group_pattern) if group_pattern else None

    paths: list[Path] = []
    labels: list[int] = []
    groups: list[str] = []
    unmatched: list[str] = []

    for label, cls in enumerate(class_names):
        for path in sorted((root / cls).rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
                continue
            paths.append(path)
            labels.append(label)
            if matcher is None:
                groups.append(str(path))
            else:
                m = matcher.match(path.stem)
                if m is None:
                    unmatched.append(path.name)
                    groups.append(str(path))
                else:
                    groups.append(f"{cls}/{m.group(1)}")

    if unmatched:
        raise ValueError(
            f"--group-pattern {group_pattern!r} matched no source id in "
            f"{len(unmatched)} filenames (e.g. {unmatched[:3]}). Fix the pattern: a "
            f"pattern that silently falls back to per-image groups reinstates the "
            f"split leakage it was supplied to prevent."
        )
    if not paths:
        raise FileNotFoundError(f"no images found under {root} in classes {class_names}")

    source = PathListSource(
        paths, labels, groups, class_names, transform, name=name or root.name
    )
    log.info("%s", source.describe())
    return source


def eurosat(root: str | Path = "data/eurosat", size: int = 224, download: bool = True) -> PathListSource:
    """EuroSAT (Helber et al., 2019) -- the BSCD-FSL satellite target.

    10 classes, 27,000 64x64 Sentinel-2 RGB patches. Downloaded by torchvision on first
    use, which extracts to ``<root>/eurosat/2750/<class>/``; an already-extracted tree
    at either that path or ``<root>`` itself is used as-is so a cloud session and the
    compute box can share one layout.

    Groups are per image. EuroSAT patches are distinct geographic tiles, so there is no
    repeated-object structure to protect against -- the one corpus here where
    per-image splitting happens to be correct.
    """
    root = Path(root)
    for candidate in (root / "eurosat" / "2750", root / "2750", root):
        if candidate.is_dir() and any(d.is_dir() for d in candidate.iterdir()):
            return ImageFolderSource(candidate, transform=build_transform(size), name="eurosat")

    if not download:
        raise FileNotFoundError(f"no EuroSAT tree under {root} and download=False")

    from torchvision.datasets import EuroSAT  # imported lazily: it pulls in requests

    log.info("EuroSAT not present under %s; downloading via torchvision (~90 MB)", root)
    root.mkdir(parents=True, exist_ok=True)
    EuroSAT(str(root), download=True)
    return ImageFolderSource(
        root / "eurosat" / "2750", transform=build_transform(size), name="eurosat"
    )
