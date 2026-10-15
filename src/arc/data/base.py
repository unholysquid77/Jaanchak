"""Dataset-agnostic few-shot plumbing.

Everything downstream of this module sees only :class:`FewShotSource` and
:class:`Episode`. Wiring a new corpus (the forensic fiber/glass/paint sets, LUWA,
anything else) means implementing one small class, not touching the training loop.

Two split modes exist because forensic trace evidence does not fit the standard
few-shot protocol:

``class_disjoint``
    The classic meta-learning split -- meta-train/val/test hold disjoint *classes*.
    Correct for large-vocabulary corpora (EuroSAT, CropDisease, miniImageNet).

``group_disjoint``
    Every split sees every class, but the *source specimens* are disjoint. Required
    when a dataset has 8-12 classes total: you cannot hold out 5 classes from 8 and
    still run 5-way episodes. This is also the only split that defends against the
    dominant leakage mode in trace evidence, where a dataset is built by taking many
    crops of a small number of physical objects. Two crops of one fiber landing on
    opposite sides of the split inflates accuracy enormously and silently.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator, Literal, Protocol, Sequence

import numpy as np
import torch

SplitMode = Literal["class_disjoint", "group_disjoint"]
SplitName = Literal["train", "val", "test"]


class FewShotSource(Protocol):
    """Minimal contract a corpus must satisfy.

    ``groups`` carries the physical provenance of each image -- specimen id, slide id,
    source object. When a corpus genuinely has one image per object, groups may equal
    ``arange(len)``; when it does not, getting this right is the difference between a
    real result and a leaked one.
    """

    labels: np.ndarray  # int64, shape (n,)
    groups: np.ndarray  # shape (n,), hashable provenance key per item
    class_names: Sequence[str]

    def __len__(self) -> int: ...

    def load(self, index: int) -> torch.Tensor:
        """Return a CHW float tensor, already transformed."""


@dataclass(frozen=True)
class Episode:
    """One N-way K-shot task.

    Labels are *remapped* to ``0..n_way-1``; ``global_classes`` retains the original
    class ids so per-class analysis survives episode sampling.
    """

    support_x: torch.Tensor  # (n_way * k_shot, C, H, W)
    support_y: torch.Tensor  # (n_way * k_shot,) in [0, n_way)
    query_x: torch.Tensor  # (n_way * n_query, C, H, W)
    query_y: torch.Tensor  # (n_way * n_query,) in [0, n_way)
    global_classes: torch.Tensor  # (n_way,) original class ids
    episode_id: int

    @property
    def n_way(self) -> int:
        return int(self.global_classes.numel())

    def to(self, device: torch.device | str) -> "Episode":
        return Episode(
            support_x=self.support_x.to(device, non_blocking=True),
            support_y=self.support_y.to(device, non_blocking=True),
            query_x=self.query_x.to(device, non_blocking=True),
            query_y=self.query_y.to(device, non_blocking=True),
            global_classes=self.global_classes,
            episode_id=self.episode_id,
        )


def _stable_hash(value: object) -> int:
    """Deterministic across processes and runs, unlike ``hash()`` on str."""
    return int.from_bytes(
        hashlib.blake2b(repr(value).encode(), digest_size=8).digest(), "big"
    )


@dataclass(frozen=True)
class Split:
    """Indices into a source, plus the classes those indices are allowed to use."""

    indices: np.ndarray
    classes: np.ndarray
    name: SplitName
    mode: SplitMode


def make_splits(
    source: FewShotSource,
    mode: SplitMode,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> dict[SplitName, Split]:
    """Partition a source into train/val/test under the chosen leakage discipline."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions} -> {sum(fractions)}")

    rng = np.random.default_rng(seed)
    labels = np.asarray(source.labels)
    groups = np.asarray(source.groups)
    names: tuple[SplitName, SplitName, SplitName] = ("train", "val", "test")

    if mode == "class_disjoint":
        classes = np.unique(labels)
        rng.shuffle(classes)
        cuts = np.cumsum([round(f * len(classes)) for f in fractions[:2]])
        parts = np.split(classes, cuts)
        if min(len(p) for p in parts) == 0:
            raise ValueError(
                f"class_disjoint split leaves an empty partition: {len(classes)} classes "
                f"into {fractions}. Use mode='group_disjoint' for small-vocabulary corpora."
            )
        return {
            name: Split(
                indices=np.flatnonzero(np.isin(labels, part)),
                classes=np.sort(part),
                name=name,
                mode=mode,
            )
            for name, part in zip(names, parts)
        }

    if mode == "group_disjoint":
        # Hash groups to [0,1) so the assignment is stable under reordering, and
        # deterministic given the seed -- re-running must reproduce the same split.
        uniq = np.unique(groups)
        keys = np.array([_stable_hash((seed, g)) for g in uniq], dtype=np.uint64)
        order = np.argsort(keys)
        uniq = uniq[order]
        cuts = np.cumsum([round(f * len(uniq)) for f in fractions[:2]])
        parts = np.split(uniq, cuts)

        splits: dict[SplitName, Split] = {}
        for name, part in zip(names, parts):
            idx = np.flatnonzero(np.isin(groups, part))
            present = np.unique(labels[idx]) if idx.size else np.array([], dtype=labels.dtype)
            splits[name] = Split(indices=idx, classes=present, name=name, mode=mode)
        return splits

    raise ValueError(f"unknown split mode {mode!r}")


class EpisodicSampler:
    """Yields :class:`Episode` objects from a split.

    Sampling is seeded per-episode rather than per-sampler, so episode *i* is identical
    no matter where iteration started. That makes a 600-episode evaluation reproducible
    and lets two controller variants be compared on literally the same tasks -- which is
    what "matched experimental conditions" in the methodology requires.
    """

    def __init__(
        self,
        source: FewShotSource,
        split: Split,
        n_way: int,
        k_shot: int,
        n_query: int,
        n_episodes: int,
        seed: int = 0,
    ) -> None:
        if len(split.classes) < n_way:
            raise ValueError(
                f"split {split.name!r} has {len(split.classes)} classes, need n_way={n_way}. "
                f"Either lower n_way or switch to mode='group_disjoint'."
            )

        self.source = source
        self.split = split
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.seed = seed

        labels = np.asarray(source.labels)
        # Pre-bucket indices by class once; episode sampling is then pure index work.
        self._by_class: dict[int, np.ndarray] = {
            int(c): split.indices[labels[split.indices] == c] for c in split.classes
        }

        need = k_shot + n_query
        thin = {c: len(i) for c, i in self._by_class.items() if len(i) < need}
        if thin:
            raise ValueError(
                f"split {split.name!r}: classes {thin} have fewer than k_shot+n_query={need} "
                f"images. Reduce k_shot/n_query, or drop these classes."
            )

    def __len__(self) -> int:
        return self.n_episodes

    def __iter__(self) -> Iterator[Episode]:
        for i in range(self.n_episodes):
            yield self.episode(i)

    def episode(self, index: int) -> Episode:
        rng = np.random.default_rng([self.seed, index])
        classes = rng.choice(
            np.asarray(list(self._by_class)), size=self.n_way, replace=False
        )

        support_idx, query_idx = [], []
        for c in classes:
            pool = self._by_class[int(c)]
            picked = rng.choice(pool, size=self.k_shot + self.n_query, replace=False)
            support_idx.append(picked[: self.k_shot])
            query_idx.append(picked[self.k_shot :])

        def stack(chunks: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
            flat = np.concatenate(chunks)
            x = torch.stack([self.source.load(int(j)) for j in flat])
            y = torch.arange(self.n_way).repeat_interleave(len(chunks[0]))
            return x, y

        sx, sy = stack(support_idx)
        qx, qy = stack(query_idx)
        return Episode(
            support_x=sx,
            support_y=sy,
            query_x=qx,
            query_y=qy,
            global_classes=torch.as_tensor(classes.astype(np.int64)),
            episode_id=index,
        )
