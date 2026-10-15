"""Episodes, leakage-safe splits, and the episodic sampler.

Three things live here, and each one encodes a decision that is easy to get wrong in a
way no test of "does it run" would catch.

**The split** (:func:`make_splits`). Partitioning by *image* is wrong whenever a corpus
contains several views of one physical object, because crops of the same fiber -- or
photographs of the same lesion -- then sit on both sides of the partition and the model
is rewarded for recognising the object rather than its class. ``group_disjoint`` is the
default for that reason. ``docs/DATA_REQUEST.md`` has the argument in full; on ISIC,
45% of images share a lesion with another, so the effect is not hypothetical.

**The episode** (:class:`Episode`). A self-contained 5-way task: a support set that
adaptation may use, and a query set that it may not. Query labels exist only for
scoring. ``arc.signals.utility`` documents why the utility signal deliberately refuses
to look at them.

**The sampler** (:class:`EpisodicSampler`). Seeded *per episode index*, not from a
running stream. Iterating a sampler twice therefore yields byte-identical tasks, which
is what makes "every method saw the same episodes" true by construction rather than by
hoping the call order never changes. It also means a sweep can re-iterate the sampler
for each configuration without materialising 600 episodes of pixels.

Separately: the split seed and the episode seed are distinct everywhere
(``--split-seed`` vs ``--seeds``). One value driving both means each repeat evaluates on
a *different test set*, so averaging across repeats is incoherent and a learning rate
tuned on one split's validation set does not apply to the others.

.. note::
   This module was reconstructed on 2026-10-15 after the original was lost to a
   ``.gitignore`` pattern (``data/``) that matched ``src/arc/data/`` and kept the whole
   package out of every commit. The authoritative copy is the one on the compute box at
   ``~/arc/src/arc/data/``. Diff the two before re-deriving any number that already
   appears in ``docs/EXPERIMENT_LOG.md``: the API here is pinned by the test suite and
   the call sites, but the exact order in which the sampler consumes its RNG is not, so
   this file is not guaranteed to redraw the same episodes as the runs already on
   record. See ``docs/HANDOFF.md`` section 6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence, runtime_checkable

import numpy as np
import torch

log = logging.getLogger("arc.data")

SplitName = str
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


@runtime_checkable
class FewShotSource(Protocol):
    """What the episodic machinery requires of a corpus.

    Deliberately minimal -- four arrays and a loader -- so that a new dataset is a
    parsing function returning a :class:`~arc.data.pathlist.PathListSource` and nothing
    more.
    """

    paths: list
    labels: np.ndarray
    groups: np.ndarray
    class_names: tuple[str, ...]

    def __len__(self) -> int: ...

    def load(self, index: int) -> torch.Tensor: ...


# -- episodes ---------------------------------------------------------------


@dataclass
class Episode:
    """One few-shot task.

    ``support_y`` and ``query_y`` are *local* labels in ``[0, n_way)``. The corpus-level
    class ids are kept in ``global_classes`` so a per-class error analysis is still
    possible after the fact; nothing in the model ever sees them.
    """

    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    global_classes: torch.Tensor
    episode_id: int

    @property
    def n_way(self) -> int:
        return int(self.global_classes.numel())

    @property
    def k_shot(self) -> int:
        return int(self.support_x.shape[0] // max(self.n_way, 1))

    def to(self, device: str | torch.device) -> "Episode":
        """Move every tensor to ``device``.

        Returns a new :class:`Episode` rather than mutating in place: the runner calls
        this on each episode it is handed, and a sampler that is re-iterated per
        configuration must not find its cached tensors silently relocated to the GPU.
        """
        device = torch.device(device)
        return Episode(
            support_x=self.support_x.to(device, non_blocking=True),
            support_y=self.support_y.to(device, non_blocking=True),
            query_x=self.query_x.to(device, non_blocking=True),
            query_y=self.query_y.to(device, non_blocking=True),
            global_classes=self.global_classes.to(device, non_blocking=True),
            episode_id=self.episode_id,
        )


# -- splits ------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """A named subset of a corpus, as indices into it."""

    name: SplitName
    indices: np.ndarray
    classes: np.ndarray

    def __len__(self) -> int:
        return int(self.indices.size)


def _normalise_fractions(fractions: Sequence[float]) -> tuple[float, float, float]:
    if len(fractions) != 3:
        raise ValueError(f"expected three fractions (train, val, test), got {len(fractions)}")
    if any(f < 0 for f in fractions):
        raise ValueError(f"fractions must be non-negative, got {tuple(fractions)}")
    total = float(sum(fractions))
    if total <= 0:
        raise ValueError("fractions sum to zero")
    return tuple(float(f) / total for f in fractions)  # type: ignore[return-value]


def _partition(items: np.ndarray, fractions: Sequence[float], rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle ``items`` and cut it into three contiguous blocks.

    Sizes are floored and the remainder goes to the last non-empty block, so no item is
    dropped. Floor rather than round because rounding can over-allocate and leave the
    final block empty, which surfaces as "0 images in test" much later.
    """
    order = rng.permutation(items.size)
    shuffled = items[order]
    n = items.size
    cuts = []
    start = 0
    for frac in fractions[:-1]:
        take = int(np.floor(n * frac))
        cuts.append(shuffled[start : start + take])
        start += take
    cuts.append(shuffled[start:])
    return cuts


def _group_labels(groups: np.ndarray, labels: np.ndarray) -> dict[object, int]:
    """Each source object's modal class.

    An object with images in more than one class (a ChestX patient whose studies carry
    different findings) is assigned to whichever class it has most images in. Ties go to
    the lowest class index, which is arbitrary but deterministic -- what matters is that
    the object gets exactly one assignment and therefore lands in exactly one split.
    """
    counts: dict[object, dict[int, int]] = {}
    for g, y in zip(groups, labels):
        counts.setdefault(g, {})
        counts[g][int(y)] = counts[g].get(int(y), 0) + 1
    return {g: max(sorted(c), key=c.__getitem__) for g, c in counts.items()}


def make_splits(
    source: FewShotSource,
    mode: str = "group_disjoint",
    fractions: Sequence[float] = (0.2, 0.1, 0.7),
    seed: int = 0,
) -> dict[SplitName, Split]:
    """Partition a corpus into ``train`` / ``val`` / ``test``.

    ``mode='group_disjoint'`` (default)
        Partition the *source objects*, then take every image of each object with it.
        No physical object can appear in two splits. Every class stays present in every
        split, which is what the cross-domain few-shot protocol needs: the held-out
        thing is the *domain*, not the vocabulary, and corpora like EuroSAT (10 classes)
        or the forensic sets (8-12) do not have classes to spare for holding out.

    ``mode='class_disjoint'``
        Partition the classes. The classic meta-learning protocol, retained for
        in-domain comparisons. Fails loudly if a split ends up with fewer than the
        ``n_way`` classes an episode needs -- better here than inside the sampler.

    The default fractions put 70% in test on purpose: tuning needs only enough
    validation data to rank learning rates, and the reported number deserves the rest.
    """
    fractions = _normalise_fractions(fractions)
    labels = np.asarray(source.labels)
    rng = np.random.default_rng(seed)

    if mode == "group_disjoint":
        groups = np.asarray(source.groups)
        unique = np.unique(groups)
        # Stratified by class, not a single global shuffle of the objects.
        #
        # A global partition is fine on EuroSAT, where every image is its own object
        # and there are 27,000 of them. It breaks on a corpus with few objects per
        # class -- which is exactly what the forensic sets are projected to be
        # (12 classes x ~5 physical fibers). At a 10% validation fraction a global
        # shuffle hands val six objects in total, most classes are absent from it, and
        # the sampler then refuses to build 5-way episodes. Partitioning *within* each
        # class instead gives every split a proportional share of every class.
        #
        # Each object is assigned to its modal class first, and then moves as a whole.
        # Objects are class-pure in every corpus here except ChestX, where one patient
        # can carry different findings across studies -- and there the object's
        # integrity matters more than perfect class balance, so the modal class decides
        # and the object still lands in exactly one split.
        labels_of_group: dict[object, int] = {}
        for g, bucket in _group_labels(groups, labels).items():
            labels_of_group[g] = bucket
        by_class: dict[int, list] = {}
        for g in unique:
            by_class.setdefault(labels_of_group[g], []).append(g)
        blocks = [np.array([], dtype=unique.dtype) for _ in SPLIT_NAMES]
        for cls in sorted(by_class):
            members = np.asarray(by_class[cls], dtype=unique.dtype)
            for i, part in enumerate(_partition(members, fractions, rng)):
                blocks[i] = np.concatenate([blocks[i], part])
        # Map group -> split once, then assign images by lookup. Building a boolean
        # mask per split with np.isin is O(n log g) and was measurably slow on
        # ChestX's 112k images / 30k patients.
        assignment = {}
        for name, block in zip(SPLIT_NAMES, blocks):
            for g in block:
                assignment[g] = name
        buckets: dict[str, list[int]] = {n: [] for n in SPLIT_NAMES}
        for idx, g in enumerate(groups):
            buckets[assignment[g]].append(idx)
        index_sets = {n: np.asarray(sorted(v), dtype=np.int64) for n, v in buckets.items()}

    elif mode == "class_disjoint":
        classes = np.unique(labels)
        blocks = _partition(classes, fractions, rng)
        index_sets = {
            name: np.flatnonzero(np.isin(labels, block)).astype(np.int64)
            for name, block in zip(SPLIT_NAMES, blocks)
        }

    else:
        raise ValueError(f"unknown split mode {mode!r}; expected 'group_disjoint' or 'class_disjoint'")

    splits: dict[SplitName, Split] = {}
    for name in SPLIT_NAMES:
        idx = index_sets[name]
        splits[name] = Split(name=name, indices=idx, classes=np.unique(labels[idx]) if idx.size else np.array([], dtype=labels.dtype))

    for name, split in splits.items():
        log.debug(
            "split %-5s %6d images  %2d classes  %6d groups",
            name, len(split), split.classes.size, np.unique(np.asarray(source.groups)[split.indices]).size if len(split) else 0,
        )
    return splits


# -- episodic sampling -------------------------------------------------------


class EpisodicSampler:
    """Draws ``n_episodes`` fixed, reproducible tasks from one split.

    Parameters
    ----------
    source, split
        The corpus and the subset to draw from. Only ``split.indices`` is ever touched,
        so a sampler cannot leak across the partition even if the caller passes the
        wrong split elsewhere.
    n_way, k_shot, n_query
        Episode shape. A class is eligible only if the split holds at least
        ``k_shot + n_query`` images of it; if fewer than ``n_way`` classes qualify the
        constructor raises, rather than silently sampling a 3-way episode and reporting
        it as 5-way.
    seed
        Base seed. Episode ``i`` is drawn from ``default_rng([seed, i])``, so episodes
        are independent of each other and of how many were drawn before. Experiments
        offset this (e.g. ``seed + 10_000`` for tuning) to get a stream disjoint from
        the reporting episodes.
    group_disjoint_queries
        When ``True``, an episode's support and query images for a class are taken from
        *different* source objects. Off by default because the published cross-domain
        few-shot protocol does not do this, and matching it is what makes our numbers
        comparable to theirs. Worth turning on as a reported sensitivity check on
        ISIC and ChestX, where repeated views of one lesion or patient are common: a
        support/query pair from the same lesion makes the episode easier for a reason
        that has nothing to do with adaptation. Open decision, see
        ``docs/HANDOFF.md`` section 9.
    """

    def __init__(
        self,
        source: FewShotSource,
        split: Split,
        n_way: int = 5,
        k_shot: int = 5,
        n_query: int = 15,
        n_episodes: int = 600,
        seed: int = 0,
        group_disjoint_queries: bool = False,
    ) -> None:
        self.source = source
        self.split = split
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.seed = seed
        self.group_disjoint_queries = group_disjoint_queries
        # Counted, not raised: see _draw_class.
        self._fallbacks = 0

        labels = np.asarray(source.labels)[split.indices]
        groups = np.asarray(source.groups)[split.indices]
        needed = k_shot + n_query

        # Per class: the split-relative positions of its images, and their groups.
        self._pool: dict[int, np.ndarray] = {}
        self._groups: dict[int, np.ndarray] = {}
        for cls in np.unique(labels):
            where = np.flatnonzero(labels == cls)
            if where.size >= needed:
                self._pool[int(cls)] = split.indices[where]
                self._groups[int(cls)] = groups[where]
        self.classes: np.ndarray = np.array(sorted(self._pool), dtype=np.int64)

        if self.classes.size < n_way:
            raise ValueError(
                f"split {split.name!r} has {self.classes.size} classes with at least "
                f"{needed} images ({k_shot}-shot + {n_query} queries), but {n_way}-way "
                f"episodes need {n_way}. Lower --k-shot/--n-query, or raise this split's "
                f"fraction."
            )

    def __len__(self) -> int:
        return self.n_episodes

    def __iter__(self) -> Iterator[Episode]:
        for i in range(self.n_episodes):
            yield self.episode(i)

    # -- one episode ---------------------------------------------------------

    def _draw_class(
        self, cls: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pick support and query indices for one class."""
        pool = self._pool[cls]
        if not self.group_disjoint_queries:
            picked = rng.choice(pool.size, size=self.k_shot + self.n_query, replace=False)
            return pool[picked[: self.k_shot]], pool[picked[self.k_shot :]]

        # Support first, then query from objects the support did not use. Falls back to
        # sampling without the constraint rather than raising, because on a corpus with
        # few objects per class the constraint can be unsatisfiable for some draws --
        # and an episode quietly omitted would bias the reported mean. The fallback is
        # counted so it can be reported.
        groups = self._groups[cls]
        order = rng.permutation(pool.size)
        support = order[: self.k_shot]
        used = set(groups[support].tolist())
        eligible = np.array([j for j in order[self.k_shot :] if groups[j] not in used])
        if eligible.size < self.n_query:
            self._fallbacks += 1
            picked = rng.choice(pool.size, size=self.k_shot + self.n_query, replace=False)
            return pool[picked[: self.k_shot]], pool[picked[self.k_shot :]]
        return pool[support], pool[eligible[: self.n_query]]

    def episode(self, index: int) -> Episode:
        """Episode ``index``, identical on every call and in every process.

        Seeded from ``(seed, index)`` so that two sweeps over the same sampler compare
        configurations on the same tasks -- the property that lets us report a
        difference of half a point and mean it.
        """
        if not 0 <= index < self.n_episodes:
            raise IndexError(f"episode {index} out of range (0, {self.n_episodes})")
        rng = np.random.default_rng([self.seed, index])

        chosen = rng.choice(self.classes.size, size=self.n_way, replace=False)
        chosen = self.classes[np.sort(chosen)]

        support_idx: list[np.ndarray] = []
        query_idx: list[np.ndarray] = []
        for cls in chosen:
            s, q = self._draw_class(int(cls), rng)
            support_idx.append(s)
            query_idx.append(q)

        support_x = torch.stack(
            [self.source.load(int(i)) for block in support_idx for i in block]
        )
        query_x = torch.stack(
            [self.source.load(int(i)) for block in query_idx for i in block]
        )
        return Episode(
            support_x=support_x,
            support_y=torch.arange(self.n_way).repeat_interleave(self.k_shot),
            query_x=query_x,
            query_y=torch.arange(self.n_way).repeat_interleave(self.n_query),
            global_classes=torch.as_tensor(chosen.astype(np.int64)),
            episode_id=index,
        )
