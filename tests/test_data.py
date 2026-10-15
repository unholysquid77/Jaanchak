"""Tests for splits and episodic sampling.

These guard the two properties that nothing downstream can recover from if they are
wrong, and that a passing run would not reveal:

1. **No source object appears in two splits.** Crops of one fiber, or photographs of
   one lesion, straddling train and test makes the model an object recogniser and
   inflates accuracy for a reason unrelated to the method. ``docs/DATA_REQUEST.md``
   calls this the dominant failure mode in small forensic imaging studies.
2. **Every method sees identical episodes.** The sampler seeds per episode *index*, so
   re-iterating it yields byte-identical tasks. Without that, a sweep's half-point
   differences are comparing configurations on different data.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from arc.data.base import Episode, EpisodicSampler, make_splits
from arc.data.pathlist import PathListSource


class _FakeSource:
    """A corpus of deterministic tensors -- no disk, no PIL.

    ``load(i)`` returns a constant image whose value *is* the index, so a test can
    assert exactly which corpus images an episode drew.
    """

    def __init__(self, n_classes: int = 6, objects_per_class: int = 10, views: int = 4) -> None:
        paths, labels, groups = [], [], []
        for c in range(n_classes):
            for o in range(objects_per_class):
                for v in range(views):
                    paths.append(f"c{c}/obj{o}_view{v}.png")
                    labels.append(c)
                    groups.append(f"c{c}_obj{o}")
        self.paths = paths
        self.labels = np.asarray(labels)
        self.groups = np.asarray(groups, dtype=object)
        self.class_names = tuple(f"class{c}" for c in range(n_classes))

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, index: int) -> torch.Tensor:
        return torch.full((3, 8, 8), float(index))


@pytest.fixture
def source() -> _FakeSource:
    return _FakeSource()


# -- splits ------------------------------------------------------------------

def test_group_disjoint_never_puts_one_object_in_two_splits(source: _FakeSource) -> None:
    """The property the whole grouping machinery exists for."""
    splits = make_splits(source, mode="group_disjoint", fractions=(0.2, 0.2, 0.6), seed=0)
    owner: dict[str, str] = {}
    for name, split in splits.items():
        for g in source.groups[split.indices]:
            assert owner.setdefault(g, name) == name, f"object {g} spans {owner[g]} and {name}"


def test_group_disjoint_keeps_every_view_of_an_object_together(source: _FakeSource) -> None:
    """Disjointness is not enough -- a split must hold *all* four views or none, or the
    corpus silently shrinks."""
    splits = make_splits(source, mode="group_disjoint", seed=3)
    for split in splits.values():
        for g, count in zip(*np.unique(source.groups[split.indices], return_counts=True)):
            assert count == 4, f"object {g} contributed {count} of its 4 views"


def test_splits_partition_the_corpus_exactly(source: _FakeSource) -> None:
    """No image dropped, none duplicated. Flooring the block sizes makes the remainder
    land somewhere; this pins where."""
    splits = make_splits(source, mode="group_disjoint", fractions=(0.2, 0.2, 0.6), seed=1)
    allocated = np.concatenate([s.indices for s in splits.values()])
    assert np.array_equal(np.sort(allocated), np.arange(len(source)))


def test_group_disjoint_keeps_every_class_in_every_split(source: _FakeSource) -> None:
    """Why group_disjoint is the default: the cross-domain protocol holds out the
    domain, not the vocabulary, and these corpora have 7-12 classes to begin with.

    This is what the per-class stratification in make_splits buys. A single global
    shuffle of the objects leaves most classes out of a 10-20% split once a class has
    only a handful of physical objects -- the projected shape of the forensic sets."""
    splits = make_splits(source, mode="group_disjoint", fractions=(0.2, 0.2, 0.6), seed=0)
    for name, split in splits.items():
        assert split.classes.size == len(source.class_names), f"{name} lost a class"


def test_class_disjoint_never_puts_one_class_in_two_splits(source: _FakeSource) -> None:
    splits = make_splits(source, mode="class_disjoint", fractions=(0.5, 0.25, 0.25), seed=0)
    seen: dict[int, str] = {}
    for name, split in splits.items():
        for c in split.classes:
            assert seen.setdefault(int(c), name) == name


def test_the_split_is_a_function_of_the_seed_alone(source: _FakeSource) -> None:
    """``--split-seed`` must fix the partition: learning rates tuned on one split's
    validation set do not transfer to a different partition, and the multi-seed
    reporting runs all assume one fixed test set."""
    a = make_splits(source, seed=0)["test"].indices
    b = make_splits(source, seed=0)["test"].indices
    c = make_splits(source, seed=1)["test"].indices
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_fractions_need_not_sum_to_one_but_must_be_sane(source: _FakeSource) -> None:
    equivalent = make_splits(source, fractions=(2, 2, 6), seed=0)["test"].indices
    assert np.array_equal(equivalent, make_splits(source, fractions=(0.2, 0.2, 0.6), seed=0)["test"].indices)
    with pytest.raises(ValueError, match="non-negative"):
        make_splits(source, fractions=(-0.1, 0.5, 0.6))
    with pytest.raises(ValueError, match="three fractions"):
        make_splits(source, fractions=(0.5, 0.5))


def test_unknown_split_mode_is_rejected(source: _FakeSource) -> None:
    with pytest.raises(ValueError, match="unknown split mode"):
        make_splits(source, mode="by_image")


# -- sampling ----------------------------------------------------------------

def _sampler(source: _FakeSource, **kw) -> EpisodicSampler:
    splits = make_splits(source, mode="group_disjoint", fractions=(0.2, 0.2, 0.6), seed=0)
    params = {"n_way": 5, "k_shot": 2, "n_query": 3, "n_episodes": 8, "seed": 0}
    params.update(kw)
    return EpisodicSampler(source, splits["test"], **params)


def test_episodes_have_the_declared_shape(source: _FakeSource) -> None:
    ep = _sampler(source).episode(0)
    assert ep.support_x.shape == (10, 3, 8, 8) and ep.query_x.shape == (15, 3, 8, 8)
    assert ep.n_way == 5 and ep.k_shot == 2
    assert torch.equal(ep.support_y, torch.arange(5).repeat_interleave(2))
    assert torch.equal(ep.query_y, torch.arange(5).repeat_interleave(3))
    assert ep.global_classes.numel() == 5


def test_re_iterating_a_sampler_yields_byte_identical_episodes(source: _FakeSource) -> None:
    """What makes 'every method saw the same episodes' structural rather than hopeful.
    sweep_lr.py re-iterates one sampler per configuration for exactly this reason."""
    sampler = _sampler(source)
    first = [e.support_x.clone() for e in sampler]
    second = [e.support_x.clone() for e in sampler]
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_an_episode_does_not_depend_on_how_many_were_drawn_before(source: _FakeSource) -> None:
    """Seeding per index, not from a running stream -- so a resumed or truncated run
    produces the same episode 7 as an uninterrupted one."""
    direct = _sampler(source).episode(7)
    streamed = list(_sampler(source))[7]
    assert torch.equal(direct.support_x, streamed.support_x)
    assert torch.equal(direct.query_x, streamed.query_x)


def test_different_seeds_draw_different_episodes(source: _FakeSource) -> None:
    assert not torch.equal(
        _sampler(source, seed=0).episode(0).support_x,
        _sampler(source, seed=1).episode(0).support_x,
    )


def test_support_and_query_never_share_an_image(source: _FakeSource) -> None:
    """Adaptation runs on support; query is scored. An image in both is a free answer."""
    for ep in _sampler(source):
        s = {int(v) for v in ep.support_x[:, 0, 0, 0]}
        q = {int(v) for v in ep.query_x[:, 0, 0, 0]}
        assert not (s & q)


def test_the_sampler_draws_only_from_its_own_split(source: _FakeSource) -> None:
    """The last line of defence: even handed the wrong split elsewhere, a sampler
    cannot reach across the partition."""
    splits = make_splits(source, mode="group_disjoint", fractions=(0.2, 0.2, 0.6), seed=0)
    allowed = set(splits["val"].indices.tolist())
    sampler = EpisodicSampler(source, splits["val"], n_way=5, k_shot=1, n_query=1, n_episodes=5)
    for ep in sampler:
        drawn = {int(v) for v in ep.support_x[:, 0, 0, 0]} | {int(v) for v in ep.query_x[:, 0, 0, 0]}
        assert drawn <= allowed


def test_too_few_eligible_classes_raises_instead_of_shrinking_the_episode(source: _FakeSource) -> None:
    """A 3-way episode reported as 5-way is a wrong number, not a smaller one."""
    splits = make_splits(source, mode="class_disjoint", fractions=(0.5, 0.25, 0.25), seed=0)
    with pytest.raises(ValueError, match="way"):
        EpisodicSampler(source, splits["val"], n_way=5, k_shot=1, n_query=1)


def test_a_class_without_enough_images_is_excluded_not_sampled_with_replacement(
    source: _FakeSource,
) -> None:
    """Duplicating a support image into the query set would be the same free answer as
    above, arrived at by a different route."""
    sampler = _sampler(source, k_shot=10, n_query=4)
    assert sampler.classes.size == len(source.class_names)
    with pytest.raises(ValueError, match="at least"):
        _sampler(source, k_shot=40, n_query=10)


def test_group_disjoint_queries_separates_support_and_query_objects(source: _FakeSource) -> None:
    """The opt-in episode-level analogue of the split-level guarantee. Off by default
    to match the published protocol; this pins that it works when asked for."""
    splits = make_splits(source, mode="group_disjoint", fractions=(0.2, 0.2, 0.6), seed=0)
    sampler = EpisodicSampler(
        source, splits["test"], n_way=5, k_shot=1, n_query=2, n_episodes=10,
        group_disjoint_queries=True,
    )
    for ep in sampler:
        s_groups = source.groups[[int(v) for v in ep.support_x[:, 0, 0, 0]]]
        q_groups = source.groups[[int(v) for v in ep.query_x[:, 0, 0, 0]]]
        assert not (set(s_groups) & set(q_groups))
    assert sampler._fallbacks == 0, "the constraint should be satisfiable on this corpus"


def test_out_of_range_episodes_are_refused(source: _FakeSource) -> None:
    sampler = _sampler(source)
    with pytest.raises(IndexError):
        sampler.episode(len(sampler))


# -- Episode -----------------------------------------------------------------

def test_episode_to_returns_a_new_episode_rather_than_mutating_in_place() -> None:
    """The runner calls ``.to(device)`` on every episode it is handed, and a sampler
    re-iterated per configuration must not find its own episodes rebound to the GPU.
    (``Tensor.to`` is a no-op returning the same object when the tensor is already on
    the target device, so what is asserted is the wrapper, not tensor identity.)"""
    ep = Episode(
        support_x=torch.zeros(2, 3, 4, 4), support_y=torch.zeros(2, dtype=torch.long),
        query_x=torch.zeros(2, 3, 4, 4), query_y=torch.zeros(2, dtype=torch.long),
        global_classes=torch.arange(2), episode_id=0,
    )
    before = ep.support_x
    moved = ep.to("cpu")
    assert moved is not ep
    assert ep.support_x is before, "the source episode must be left alone"
    assert moved.episode_id == ep.episode_id
    assert torch.equal(moved.support_x, before)


# -- PathListSource ----------------------------------------------------------

def test_pathlist_reports_the_corpus_inflation_a_dataset_table_needs() -> None:
    """'1,200 images' from 60 objects oversells the sample size by 20x. This is the
    number that belongs in the paper."""
    src = PathListSource(
        paths=["a_c0.png", "a_c1.png", "b_c0.png"],
        labels=[0, 0, 1], groups=["a", "a", "b"],
        class_names=("x", "y"), transform=lambda im: torch.zeros(3, 4, 4),
    )
    assert src.group_summary() == {
        "images": 3, "groups": 2, "images_sharing_a_group": 2, "largest_group": 2,
    }


def test_pathlist_rejects_a_label_with_no_class_name() -> None:
    with pytest.raises(ValueError, match="class_names"):
        PathListSource(["a.png"], [5], None, ("x",), lambda im: torch.zeros(3, 4, 4))


def test_pathlist_defaults_to_one_group_per_image() -> None:
    src = PathListSource(["a.png", "b.png"], [0, 0], None, ("x",), lambda im: torch.zeros(1))
    assert len(np.unique(src.groups)) == 2
