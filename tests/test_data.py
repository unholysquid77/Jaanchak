"""Tests for splits, episodic sampling and the corpus loaders.

`src/arc/data/` was absent from every commit until 2026-10-15 (an unanchored `data/`
in .gitignore matched it) and therefore had no tests, despite holding the two
properties nothing downstream can recover from if they are wrong:

1. **No source specimen appears in two splits.** Crops of one fiber, or dermoscopic
   views of one lesion, straddling train and test makes the model a specimen
   recogniser and inflates accuracy for reasons unrelated to the method.
   `docs/DATA_REQUEST.md` calls this the dominant failure mode in small forensic
   imaging studies; on ISIC it is known present, not merely suspected.
2. **Every method sees identical episodes.** Sampling is seeded per episode *index*,
   so episode i is the same task no matter where iteration started or which
   configuration is being evaluated. `sweep_lr.py` re-iterates one sampler per
   configuration and relies on exactly this.

A third property is specific to how `group_disjoint` is implemented: assignment is by
stable hash of `(seed, group)` rather than by an RNG permutation, so it survives a
reordering of the corpus. That is what makes a split reproducible across machines,
numpy versions and a re-scan of a directory tree that returns files in a different
order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from arc.data.base import Episode, EpisodicSampler, make_splits
from arc.data.pathlist import PathListSource, read_csv
from arc.data.sources import ImageFolderSource, TensorSource, build_transform


def _corpus(n_classes: int = 6, specimens: int = 10, views: int = 4) -> TensorSource:
    """A corpus with real repeated-view structure.

    Each image's pixel value is its own index, so a test can assert exactly which
    corpus items an episode drew.
    """
    n = n_classes * specimens * views
    images = torch.stack([torch.full((3, 8, 8), float(i)) for i in range(n)])
    labels, groups = [], []
    for c in range(n_classes):
        for s in range(specimens):
            labels.extend([c] * views)
            groups.extend([f"c{c}_s{s:02d}"] * views)
    return TensorSource(
        images, np.asarray(labels), np.asarray(groups, dtype=object),
        [f"class{c}" for c in range(n_classes)],
    )


@pytest.fixture
def corpus() -> TensorSource:
    return _corpus()


def _ids(x: torch.Tensor) -> set[int]:
    return {int(v) for v in x[:, 0, 0, 0]}


# -- splits: the leakage guarantee -------------------------------------------

def test_group_disjoint_never_puts_one_specimen_in_two_splits(corpus: TensorSource) -> None:
    """The property the whole grouping machinery exists for."""
    splits = make_splits(corpus, "group_disjoint", (0.2, 0.1, 0.7), seed=0)
    owner: dict[object, str] = {}
    for name, split in splits.items():
        for g in corpus.groups[split.indices]:
            assert owner.setdefault(g, name) == name, f"specimen {g} spans {owner[g]} and {name}"


def test_group_disjoint_keeps_every_view_of_a_specimen_together(corpus: TensorSource) -> None:
    """Disjointness alone is not enough: a split must take all four views of a
    specimen or none, or the corpus silently shrinks."""
    splits = make_splits(corpus, "group_disjoint", (0.2, 0.1, 0.7), seed=3)
    for split in splits.values():
        _, counts = np.unique(corpus.groups[split.indices], return_counts=True)
        assert set(counts.tolist()) <= {4}


def test_splits_partition_the_corpus_exactly(corpus: TensorSource) -> None:
    splits = make_splits(corpus, "group_disjoint", (0.2, 0.1, 0.7), seed=1)
    allocated = np.concatenate([s.indices for s in splits.values()])
    assert np.array_equal(np.sort(allocated), np.arange(len(corpus)))


def test_group_assignment_survives_a_reordering_of_the_corpus(corpus: TensorSource) -> None:
    """Why the implementation hashes `(seed, group)` instead of shuffling with an RNG.

    A permutation-based split depends on the order `np.unique` happened to return,
    which depends on the order the corpus was built -- and that changes when a
    directory re-scan returns files differently or a loader skips a missing file. The
    same specimen must land in the same split regardless.
    """
    order = np.random.default_rng(7).permutation(len(corpus))
    shuffled = TensorSource(
        corpus.images[order], corpus.labels[order], corpus.groups[order], corpus.class_names
    )
    a = make_splits(corpus, "group_disjoint", (0.2, 0.1, 0.7), seed=0)
    b = make_splits(shuffled, "group_disjoint", (0.2, 0.1, 0.7), seed=0)
    for name in ("train", "val", "test"):
        assert set(np.unique(corpus.groups[a[name].indices])) == set(
            np.unique(shuffled.groups[b[name].indices])
        ), f"{name} holds different specimens after a reorder"


def test_the_split_is_a_function_of_the_seed_alone(corpus: TensorSource) -> None:
    """`--split-seed` fixes the partition. Learning rates tuned on one split's
    validation set do not transfer to a different partition, and the multi-seed
    reporting runs all assume one fixed test set -- which is why the split seed and the
    episode seed are separate arguments everywhere."""
    a = make_splits(corpus, "group_disjoint", (0.2, 0.1, 0.7), seed=0)["test"].indices
    b = make_splits(corpus, "group_disjoint", (0.2, 0.1, 0.7), seed=0)["test"].indices
    c = make_splits(corpus, "group_disjoint", (0.2, 0.1, 0.7), seed=1)["test"].indices
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_class_disjoint_never_puts_one_class_in_two_splits(corpus: TensorSource) -> None:
    splits = make_splits(corpus, "class_disjoint", (0.5, 0.25, 0.25), seed=0)
    seen: dict[int, str] = {}
    for name, split in splits.items():
        for c in split.classes:
            assert seen.setdefault(int(c), name) == name


def test_class_disjoint_refuses_a_vocabulary_too_small_to_partition() -> None:
    """Why group_disjoint is the default. You cannot hold out 5 classes from 8 and
    still run 5-way episodes, and the forensic sets are projected at 8-12 classes."""
    tiny = _corpus(n_classes=2, specimens=4, views=2)
    with pytest.raises(ValueError, match="empty partition"):
        make_splits(tiny, "class_disjoint", (0.6, 0.2, 0.2), seed=0)


def test_fractions_must_sum_to_one(corpus: TensorSource) -> None:
    """Not normalised silently: fractions that do not sum to 1 are a typo, and
    normalising one would hand back a split the caller did not ask for."""
    with pytest.raises(ValueError, match="sum to 1"):
        make_splits(corpus, "group_disjoint", (0.2, 0.2, 0.2), seed=0)


def test_unknown_split_mode_is_rejected(corpus: TensorSource) -> None:
    with pytest.raises(ValueError, match="unknown split mode"):
        make_splits(corpus, "by_image", (0.2, 0.1, 0.7), seed=0)


# -- sampling: the matched-conditions guarantee ------------------------------

def _sampler(corpus: TensorSource, **kw) -> EpisodicSampler:
    split = make_splits(corpus, "group_disjoint", (0.2, 0.1, 0.7), seed=0)["test"]
    params = {"n_way": 5, "k_shot": 2, "n_query": 3, "n_episodes": 8, "seed": 0}
    params.update(kw)
    return EpisodicSampler(corpus, split, **params)


def test_episodes_have_the_declared_shape(corpus: TensorSource) -> None:
    ep = _sampler(corpus).episode(0)
    assert ep.support_x.shape == (10, 3, 8, 8) and ep.query_x.shape == (15, 3, 8, 8)
    assert ep.n_way == 5
    assert torch.equal(ep.support_y, torch.arange(5).repeat_interleave(2))
    assert torch.equal(ep.query_y, torch.arange(5).repeat_interleave(3))
    assert ep.global_classes.numel() == 5


def test_re_iterating_a_sampler_yields_byte_identical_episodes(corpus: TensorSource) -> None:
    """What makes "every method saw the same episodes" structural rather than hopeful.
    sweep_lr.py re-iterates one sampler per configuration for exactly this reason, and
    notes that it deliberately does not materialise the episodes."""
    sampler = _sampler(corpus)
    first = [e.support_x.clone() for e in sampler]
    second = [e.support_x.clone() for e in sampler]
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_an_episode_does_not_depend_on_how_many_were_drawn_before(corpus: TensorSource) -> None:
    """Seeded per index, not from a running stream -- so a resumed or truncated run
    produces the same episode 7 as an uninterrupted one. --skip-existing depends on it."""
    direct = _sampler(corpus).episode(7)
    streamed = list(_sampler(corpus))[7]
    assert torch.equal(direct.support_x, streamed.support_x)
    assert torch.equal(direct.query_x, streamed.query_x)


def test_different_episode_seeds_draw_different_episodes(corpus: TensorSource) -> None:
    assert not torch.equal(
        _sampler(corpus, seed=0).episode(0).support_x,
        _sampler(corpus, seed=1).episode(0).support_x,
    )


def test_support_and_query_never_share_an_image(corpus: TensorSource) -> None:
    """Adaptation runs on support; query is scored under no_grad. An image in both is
    a free answer. `replace=False` over one pooled draw is what prevents it."""
    for ep in _sampler(corpus):
        assert not (_ids(ep.support_x) & _ids(ep.query_x))


def test_the_sampler_draws_only_from_its_own_split(corpus: TensorSource) -> None:
    """The last line of defence: even handed the wrong split elsewhere, a sampler
    cannot reach across the partition."""
    split = make_splits(corpus, "group_disjoint", (0.2, 0.1, 0.7), seed=0)["val"]
    allowed = set(split.indices.tolist())
    for ep in EpisodicSampler(corpus, split, n_way=5, k_shot=1, n_query=1, n_episodes=5):
        assert (_ids(ep.support_x) | _ids(ep.query_x)) <= allowed


def test_too_few_classes_for_n_way_raises_instead_of_shrinking_the_episode(
    corpus: TensorSource,
) -> None:
    """A 3-way episode reported as 5-way is a wrong number, not a smaller one."""
    split = make_splits(corpus, "class_disjoint", (0.5, 0.25, 0.25), seed=0)["val"]
    with pytest.raises(ValueError, match="need n_way"):
        EpisodicSampler(corpus, split, n_way=5, k_shot=1, n_query=1, n_episodes=1)


def test_a_class_with_too_few_images_raises_rather_than_being_dropped(
    corpus: TensorSource,
) -> None:
    """Loudly, and naming the classes. Silently excluding a thin class would change
    the label vocabulary an experiment ran on without saying so -- the same class of
    defect as the untuned baselines that were reported as tuned."""
    with pytest.raises(ValueError, match="fewer than k_shot"):
        _sampler(corpus, k_shot=40, n_query=10)


def test_out_of_range_episode_indices_do_not_silently_wrap(corpus: TensorSource) -> None:
    """`episode(i)` is seeded from i, so any i produces a valid-looking task. What
    must hold is that iteration stops at n_episodes."""
    sampler = _sampler(corpus, n_episodes=4)
    assert len(list(sampler)) == 4 == len(sampler)


# -- Episode -----------------------------------------------------------------

def test_episode_to_returns_a_new_episode_rather_than_mutating_in_place() -> None:
    """The runner calls `.to(device)` on every episode it is handed, and a sampler
    re-iterated per configuration must not find its own episodes rebound to the GPU.
    (`Episode` is frozen, so this is enforced by construction; the test documents the
    contract the runner relies on.)"""
    ep = Episode(
        support_x=torch.zeros(2, 3, 4, 4), support_y=torch.zeros(2, dtype=torch.long),
        query_x=torch.zeros(2, 3, 4, 4), query_y=torch.zeros(2, dtype=torch.long),
        global_classes=torch.arange(2), episode_id=0,
    )
    moved = ep.to("cpu")
    assert moved is not ep and moved.episode_id == ep.episode_id
    assert torch.equal(moved.support_x, ep.support_x)


# -- loaders -----------------------------------------------------------------

def _write_tree(root: Path, pattern: str = "{cls}_f{spec:02d}_c{view:02d}.png") -> Path:
    for c in range(3):
        for spec in range(2):
            for view in range(2):
                d = root / f"class{c}"
                d.mkdir(parents=True, exist_ok=True)
                name = pattern.format(cls=f"class{c}", spec=spec, view=view)
                Image.new("RGB", (16, 16), (10 * c, 20 * spec, 30 * view)).save(d / name)
    return root


def test_image_folder_groups_crops_of_one_specimen_together(tmp_path: Path) -> None:
    """The concrete ask in docs/DATA_REQUEST.md: specimen id in the filename, loader
    takes a regex. 12 images, 6 physical specimens."""
    src = ImageFolderSource(_write_tree(tmp_path), group_pattern=r"(.+?)_c\d+")
    assert len(src) == 12
    assert len(np.unique(src.groups)) == 6
    assert src.load(0).shape == (3, 224, 224)


def test_image_folder_warns_when_no_group_pattern_is_given(tmp_path: Path, caplog) -> None:
    """Leaving it off asserts every image is an independent specimen. That is often
    false for trace evidence, so it must not pass silently."""
    with caplog.at_level("WARNING"):
        src = ImageFolderSource(_write_tree(tmp_path))
    assert len(np.unique(src.groups)) == 12
    assert any("independent specimen" in r.message for r in caplog.records)


def test_image_folder_warns_when_the_group_pattern_matches_nothing(
    tmp_path: Path, caplog
) -> None:
    """A pattern that silently fell back to per-image groups would reinstate exactly
    the leakage it was passed to prevent, so the count of unmatched names is reported."""
    with caplog.at_level("WARNING"):
        src = ImageFolderSource(_write_tree(tmp_path), group_pattern=r"(.+?)_zzz\d+")
    assert len(np.unique(src.groups)) == 12
    assert any("did not match" in r.message for r in caplog.records)


def test_image_folder_rejects_a_root_with_no_class_directories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no class subdirectories"):
        ImageFolderSource(tmp_path)


def test_read_csv_tolerates_a_byte_order_mark_and_indexes_columns(tmp_path: Path) -> None:
    """Several of these manifests ship with a BOM, which otherwise corrupts the first
    column name and surfaces as a baffling KeyError on 'image'."""
    path = tmp_path / "bom.csv"
    path.write_bytes("﻿image,MEL,NV\nISIC_0,1.0,0.0\n".encode())
    cols, rows = read_csv(path)
    assert cols == {"image": 0, "MEL": 1, "NV": 2}
    assert rows == [["ISIC_0", "1.0", "0.0"]]


def test_read_csv_rejects_an_empty_manifest(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(ValueError, match="is empty"):
        read_csv(path)


def test_pathlist_rejects_mismatched_lists() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        PathListSource([Path("a.png")], [0, 1], ["g"], ["x"])


def test_pathlist_rejects_an_empty_corpus() -> None:
    with pytest.raises(ValueError, match="no images"):
        PathListSource([], [], [], ["x"])


def test_build_transform_is_deterministic_in_eval_mode_and_random_in_train_mode() -> None:
    """Augmentation belongs to the inner loop (arc.train.augment, on the GPU, applied
    to an already-batched episode). Putting randomness in the corpus transform would
    put it inside the "identical episodes for every method" guarantee."""
    # A gradient, not a flat colour: random crops of a uniform image are identical,
    # which would make the train-mode half of this test pass for the wrong reason.
    img = Image.fromarray(
        (np.mgrid[0:48, 0:64][1] * 4 % 256).astype("uint8")[:, :, None].repeat(3, 2)
    )
    evaluate = build_transform(32)
    assert torch.equal(evaluate(img), evaluate(img))
    assert evaluate(img).shape == (3, 32, 32)
    torch.manual_seed(0)
    train = build_transform(32, train=True)
    assert not torch.equal(train(img), train(img))
