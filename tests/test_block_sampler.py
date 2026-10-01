"""
Tests for BlockShuffleSampler.

The sampler exists to satisfy two requirements at once: batches must be
decorrelated, and each batch must still resolve to a handful of contiguous
reads. Losing either one is silent -- the first shows up as a worse-
converging model, the second as a training run that is suddenly input-bound
again -- so both are asserted here rather than left to a benchmark.
"""

import itertools

import pytest

from mach3sbitools.data_loaders import BlockShuffleSampler


def _contiguous_runs(indices):
    """How many separate reads a sorted index list would cost."""
    runs = 1
    for prev, nxt in itertools.pairwise(sorted(indices)):
        if nxt != prev + 1:
            runs += 1
    return runs


class TestOrdering:
    def test_indices_are_unique_and_in_range(self):
        s = BlockShuffleSampler(10_000, block_size=128)
        idx = list(s)
        assert len(set(idx)) == len(idx)
        assert 0 <= min(idx) and max(idx) < 10_000
        assert len(idx) == len(s)

    def test_epochs_differ_but_are_reproducible(self):
        s = BlockShuffleSampler(10_000, block_size=128, seed=42)
        s.set_epoch(0)
        first = list(s)
        s.set_epoch(1)
        second = list(s)
        s.set_epoch(0)
        again = list(s)
        assert first == again, "same epoch must reproduce"
        assert first != second, "different epochs must differ"

    def test_advances_without_set_epoch(self):
        """Plain PyTorch loops never call set_epoch; reshuffle anyway."""
        s = BlockShuffleSampler(10_000, block_size=128)
        assert list(s) != list(s)

    def test_set_epoch_disables_auto_advance(self):
        """Otherwise both would advance and silently skip an ordering."""
        s = BlockShuffleSampler(10_000, block_size=128)
        s.set_epoch(4)
        assert list(s) == list(s)

    def test_shuffle_false_is_strictly_sequential(self):
        s = BlockShuffleSampler(1000, block_size=16, shuffle=False)
        idx = list(s)
        assert idx == list(range(len(idx)))

    def test_order_is_not_sequential_when_shuffling(self):
        s = BlockShuffleSampler(10_000, block_size=128)
        s.set_epoch(0)
        idx = list(s)
        assert idx != sorted(idx)


class TestReadLocality:
    @pytest.mark.parametrize("block_size", [16, 64, 128])
    def test_batch_costs_one_read_per_block(self, block_size):
        """
        The whole point: a batch of B rows should cost about B/block_size
        reads, not B. This is what keeps the dataloader off the slow path.
        """
        batch = 1024
        s = BlockShuffleSampler(5_000_000, block_size=block_size, seed=1)
        s.set_epoch(0)
        idx = list(itertools.islice(iter(s), batch))
        assert _contiguous_runs(idx) <= batch // block_size

    def test_full_shuffling_would_be_far_worse(self):
        """Guards the premise: blocks must beat row-granular shuffling."""
        import random

        batch = 1024
        s = BlockShuffleSampler(5_000_000, block_size=128, seed=1)
        s.set_epoch(0)
        blocked = _contiguous_runs(list(itertools.islice(iter(s), batch)))
        scattered = _contiguous_runs(random.Random(0).sample(range(5_000_000), batch))
        assert blocked * 20 < scattered


class TestDistributed:
    @pytest.mark.parametrize("world", [1, 2, 4, 8])
    def test_ranks_are_disjoint_and_equal_length(self, world):
        """Unequal per-rank lengths deadlock DDP on the next collective."""
        shards = []
        for r in range(world):
            s = BlockShuffleSampler(
                100_000, block_size=128, seed=7, num_replicas=world, rank=r
            )
            s.set_epoch(3)
            shards.append(list(s))

        assert len({len(sh) for sh in shards}) == 1, "unequal rank lengths"
        flat = [i for sh in shards for i in sh]
        assert len(set(flat)) == len(flat), "ranks overlap"

    def test_rank_must_be_in_range(self):
        with pytest.raises(ValueError):
            BlockShuffleSampler(1000, num_replicas=4, rank=4)

    def test_rejects_impossible_split(self):
        with pytest.raises(ValueError, match="disjoint share"):
            BlockShuffleSampler(3, block_size=128, num_replicas=4)


class TestSmallDatasets:
    @pytest.mark.parametrize(
        "n,world", [(20, 1), (20, 4), (200, 1), (200, 4), (1000, 1)]
    )
    def test_small_datasets_keep_almost_every_row(self, n, world):
        """
        block_size is clamped when the dataset is too small for it, so a tiny
        validation split does not lose a third of its rows to the tail.
        """
        lens = [
            len(BlockShuffleSampler(n, block_size=128, num_replicas=world, rank=r))
            for r in range(world)
        ]
        assert len(set(lens)) == 1
        covered = lens[0] * world
        assert covered >= 0.98 * n, f"{n - covered}/{n} rows dropped"

    def test_block_size_is_not_clamped_at_scale(self):
        s = BlockShuffleSampler(97_800_000, block_size=128)
        assert s.block_size == 128 == s.requested_block_size
        assert len(s) >= 97_800_000 - 128

    def test_validation_fraction_sized_split_works(self):
        """The case that used to raise: 10% of a small dataset."""
        s = BlockShuffleSampler(20, block_size=128, shuffle=False)
        assert list(s) == list(range(20))


class TestValidation:
    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_bad_block_size(self, bad):
        with pytest.raises(ValueError, match="block_size"):
            BlockShuffleSampler(1000, block_size=bad)

    @pytest.mark.parametrize("bad", [0, -5])
    def test_rejects_bad_n_samples(self, bad):
        with pytest.raises(ValueError, match="n_samples"):
            BlockShuffleSampler(bad)
