"""
Block-shuffled sampling for datasets too large to shuffle row-by-row.

Why not ``shuffle=True``
------------------------
A fully random permutation turns every batch into ``batch_size`` scattered
single-row reads. Storage does not serve single rows: a parallel filesystem
serves a stripe and a local filesystem serves a record, so each 1 KiB row
costs a full record of actual I/O. At a 128 KiB ZFS recordsize an 18 MB
batch becomes roughly 1 GB of reads -- a ~55x amplification that turns a
compute-bound run straight back into an input-bound one.

Why not ``shuffle=False`` either
--------------------------------
Reading in file order means every epoch sees identical rows in identical
order, and consecutive samples come from whichever shard happened to be
merged there. Correlated batches give worse gradients, so the GPU time is
spent less usefully even though it is fully occupied.

The compromise
--------------
Shuffle *blocks* of contiguous rows rather than rows. One block is sized to
land near the filesystem's record/stripe size, so each read stays as cheap
as a sequential one, while the batch as a whole is assembled from blocks
drawn from all over the dataset. With ~100M rows and a 128-row block there
are ~760k independently-ordered blocks per epoch, which decorrelates batches
as well as a true shuffle does for training purposes.

Rows *within* a block are shuffled too. That is free:
:meth:`TrainingDataset._read_rows` sorts indices before reading and restores
the caller's order with an in-RAM gather, so intra-block order never reaches
the filesystem.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

#: Blocks each rank should get at least, before block_size is shrunk to
#: provide them. Only whole blocks are yielded -- a short tail block would
#: make per-rank lengths unequal, which is a DDP hang rather than a visible
#: bug -- so the trailing partial block is dropped, and this bounds that loss
#: to roughly 1/_MIN_BLOCKS_PER_RANK of the dataset. At 97.8M rows and a
#: 128-row block nothing is clamped and the loss is 64 rows; at 200 rows the
#: block shrinks so the loss stays a couple of rows rather than a third.
_MIN_BLOCKS_PER_RANK = 64


class BlockShuffleSampler(Sampler[int]):
    """
    Yield indices in shuffled fixed-size blocks of contiguous rows.

    Distribution-aware: pass ``num_replicas``/``rank`` to give each DDP rank a
    disjoint, equal-length share. Equal length matters -- ranks that disagree
    on batch count deadlock on the next collective -- so the trailing partial
    block is dropped and blocks are padded out to a whole multiple of the
    world size. At realistic sizes this discards well under one block's worth
    of rows per rank.

    :param n_samples: Length of the dataset being sampled.
    :param block_size: Rows per contiguous block. Aim for one filesystem
        record or stripe: 128 rows of ~1 KiB lands near a 128 KiB ZFS record.
        Silently reduced when the dataset is too small to yield one block per
        rank; :attr:`requested_block_size` keeps the original.
    :param shuffle: Shuffle block order (and rows within a block). ``False``
        yields plain sequential order, which is what validation wants.
    :param seed: Base seed; the epoch is added to it so each epoch differs
        while all ranks stay in agreement.
    :param num_replicas: DDP world size. ``None`` reads it from the default
        process group, falling back to 1.
    :param rank: This process's DDP rank, resolved the same way.
    """

    def __init__(
        self,
        n_samples: int,
        block_size: int = 128,
        *,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        super().__init__()
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        if num_replicas is None or rank is None:
            world, this_rank = _resolve_dist()
            num_replicas = world if num_replicas is None else num_replicas
            rank = this_rank if rank is None else rank
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} out of range for {num_replicas} replicas")

        if n_samples < num_replicas:
            raise ValueError(
                f"cannot give {num_replicas} ranks a disjoint share of {n_samples} rows"
            )

        self.n_samples = n_samples
        self.requested_block_size = block_size
        # A block size tuned for the training set is routinely too coarse for
        # the validation split, or for a small test dataset -- 20 rows at a
        # 128-row block is zero blocks. Shrink to fit rather than refusing:
        # on a dataset that small the read pattern is irrelevant anyway, and
        # the limit is reached only when the dataset is too small to care.
        self.block_size = min(
            block_size,
            max(1, n_samples // (num_replicas * _MIN_BLOCKS_PER_RANK)),
        )
        self.shuffle = shuffle
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self._epoch_set_externally = False

        # Whole blocks only: a short tail block would make per-rank lengths
        # unequal, and an unequal length is a DDP hang rather than a bug you
        # notice in the logs.
        self.n_blocks = n_samples // self.block_size
        self.blocks_per_rank = self.n_blocks // num_replicas
        self._len = self.blocks_per_rank * self.block_size

    def set_epoch(self, epoch: int) -> None:
        """
        Reseed for *epoch*.

        Lightning calls this on every sampler that defines it at the start of
        each epoch. Calling it also switches off the internal epoch counter,
        so the two cannot both advance and skip an ordering.
        """
        self.epoch = epoch
        self._epoch_set_externally = True

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[int]:
        if self.shuffle and not self._epoch_set_externally:
            # Nothing is driving set_epoch (plain PyTorch loop, or a Lightning
            # version that does not). Advance here so epochs still differ.
            self.epoch += 1

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            block_order = torch.randperm(self.n_blocks, generator=g)
        else:
            g = None
            block_order = torch.arange(self.n_blocks)

        # Strided rather than chunked so that a rank's blocks are spread over
        # the whole file instead of confined to one contiguous fifth of it.
        mine = block_order[self.rank :: self.num_replicas][: self.blocks_per_rank]

        for block in mine.tolist():
            start = block * self.block_size
            if g is None:
                yield from range(start, start + self.block_size)
            else:
                offsets = torch.randperm(self.block_size, generator=g)
                yield from (start + int(o) for o in offsets)


def _resolve_dist() -> tuple[int, int]:
    """World size and rank from the default process group, or ``(1, 0)``."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    return 1, 0
