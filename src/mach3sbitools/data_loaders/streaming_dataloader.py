"""
Shard-sequential, shuffle-buffer streaming dataset.

:class:`TrainingDataset` does fully random, per-row memory-mapped access
across every ``.feather`` file. That's fine as long as the OS page cache
can hold most of the working set. Once the dataset is large enough that
the RAM budget is a small fraction of total data on disk (e.g. ~100GB of
RAM against ~1TB of simulations), fully random row access means most
batches miss the page cache and hit real disk I/O -- fine on local NVMe,
painful on spinning disk or networked storage.

``ShuffleBufferDataset`` trades perfect global shuffling for a *bounded*
RAM footprint and a much friendlier I/O pattern: shards are read mostly
sequentially (cheap even on slow storage), rows are pushed into a
fixed-size in-RAM buffer, and output rows are popped from random
positions in that buffer while it's continuously refilled. This is the
same technique used by WebDataset / tf.data for datasets that don't fit
in memory.

This is an ``IterableDataset``: it doesn't support index-based access,
``len()``, or ``random_split``. Train/validation splitting has to happen
at the shard (file) level *before* constructing the dataset -- e.g. split
``data_folder`` into two subdirectories, or pass an explicit file list.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

from mach3sbitools.simulator import Prior
from mach3sbitools.utils import FeatherFileHandle


class ShuffleBufferDataset(IterableDataset):
    """
    Streams ``(theta, x)`` rows from many feather shards via a shuffle buffer.

    :param data_folder: Directory containing ``.feather`` files, or an
        explicit ``list[Path]`` of shard files (useful for a pre-split
        validation set drawn from the same folder).
    :param prior: Prior providing ``nuisance_filter`` used to select which
        parameters are kept in *theta*.
    :param buffer_size: Number of rows held in the shuffle buffer at once.
        Size this to comfortably fit your RAM budget -- roughly
        ``buffer_size * (theta_dim + x_dim) * 4`` bytes per worker. A few
        million rows is typically a few to a few tens of GB depending on
        dimensionality; with several DataLoader workers each holding their
        own buffer, divide your RAM budget by ``num_workers`` accordingly.
    :param seed: Base seed, combined with worker id and epoch so shard
        order and in-shard shuffling differ each epoch but stay
        reproducible.
    :raises FileNotFoundError: If no ``.feather`` files are found.
    """

    def __init__(
        self,
        data_folder: Path | list[Path],
        prior: Prior,
        buffer_size: int = 2_000_000,
        seed: int = 0,
    ) -> None:
        if isinstance(data_folder, (list, tuple)):
            self.files = sorted(Path(f) for f in data_folder)
        else:
            data_folder = Path(data_folder)
            self.files = sorted(data_folder.glob("*.feather"))

        if not self.files:
            raise FileNotFoundError(f"No .feather files found for {data_folder}")

        self.prior = prior
        self.buffer_size = buffer_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """
        Call at the start of each epoch (e.g. from a training callback) to
        reshuffle shard order and in-shard row order. Without this every
        epoch streams shards in the same order.
        """
        self.epoch = epoch

    def _worker_files(self) -> list[Path]:
        info = get_worker_info()
        files = list(self.files)
        random.Random(self.seed + self.epoch).shuffle(files)
        if info is None:
            return files
        # Disjoint, roughly-equal shard subsets per worker, each streamed
        # sequentially -- avoids workers contending for the same files.
        return files[info.id :: info.num_workers]

    def __iter__(self):
        info = get_worker_info()
        worker_seed = self.seed + self.epoch * 1_000 + (info.id if info else 0)
        rng = random.Random(worker_seed)

        nuisance_filter = self.prior.nuisance_filter
        if isinstance(nuisance_filter, torch.Tensor):
            nuisance_filter = nuisance_filter.to("cpu").numpy()

        buffer: list[tuple[torch.Tensor, torch.Tensor]] = []

        for f in self._worker_files():
            handle = FeatherFileHandle(f)
            try:
                order = list(range(handle.num_rows))
                rng.shuffle(order)  # local shuffle within the shard
                for local_idx in order:
                    theta = handle.theta[local_idx]
                    x = handle.x[local_idx]
                    if nuisance_filter is not None:
                        theta = theta[nuisance_filter]

                    theta_t = torch.from_numpy(theta.astype("float32", copy=True))
                    x_t = torch.from_numpy(x.astype("float32", copy=True))

                    if len(buffer) < self.buffer_size:
                        buffer.append((theta_t, x_t))
                        continue

                    # Buffer full: evict a random slot, insert the new row.
                    # Bounded-memory streaming (reservoir-style) shuffle.
                    j = rng.randrange(self.buffer_size)
                    out = buffer[j]
                    buffer[j] = (theta_t, x_t)
                    yield out
            finally:
                handle.close()

        rng.shuffle(buffer)
        yield from buffer
