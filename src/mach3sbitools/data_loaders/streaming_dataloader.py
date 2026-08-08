"""
Row-level streaming dataset over a folder of ``.feather`` simulation shards.

Unlike :class:`~mach3sbitools.data_loaders.TrainingDataset` +
:meth:`~mach3sbitools.data_loaders.TrainingDataset.to_tensor_dataset`, which
concatenates every shard into a single CPU tensor before training starts,
:class:`StreamingFeatherDataset` never holds more than a handful of shards in
memory at once. It is the dataset to reach for once ``n_simulations`` is
large enough that the full ``(theta, x)`` corpus no longer fits in RAM.

Design
------
* A one-off, cached index maps each shard file to its row count (row counts
  are read from Arrow IPC batch metadata — see
  :func:`~mach3sbitools.utils.file_utils.count_feather_rows` — so this does
  *not* require decoding ``theta``/``x`` for every file). The index is
  memoised to a small JSON manifest next to the data so repeated runs over
  the same folder don't re-scan it.
* ``__getitem__`` resolves a global row index to ``(file_idx, local_idx)``
  via a cumulative-offset binary search, then serves the row from a bounded
  per-process LRU cache of decoded shards (``cache_size`` shards resident at
  once), loading on a cache miss via the existing
  :func:`~mach3sbitools.utils.file_utils.from_feather`.
* Used with ``DataLoader(num_workers=N, shuffle=True)``, each worker process
  gets its own copy of this dataset (and therefore its own LRU cache), so
  peak resident shards is roughly ``N * cache_size`` — size accordingly.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mach3sbitools.simulator import Prior
from mach3sbitools.utils import count_feather_rows, from_feather, get_logger

logger = get_logger()

_MANIFEST_NAME = ".mach3sbi_row_index.json"


class _LRUShardCache:
    """Bounded LRU cache of decoded ``(theta, x)`` shard arrays."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("cache_size must be >= 1")
        self.capacity = capacity
        self._store: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, key: int) -> tuple[np.ndarray, np.ndarray] | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: int, value: tuple[np.ndarray, np.ndarray]) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)


def _build_row_index(
    files: list[Path], manifest_path: Path | None, rebuild: bool = False
) -> list[int]:
    """
    Return the row count of each file, using a cached manifest when valid.

    The manifest keys each file by name and stores its ``(mtime_ns, size,
    row_count)`` — if either mtime or size has changed since the manifest
    was written, that file's row count is recomputed. Manifest writes are
    best-effort: a read-only data directory simply means the index is
    rebuilt (cheaply — metadata only) on every run instead of being cached.
    """
    manifest: dict[str, list[int]] = {}
    if manifest_path is not None and manifest_path.exists() and not rebuild:
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            manifest = {}

    row_counts: list[int] = []
    updated = False
    for f in files:
        stat = f.stat()
        key = f.name
        cached = manifest.get(key)
        if (
            not rebuild
            and cached is not None
            and cached[0] == stat.st_mtime_ns
            and cached[1] == stat.st_size
        ):
            row_counts.append(cached[2])
            continue

        n_rows = count_feather_rows(f)
        row_counts.append(n_rows)
        manifest[key] = [stat.st_mtime_ns, stat.st_size, n_rows]
        updated = True

    if manifest_path is not None and updated:
        try:
            manifest_path.write_text(json.dumps(manifest))
        except OSError:
            logger.debug(f"Could not write row-index manifest to {manifest_path}")

    return row_counts


class StreamingFeatherDataset(Dataset):
    """
    Row-level lazy :class:`~torch.utils.data.Dataset` over a folder of
    ``.feather`` shards, backed by a bounded per-process LRU shard cache.

    Use this in place of ``TrainingDataset(...).to_tensor_dataset()`` once
    the full simulation corpus no longer fits comfortably in RAM. Global
    shuffling still works as normal via ``DataLoader(shuffle=True)`` /
    Lightning's ``DistributedSampler`` under DDP — both only need
    ``__len__``/``__getitem__``, not the data itself, to build their index.

    :param data_folder: Directory containing ``.feather`` shards.
    :param prior: Prior providing the nuisance-parameter filter for theta.
    :param cache_size: Number of decoded shards to keep resident at once,
        per DataLoader worker process. Increase for smoother throughput at
        the cost of more RAM; ``cache_size=1`` is the minimum-memory option
        and works best when combined with several DataLoader workers so
        I/O for the *next* shard overlaps with training on the current one.
    :param manifest_path: Where to cache the per-shard row-count index.
        Defaults to a hidden file inside *data_folder*. Pass ``None`` to
        disable caching (row counts get recomputed, cheaply, every run).
    :param rebuild_manifest: Force recomputation of the row-count index
        even if a valid cached manifest is found.
    """

    def __init__(
        self,
        data_folder: Path,
        prior: Prior,
        cache_size: int = 8,
        manifest_path: Path | None | str = "__default__",
        rebuild_manifest: bool = False,
    ) -> None:
        if not isinstance(data_folder, Path):
            data_folder = Path(data_folder)

        self.data_folder = data_folder
        self.prior = prior
        self.files = sorted(data_folder.glob("*.feather"))

        if not self.files:
            raise FileNotFoundError(f"No .feather shards found in {data_folder}")

        if manifest_path == "__default__":
            manifest_path = data_folder / _MANIFEST_NAME

        if manifest_path is not None:
            manifest_path = Path(manifest_path)

        t0 = time.time()
        self.row_counts = _build_row_index(
            self.files, manifest_path, rebuild=rebuild_manifest
        )
        self.offsets = np.cumsum([0, *self.row_counts])
        self._n_rows = int(self.offsets[-1])

        self._cache = _LRUShardCache(cache_size)

        logger.info(
            f"StreamingFeatherDataset: {self._n_rows:,} simulations across "
            f"{len(self.files)} shards (indexed in {time.time() - t0:.2f}s) | "
            f"cache holds {cache_size} shard(s)/worker"
        )

    def __len__(self) -> int:
        """Total number of simulations across all shards."""
        return self._n_rows

    def _load_shard(self, file_idx: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get(file_idx)
        if cached is not None:
            return cached

        theta, x = from_feather(self.files[file_idx], self.prior.nuisance_filter)
        self._cache.put(file_idx, (theta, x))
        return theta, x

    def _resolve(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self._n_rows
        if not 0 <= idx < self._n_rows:
            raise IndexError(idx)

        file_idx = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        local_idx = idx - int(self.offsets[file_idx])
        return file_idx, local_idx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Resolve a global row index to a shard and return that ``(theta, x)`` row.

        :param idx: Global simulation index in ``[0, len(self))``.
        :returns: ``(theta, x)`` float tensors for a single simulation.
        """
        file_idx, local_idx = self._resolve(idx)
        theta, x = self._load_shard(file_idx)
        return torch.from_numpy(theta[local_idx]), torch.from_numpy(x[local_idx])

    def sample_rows(self, n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Draw a random subsample of rows directly from disk.

        Used for fitting compressors (e.g. PCA) without first materialising
        the full corpus: only enough shards to cover *n* rows are ever
        decoded, regardless of total dataset size.

        :param n: Number of rows to sample (capped at dataset size).
        :param seed: RNG seed for reproducible sampling.
        :returns: ``(theta, x)`` tensors of shape ``(min(n, len(self)), ...)``.
        """
        n = min(n, self._n_rows)
        rng = np.random.default_rng(seed)
        idx = rng.choice(self._n_rows, size=n, replace=False)
        idx.sort()  # cluster reads per shard rather than random-seeking

        thetas, xs = [], []
        for i in idx:
            file_idx, local_idx = self._resolve(int(i))
            theta, x = self._load_shard(file_idx)
            thetas.append(theta[local_idx])
            xs.append(x[local_idx])

        return torch.from_numpy(np.stack(thetas)), torch.from_numpy(np.stack(xs))


class CompressedDatasetWrapper(Dataset):
    """
    Apply fitted ``theta``/``x`` compressors to an underlying dataset lazily,
    per item, instead of transforming and re-materialising the whole corpus.

    This is the streaming-dataset analogue of
    :meth:`~mach3sbitools.inference.InferenceHandler._apply_compression`'s
    in-memory path (which rebuilds a compressed ``TensorDataset`` up front):
    since the compressor transform is just a mean-subtract + matmul, doing
    it on each batch as it's produced is cheap and avoids ever holding a
    compressed copy of the full dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        theta_compressor=None,
        x_compressor=None,
    ) -> None:
        self.dataset = dataset
        self.theta_compressor = theta_compressor
        self.x_compressor = x_compressor

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        theta, x = self.dataset[idx]
        if self.theta_compressor is not None:
            theta = self.theta_compressor.transform(theta)
        if self.x_compressor is not None:
            x = self.x_compressor.transform(x)
        return theta, x
