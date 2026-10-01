"""
Merge feather shards into memmap-backed .npy files (theta.npy, x.npy).

Same two-pass structure as the HDF5 version (count rows, then peek dims,
then stream-copy), but writes straight to np.lib.format.open_memmap instead
of h5py. Output is two files in output_dir: theta.npy and x.npy, directly
usable with np.load(path, mmap_mode="r") / MemmapPairDataset - no separate
conversion step needed.

Two optional transforms are applied here, at merge time, because both are
one-off costs that buy back time on *every* epoch of *every* subsequent
training run:

Nuisance filtering (``prior_path``)
    ``TrainingDataset`` discards non-kept theta columns after reading them.
    Because .npy is row-major and a 4 KiB page spans several complete rows,
    touching any column of a row faults in *all* of its columns - so masking
    at read time saves no I/O whatsoever. Dropping the columns here is the
    only way to actually stop paying for them. Halves per-epoch bytes for a
    typical nuisance filter.

Shuffling (``shuffle``)
    The training DataLoader deliberately runs with ``shuffle=False`` so that
    reads stay sequential, and the train/val split is contiguous. That means
    shard order becomes epoch order and the validation set is literally the
    tail of the file. Shuffling once here fixes both without costing
    anything at train time.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from mach3sbitools.simulator import load_prior
from mach3sbitools.utils import from_feather, get_logger, peek_num_rows

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

# Set to np.float32 to downcast during merge and roughly halve output size
# vs the source float64 feather data. Set to None to keep source dtype.
FORCE_DTYPE = None

#: Sidecar written next to theta.npy / x.npy recording what was applied.
#: TrainingDataset reads it back to fail loudly on a prior/data mismatch.
METADATA_FILENAME = "merge_metadata.json"


def _truncate_npy(path: Path, new_n_rows: int, chunk_rows: int = 100_000):
    """
    Rewrite a memmap-backed .npy file so it only contains its first
    new_n_rows rows, dropping any trailing unused/uninitialized rows.

    We can't just slice the file in place because the .npy header (which
    encodes the shape) is padded/aligned to 64 bytes, and that padding can
    shift width depending on the digit count of the shape - so we rewrite
    via open_memmap with the correct shape and stream-copy the data across
    in chunks to avoid loading the whole array into memory.
    """
    arr = np.load(path, mmap_mode="r")
    if new_n_rows == arr.shape[0]:
        return

    new_shape = (new_n_rows, *arr.shape[1:])
    tmp_path = path.with_suffix(".npy.tmp")
    new_arr = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=arr.dtype, shape=new_shape
    )
    for start in range(0, new_n_rows, chunk_rows):
        end = min(start + chunk_rows, new_n_rows)
        new_arr[start:end] = arr[start:end]
    new_arr.flush()

    del arr, new_arr  # release mmaps before replacing the file
    tmp_path.replace(path)


class _StreamWriter:
    """
    Streams (theta, x) row blocks into the output memmaps, optionally
    shuffling on the way through.

    With ``buffer_rows == 0`` rows are written straight through in arrival
    order. Otherwise rows accumulate until the buffer holds at least
    ``buffer_rows``, at which point the whole buffer is permuted and flushed.

    This is an *approximate* shuffle: it mixes rows within a sliding window
    rather than across the entire dataset. That is deliberate. A true global
    permutation means random writes scattered over a multi-hundred-GB file,
    which is precisely the access pattern that makes this data slow to read
    in the first place. Combined with a randomised shard order (see
    ``merge_shards_module``), a window spanning many shards mixes more than
    well enough for training.
    """

    def __init__(
        self,
        theta_out: np.ndarray,
        x_out: np.ndarray,
        buffer_rows: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._theta_out = theta_out
        self._x_out = x_out
        self._buffer_rows = buffer_rows
        self._rng = rng
        self._t_buf: list[np.ndarray] = []
        self._x_buf: list[np.ndarray] = []
        self._buffered = 0
        self.offset = 0

    def add(self, theta: np.ndarray, x: np.ndarray) -> None:
        """Queue a block of rows for writing."""
        if self._buffer_rows <= 0:
            self._write(theta, x)
            return

        self._t_buf.append(theta)
        self._x_buf.append(x)
        self._buffered += len(theta)

        if self._buffered >= self._buffer_rows:
            self.flush()

    def flush(self) -> None:
        """Permute and write whatever is currently buffered."""
        if not self._t_buf:
            return

        theta = np.concatenate(self._t_buf)
        x = np.concatenate(self._x_buf)

        assert self._rng is not None
        perm = self._rng.permutation(len(theta))
        self._write(theta[perm], x[perm])

        self._t_buf.clear()
        self._x_buf.clear()
        self._buffered = 0

    def _write(self, theta: np.ndarray, x: np.ndarray) -> None:
        n = len(theta)
        self._theta_out[self.offset : self.offset + n] = theta
        self._x_out[self.offset : self.offset + n] = x
        self.offset += n


def _load_nuisance_filter(prior_path: Path) -> tuple[np.ndarray, list[str]]:
    """
    Read the boolean keep-mask over theta columns out of a saved prior.

    :returns: ``(keep_mask, kept_parameter_names)``. The mask is over *all*
        parameters; ``prior.prior_data`` is already filtered, so its
        ``parameter_names`` are the kept ones.
    """
    prior = load_prior(prior_path)
    keep = prior.nuisance_filter.cpu().numpy().astype(bool)
    kept_names = list(prior.prior_data.parameter_names)
    return keep, kept_names


def merge_shards_module(
    simulation_dir: Path,
    output_dir: Path,
    prior_path: Path | None = None,
    shuffle: bool = False,
    shuffle_buffer_rows: int = 5_000_000,
    seed: int = 42,
):
    """
    Merge a folder of feather shard files into memmap-backed theta.npy / x.npy
    in output_dir.

    :param simulation_dir: Directory of ``*.feather`` shards.
    :param output_dir: Destination for ``theta.npy``, ``x.npy`` and the
        metadata sidecar.
    :param prior_path: If given, drop theta columns excluded by this prior's
        nuisance filter. This is baked into the output, so the merged data is
        tied to this nuisance choice -- changing it later means re-merging.
    :param shuffle: Randomise shard order and permute within a sliding
        window. See :class:`_StreamWriter` for what this does and does not
        guarantee.
    :param shuffle_buffer_rows: Window size, in rows. Larger mixes better and
        costs more RAM: roughly ``rows x (x_dim + theta_dim) x 4`` bytes.
    :param seed: Seed for shard ordering and window permutation.
    """
    logger = get_logger()

    theta_path = output_dir / "theta.npy"
    x_path = output_dir / "x.npy"
    metadata_path = output_dir / METADATA_FILENAME

    for p in (theta_path, x_path):
        if p.exists():
            raise FileExistsError(
                f"{p} already exists please rename or save to another output dir"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sorted, because glob() order is filesystem-dependent: without this the
    # merged row order (and so what --seed reproduces) varies between runs.
    sims_files = sorted(simulation_dir.glob("*feather"))
    if not sims_files:
        raise FileNotFoundError(f"Cannot find any .feather files in {simulation_dir}")

    rng = np.random.default_rng(seed)
    if shuffle:
        # Randomising shard order is what lets a bounded window mix rows
        # from all over the dataset rather than from a few adjacent shards.
        rng.shuffle(sims_files)  # type: ignore[arg-type]

    logger.info(
        "Merging %d files in %s -> %s", len(sims_files), simulation_dir, output_dir
    )

    n_rows = 0
    for shard in tqdm(sims_files, desc="Counting number of rows"):
        n_rows += peek_num_rows(shard)

    # We now peek the first entry
    t_test, x_test = from_feather(sims_files[0])

    # We get the theta and x_dim
    t_dim = len(t_test[0])
    x_dim = len(x_test[0])

    theta_dtype = FORCE_DTYPE or t_test.dtype
    x_dtype = FORCE_DTYPE or x_test.dtype

    del t_test, x_test

    # ── Nuisance filter ───────────────────────────────────────────────────
    keep: np.ndarray | None = None
    kept_names: list[str] | None = None
    if prior_path is not None:
        keep, kept_names = _load_nuisance_filter(prior_path)
        if len(keep) != t_dim:
            raise ValueError(
                f"Prior nuisance filter covers {len(keep)} parameters but the "
                f"shards have {t_dim} theta columns. Wrong prior for this data?"
            )
        logger.info(
            "Nuisance filter: keeping [bold]%d[/]/%d theta columns "
            "(%.1f%% of theta bytes dropped)",
            int(keep.sum()),
            t_dim,
            100.0 * (1.0 - keep.sum() / t_dim),
        )

    t_dim_out = int(keep.sum()) if keep is not None else t_dim

    if shuffle:
        window_gb = shuffle_buffer_rows * (x_dim + t_dim_out) * 4 / 1e9
        logger.info(
            "Shuffling with a %s-row window (~%.1f GB RAM) over randomised shard order",
            f"{shuffle_buffer_rows:,}",
            window_gb,
        )

    # Create the memmap-backed .npy files up front, sized for the full
    # merged dataset (pre-filter) - same role as h5py.File.create_dataset
    # before. Since rows get dropped by the filter below, we may end up
    # writing fewer than n_rows rows; the tail is trimmed off at the end.
    theta_out = np.lib.format.open_memmap(
        theta_path, mode="w+", dtype=theta_dtype, shape=(n_rows, t_dim_out)
    )
    x_out = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=x_dtype, shape=(n_rows, x_dim)
    )

    writer = _StreamWriter(
        theta_out,
        x_out,
        buffer_rows=shuffle_buffer_rows if shuffle else 0,
        rng=rng,
    )

    desc_str = f"Adding sims to {output_dir} | current file: "

    total_filtered = 0

    for shard in (pbar := tqdm(sims_files, desc=desc_str + str(sims_files[0]))):
        pbar.set_description(desc_str + str(shard))

        t, x = from_feather(shard)

        # HACK
        # before = len(t)
        # t_filter = np.where(t[:, -2] > 0)

        # t = t[t_filter]
        # x = x[t_filter]

        # total_filtered += before - len(t)

        if keep is not None:
            t = t[:, keep]

        writer.add(t, x)

    writer.flush()
    offset = writer.offset

    theta_out.flush()
    x_out.flush()
    del theta_out, x_out  # release mmaps so the files can be truncated/replaced

    if offset < n_rows:
        logger.info(
            "Trimming %d unused/uninitialized rows from output arrays",
            n_rows - offset,
        )
        _truncate_npy(theta_path, offset)
        _truncate_npy(x_path, offset)

    metadata = {
        "n_rows": int(offset),
        "x_dim": int(x_dim),
        "theta_dim": int(t_dim_out),
        "theta_filtered": keep is not None,
        "n_theta_params_full": int(t_dim),
        "kept_parameter_names": kept_names,
        "prior_path": str(prior_path) if prior_path is not None else None,
        "shuffled": bool(shuffle),
        "shuffle_seed": int(seed) if shuffle else None,
        "shuffle_buffer_rows": int(shuffle_buffer_rows) if shuffle else None,
        "n_shards": len(sims_files),
        "source_dir": str(simulation_dir),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Wrote merge metadata to [cyan]{metadata_path}[/]")

    logger.info("Finished merge. Filtered out %d/%d entries", total_filtered, n_rows)
