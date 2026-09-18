"""
Strip nuisance columns out of an already-merged theta.npy.

Does the same thing ``merge_shards --prior_path`` does, but against merged
output rather than the original feather shards -- so you don't pay to re-read
the whole shard set just to drop columns.

Because .npy is row-major and a 4 KiB page spans several complete rows,
masking theta at read time saves no I/O at all: the discarded columns are
faulted in regardless. Narrowing the file on disk is the only way to stop
paying for them, on every epoch of every subsequent run.

x.npy is not touched -- it has no nuisance columns. In the default
(non-in-place) mode it is symlinked into the output directory rather than
copied, so this costs roughly ``n_rows x n_kept x 4`` bytes of new disk and
nothing more.

Both the read and the write are sequential, so expect this to run at
whatever streaming bandwidth the underlying storage gives you.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from mach3sbitools.simulator import load_prior
from mach3sbitools.utils import (
    CAN_DROP_CACHE,
    advise_sequential,
    drop_from_cache,
    get_logger,
)

from .merge_shards import METADATA_FILENAME

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

#: Rows per streamed chunk. At 299 float32 columns this is ~310 MB per read.
DEFAULT_CHUNK_ROWS = 262_144

#: Rows spot-checked against the source after the rewrite.
_VERIFY_ROWS = 16


def _memory_snapshot() -> str:
    """
    Process RSS and cgroup usage, for the progress line.

    Worth surfacing because the two disagree in exactly the way that matters
    here: a streaming copy keeps RSS flat while page cache climbs, and it is
    the cgroup number -- which counts that cache -- that gets a SLURM job
    killed. Returns "" off Linux, where neither file exists.
    """
    parts = []
    try:
        with open("/proc/self/statm") as f:
            rss_pages = int(f.read().split()[1])
        parts.append(f"rss {rss_pages * os.sysconf('SC_PAGE_SIZE') / 1e9:.1f}G")
    except (OSError, IndexError, ValueError):
        pass

    for cgroup_file in (
        "/sys/fs/cgroup/memory.current",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",  # cgroup v1
    ):
        try:
            with open(cgroup_file) as f:
                parts.append(f"cgroup {int(f.read().strip()) / 1e9:.1f}G")
            break
        except (OSError, ValueError):
            continue

    return " ".join(parts)


#: True when we can actually manage the page cache; see _drop_from_cache.
_CAN_DROP_CACHE = CAN_DROP_CACHE


def _verify(source: np.ndarray, dest_path: Path, keep: np.ndarray, rng) -> None:
    """Spot-check that randomly chosen output rows match the masked source."""
    dest = np.load(dest_path, mmap_mode="r")
    n = min(_VERIFY_ROWS, len(source))
    idx = rng.choice(len(source), size=n, replace=False)

    for i in idx:
        if not np.array_equal(dest[i], source[i][keep]):
            raise RuntimeError(
                f"Verification failed at row {i}: {dest_path} does not match "
                f"the masked source. Output left in place for inspection."
            )

    del dest
    get_logger().info(f"Verified {n} randomly chosen rows against the source")


def strip_theta_module(
    data_dir: Path,
    prior_path: Path,
    output_dir: Path | None = None,
    in_place: bool = False,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    seed: int = 42,
):
    """
    Rewrite ``data_dir/theta.npy`` keeping only the columns this prior's
    nuisance filter selects.

    :param data_dir: Directory holding ``theta.npy`` and ``x.npy``.
    :param prior_path: Prior whose nuisance filter selects the kept columns.
    :param output_dir: Destination directory. Gets the narrowed ``theta.npy``,
        a symlink to the original ``x.npy``, and a metadata sidecar. Ignored
        when *in_place* is set.
    :param in_place: Replace ``data_dir/theta.npy`` instead, via a temporary
        file and an atomic rename. Destroys the unfiltered theta -- you would
        have to re-merge from the shards to get it back.
    :param chunk_rows: Rows per streamed chunk.
    :param seed: Seed for choosing verification rows.
    """
    logger = get_logger()

    data_dir = Path(data_dir)
    theta_path = data_dir / "theta.npy"
    x_path = data_dir / "x.npy"

    if not theta_path.is_file():
        raise FileNotFoundError(f"Cannot find {theta_path}")
    if not x_path.is_file():
        raise FileNotFoundError(f"Cannot find {x_path}")

    if not in_place:
        if output_dir is None:
            raise ValueError("Provide --output_dir, or pass --in_place.")
        output_dir = Path(output_dir)
        if output_dir.resolve() == data_dir.resolve():
            raise ValueError(
                "output_dir is the same as data_dir; use --in_place if that's "
                "what you meant."
            )

    # ── Work out the keep-mask ────────────────────────────────────────────
    prior = load_prior(prior_path)
    keep = prior.nuisance_filter.cpu().numpy().astype(bool)
    kept_names = list(prior.prior_data.parameter_names)

    theta = np.load(theta_path, mmap_mode="r")
    n_rows, n_cols = theta.shape
    n_keep = int(keep.sum())

    if n_cols == n_keep and len(keep) != n_cols:
        logger.info(
            f"{theta_path} already has {n_cols} columns, matching this prior's "
            f"filter. Nothing to do."
        )
        return

    if len(keep) != n_cols:
        raise ValueError(
            f"Prior nuisance filter covers {len(keep)} parameters but "
            f"{theta_path} has {n_cols} columns. Wrong prior for this data?"
        )

    if n_keep == n_cols:
        logger.info("Prior keeps every theta column -- nothing to strip.")
        return

    old_gb = theta.nbytes / 1e9
    new_gb = n_rows * n_keep * 4 / 1e9
    x_gb = np.load(x_path, mmap_mode="r").nbytes / 1e9
    logger.info(
        "Stripping theta: [bold]%d[/] -> [bold]%d[/] columns (%.1f GB -> %.1f GB)",
        n_cols,
        n_keep,
        old_gb,
        new_gb,
    )
    logger.info(
        "Per-epoch bytes: %.1f GB -> %.1f GB (%.0f%% of current)",
        x_gb + old_gb,
        x_gb + new_gb,
        100.0 * (x_gb + new_gb) / (x_gb + old_gb),
    )

    # ── Stream the rewrite ────────────────────────────────────────────────
    if in_place:
        dest_path = theta_path.with_suffix(".npy.tmp")
    else:
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        dest_path = output_dir / "theta.npy"
        if dest_path.exists():
            raise FileExistsError(
                f"{dest_path} already exists please rename or use another output dir"
            )

    # Everything below deliberately avoids mmap for the bulk copy. Both files
    # are touched exactly once, front to back, which is the case plain
    # buffered read/write handles best: one large sequential syscall per
    # chunk. An mmap'd write instead faults page by page, and on a network
    # filesystem each fault can become its own round trip -- which is how a
    # streaming rewrite ends up running at a fraction of disk bandwidth.
    src_dtype = theta.dtype
    src_offset = theta.offset  # where the .npy header ends and data begins
    del theta  # release the source mmap; we reopen it as a plain file below

    # open_memmap writes a correct .npy header and preallocates the file; we
    # then close it and write the body through a plain file handle.
    dst_mm = np.lib.format.open_memmap(
        dest_path, mode="w+", dtype=src_dtype, shape=(n_rows, n_keep)
    )
    dst_offset = dst_mm.offset
    del dst_mm

    cols = np.flatnonzero(keep)
    row_bytes = n_cols * src_dtype.itemsize

    # Allocated once and reused. The previous version materialised a fresh
    # chunk-sized array on every iteration.
    src_buf = np.empty((chunk_rows, n_cols), dtype=src_dtype)
    src_bytes = src_buf.reshape(-1).view(np.uint8)

    if _CAN_DROP_CACHE:
        logger.info(
            "Dropping page cache behind the copy; expected RSS ~%.1f GB",
            chunk_rows * n_cols * src_dtype.itemsize / 1e9,
        )
    else:
        logger.warning(
            "posix_fadvise unavailable on this platform: page cache will grow "
            "as the copy proceeds. Harmless with free RAM, but under a cgroup "
            "memory limit it can get the job killed."
        )

    t_start = time.perf_counter()
    bytes_read = 0

    with open(theta_path, "rb") as f_src, open(dest_path, "r+b") as f_dst:
        advise_sequential(f_src.fileno())
        f_src.seek(src_offset)
        f_dst.seek(dst_offset)

        src_pos = src_offset
        dst_pos = dst_offset
        done = 0
        pbar = tqdm(total=n_rows, desc=f"Stripping {theta_path.name}", unit="row")
        while done < n_rows:
            rows = min(chunk_rows, n_rows - done)
            want = rows * row_bytes

            got = f_src.readinto(src_bytes[:want].data)
            if got != want:
                raise OSError(
                    f"Short read from {theta_path} at row {done}: "
                    f"wanted {want} bytes, got {got}"
                )

            # np.take returns a fresh C-contiguous block, so tofile writes it
            # straight out with no further copy.
            out_block = np.take(src_buf[:rows], cols, axis=1)
            out_block.tofile(f_dst)
            out_bytes = out_block.nbytes

            if _CAN_DROP_CACHE:
                # Source pages are spent as soon as they are copied.
                drop_from_cache(f_src.fileno(), src_pos, want)
                # Written pages can only be dropped once they are clean, so
                # force this chunk out before discarding it. This also bounds
                # dirty memory, which is the part the kernel cannot reclaim
                # under pressure.
                f_dst.flush()
                os.fsync(f_dst.fileno())
                drop_from_cache(f_dst.fileno(), dst_pos, out_bytes)

            src_pos += want
            dst_pos += out_bytes
            done += rows
            bytes_read += want
            elapsed = time.perf_counter() - t_start
            pbar.set_postfix_str(
                f"{bytes_read / elapsed / 1e6:.0f} MB/s  {_memory_snapshot()}"
            )
            pbar.update(rows)
        pbar.close()

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Copied %.1f GB in %.0f s (%.0f MB/s read, %.0f MB/s written)",
        bytes_read / 1e9,
        elapsed,
        bytes_read / max(elapsed, 1e-9) / 1e6,
        n_rows * n_keep * src_dtype.itemsize / max(elapsed, 1e-9) / 1e6,
    )

    theta = np.load(theta_path, mmap_mode="r")  # reopened only for _verify

    _verify(theta, dest_path, keep, np.random.default_rng(seed))
    del theta  # release the source mmap before any rename

    if in_place:
        dest_path.replace(theta_path)
        final_theta = theta_path
        final_dir = data_dir
        logger.info(f"Replaced [cyan]{theta_path}[/] in place")
    else:
        assert output_dir is not None
        final_theta = dest_path
        final_dir = output_dir
        link = output_dir / "x.npy"
        if not link.exists():
            try:
                os.symlink(x_path.resolve(), link)
                logger.info(f"Symlinked [cyan]{link}[/] -> {x_path.resolve()}")
            except OSError as exc:
                logger.warning(
                    f"Could not symlink x.npy ({exc}); point training at "
                    f"{x_path} yourself, or copy it in."
                )

    # ── Sidecar ───────────────────────────────────────────────────────────
    metadata_path = final_dir / METADATA_FILENAME
    metadata = {}
    source_metadata = data_dir / METADATA_FILENAME
    if source_metadata.is_file():
        try:
            metadata = json.loads(source_metadata.read_text())
        except (OSError, json.JSONDecodeError):
            metadata = {}

    metadata.update(
        {
            "n_rows": int(n_rows),
            "theta_dim": int(n_keep),
            "theta_filtered": True,
            "n_theta_params_full": int(n_cols),
            "kept_parameter_names": kept_names,
            "prior_path": str(prior_path),
            "stripped_from": str(theta_path),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Wrote merge metadata to [cyan]{metadata_path}[/]")
    logger.info(f"Done. Train against [cyan]{final_theta.parent}[/]")
