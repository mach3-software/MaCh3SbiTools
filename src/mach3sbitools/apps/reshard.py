"""Reshard application module."""

from pathlib import Path

from mach3sbitools.utils import get_logger, iter_feather_chunks, to_feather


def reshard_module(
    input_path: Path,
    output_folder: Path,
    rows_per_shard: int,
    prefix: str,
) -> None:
    """Split one or more feather files into many smaller, fixed-size shards.

    Reads each input file in ``rows_per_shard``-sized chunks via a
    memory-mapped, zero-copy slice — the whole file is never materialised in
    RAM at once, so this works even on a single monolithic file containing
    every simulation (e.g. one that a pre-existing ``simulate`` run wrote in
    one call). This is the tool to reach for before switching training to
    ``--streaming``, which needs many reasonably sized shards to bound its
    per-worker memory footprint against, rather than one huge file.

    Output shards are written in the exact same format produced by
    ``mach3sbi simulate`` / :func:`~mach3sbitools.utils.to_feather`, so
    nothing downstream (``TrainingDataset``, ``StreamingFeatherDataset``)
    needs to know a reshard ever happened.

    .. note::

        Chunking is done per input file, not merged across files — an
        input file smaller than ``rows_per_shard`` is written out whole as
        one (smaller) shard rather than combined with the next file. This
        keeps the memory footprint bounded and predictable; if you have
        many small files you'd rather merge upward, glob them into one
        pass with a script, or open an issue if that's a common enough need.

    Example::

        mach3sbi reshard -i sims.feather -o sims_sharded/ --rows_per_shard 100000

        # Reshard every .feather file already in a folder (e.g. to make
        # a handful of oversized shards more DataLoader-worker friendly):
        mach3sbi reshard -i sims_folder/ -o sims_sharded/ --rows_per_shard 100000
    """
    logger = get_logger()

    input_path = Path(input_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        input_files = sorted(input_path.glob("*.feather"))
    else:
        input_files = [input_path]

    if not input_files:
        raise FileNotFoundError(f"No .feather files found at {input_path}")

    logger.info(
        f"Resharding [bold]{len(input_files)}[/] file(s) into "
        f"~{rows_per_shard:,}-row shards -> [cyan]{output_folder}[/]"
    )

    shard_idx = 0
    total_rows = 0
    for f in input_files:
        for theta, x in iter_feather_chunks(f, rows_per_shard):
            out_path = output_folder / f"{prefix}_{shard_idx:06d}.feather"
            to_feather(out_path, theta, x)
            total_rows += theta.shape[0]
            shard_idx += 1

    logger.info(
        f"Wrote [bold]{shard_idx}[/] shard(s), [bold]{total_rows:,}[/] "
        f"simulations total, to [cyan]{output_folder}[/]"
    )
