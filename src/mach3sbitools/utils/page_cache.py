"""
Page-cache control for large streaming reads.

Both the merge/strip tools and the training dataloader walk files far larger
than the job's memory allowance. Left alone the kernel caches every page it
serves, and under a cgroup memory limit -- which is what SLURM gives a job --
that cache counts against the limit. Once it fills, the kernel spends its time
evicting and re-faulting rather than doing useful work, and the job either
crawls or gets OOM-killed, all while process RSS looks perfectly healthy.

These helpers let a caller say "I am done with this byte range" so the
footprint stays flat. Everything no-ops where posix_fadvise is unavailable
(macOS), so callers do not need to branch.
"""

from __future__ import annotations

import os

#: True when page-cache hints are actually available on this platform.
CAN_DROP_CACHE = hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")


def advise_sequential(fd: int) -> None:
    """Hint that *fd* is read front-to-back, so the kernel reads ahead."""
    fadvise = getattr(os, "posix_fadvise", None)
    flag = getattr(os, "POSIX_FADV_SEQUENTIAL", None)
    if fadvise is None or flag is None:
        return
    try:
        fadvise(fd, 0, 0, flag)
    except OSError:
        pass


def drop_from_cache(fd: int, offset: int, length: int) -> None:
    """Tell the kernel the given byte range of *fd* is no longer needed."""
    fadvise = getattr(os, "posix_fadvise", None)
    flag = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or flag is None:
        return
    try:
        fadvise(fd, offset, length, flag)
    except OSError:
        pass


def cgroup_memory_limit() -> int | None:
    """
    The job's cgroup memory limit in bytes, or ``None`` if unlimited/unknown.

    Checks cgroup v2 then v1. A limit of "max" or an implausibly large value
    means no meaningful limit is set.
    """
    for path in (
        "/sys/fs/cgroup/memory.max",  # v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # v1
    ):
        try:
            with open(path) as f:
                raw = f.read().strip()
        except OSError:
            continue

        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue

        # v1 reports a sentinel near 2**63 when unlimited.
        if value <= 0 or value > 1 << 62:
            return None
        return value

    return None
