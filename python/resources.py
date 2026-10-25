"""Keeping training to a share of the machine: a set of CPU cores and a share of the GPU's time.

    cpus = limit_cpus(4)          # this process and its threads run on 4 physical cores only
    gpu = GpuShare(0.5, device)   # GPU work, timed in `with gpu.section():`, is followed by
                                  # enough idle time to keep the GPU busy at most half the time

Child processes (such as the Rust arena) inherit the CPU affinity.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import torch


def _linux_cores() -> list[list[int]]:
    """Logical CPUs grouped by physical core, in core order (Linux, from sysfs)."""
    groups: dict[tuple[int, int], list[int]] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = (int((topo / "physical_package_id").read_text()), int((topo / "core_id").read_text()))
        except (OSError, ValueError):
            key = (0, cpu)
        groups.setdefault(key, []).append(cpu)
    return [groups[k] for k in sorted(groups)]


def limit_cpus(cores: int) -> list[int]:
    """Pins this process (and the threads and children it starts) to `cores` physical cores,
    both hardware threads of each, and caps PyTorch's and the math libraries' thread pools to
    `cores`. Returns the logical CPUs used, or [] where affinity can't be set (then only the
    thread counts limit it)."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(cores)
    torch.set_num_threads(cores)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set; harmless
    cpus: list[int] = []
    if hasattr(os, "sched_setaffinity"):
        groups = _linux_cores()
        cpus = [c for g in groups[:cores] for c in g]
        os.sched_setaffinity(0, cpus)
    else:
        try:
            import psutil
        except ImportError:
            warnings.warn("install psutil to pin training to the chosen cores (pip install psutil); "
                          "until then only thread counts limit it", stacklevel=2)
            return []
        logical = psutil.cpu_count(logical=True) or 1
        physical = psutil.cpu_count(logical=False) or logical
        per_core = max(1, logical // physical)
        # Windows numbers both hardware threads of a core next to each other
        cpus = list(range(min(logical, cores * per_core)))
        try:
            psutil.Process().cpu_affinity(cpus)
        except (AttributeError, psutil.Error):  # macOS has no affinity
            warnings.warn(f"can't set CPU affinity on {sys.platform}; only thread counts limit it",
                          stacklevel=2)
            return []
    return cpus


class GpuShare:
    """Holds GPU use to about `share` of the time, measured over short windows: each GPU section
    is timed (synchronizing at its end, so queued work counts) and followed by enough idle time
    that busy / elapsed stays at or below `share`. The timing includes the CPU's part of each
    section, so the GPU's own utilization stays a little under the target. Also caps the
    process's GPU memory at the same share."""

    WINDOW = 2.0  # seconds; nvidia-smi reports utilization over about a second

    def __init__(self, share: float, device: torch.device):
        self.share = share
        self.enabled = device.type == "cuda" and share < 1.0
        if self.enabled:
            torch.cuda.set_per_process_memory_fraction(share, device)
        self.start = time.monotonic()
        self.busy = 0.0

    @contextmanager
    def section(self):
        if not self.enabled:
            yield
            return
        t = time.monotonic()
        yield
        torch.cuda.synchronize()
        now = time.monotonic()
        self.busy += now - t
        idle = self.busy / self.share - (now - self.start)
        if idle > 0:
            time.sleep(idle)
        if time.monotonic() - self.start > self.WINDOW:
            self.start, self.busy = time.monotonic(), 0.0
