"""GPUs from NVIDIA (nvidia.py) or AMD (amd.py). With nvidia-smi present, only
NVIDIA GPUs are used.

``GPUSHIFT_FAKE_GPUS`` (comma-separated memory sizes in MiB, e.g. ``8192,6144``;
empty or ``0`` for none) replaces detection; ``GPUSHIFT_FAKE_GPU_VENDOR`` sets
their vendor. ``GPUSHIFT_FAKE_GPU_PROCS=<file.json>`` ({"pid": MiB, ...})
replaces per-process memory. Tests and CI use them.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass


@dataclass
class GPU:
    index: int
    name: str
    mem_total: int  # MiB
    mem_used: int  # MiB
    util: int | None = None  # % of time the GPU was busy
    temp: int | None = None  # degrees C
    power: float | None = None  # W drawn
    power_limit: float | None = None  # W
    fan: int | None = None  # %
    clock: int | None = None  # SM clock, MHz
    vendor: str = "nvidia"  # nvidia | amd

    @property
    def free_perc(self) -> float:
        if not self.mem_total:
            return 0.0
        return 100.0 * (self.mem_total - self.mem_used) / self.mem_total

    def to_dict(self) -> dict:
        return asdict(self)


class GPUQueryError(RuntimeError):
    pass


def backend(vendor: str):
    from . import amd, nvidia

    return amd if vendor == "amd" else nvidia


def _fake() -> list[GPU] | None:
    fake = os.environ.get("GPUSHIFT_FAKE_GPUS")
    if fake is None:
        return None
    fake = fake.strip()
    if fake in ("", "0"):
        return []
    vendor = os.environ.get("GPUSHIFT_FAKE_GPU_VENDOR", "nvidia")
    return [GPU(i, f"Fake GPU {i}", int(m), 0, vendor=vendor) for i, m in enumerate(fake.split(","))]


def query() -> list[GPU]:
    """All GPUs, ``[]`` if there is no GPU driver.

    Raises GPUQueryError if nvidia-smi exists but fails."""
    fake = _fake()
    if fake is not None:
        return fake
    from . import amd, nvidia

    return nvidia.query() if nvidia.available() else amd.query()


def detect() -> list[GPU]:
    """Like query(), but returns [] instead of raising."""
    try:
        return query()
    except GPUQueryError:
        return []


def per_process(pids=None) -> dict[int, int] | None:
    """pid -> MiB of GPU memory. None if unknown."""
    fake = os.environ.get("GPUSHIFT_FAKE_GPU_PROCS")
    if fake is not None:
        try:
            with open(fake, encoding="utf-8") as f:
                return {int(k): int(v) for k, v in json.load(f).items()}
        except (OSError, ValueError):
            return {}
    try:
        if os.name == "nt":  # nvidia-smi can't report per-process memory under WDDM
            from . import windows

            return windows.process_memory()
        from . import amd, nvidia

        return nvidia.per_process() if nvidia.available() else amd.per_process(pids)
    except Exception:
        return None


def visible_env(env: dict[str, str], gpus: list[int], vendor: str) -> None:
    backend(vendor).visible_env(env, gpus)
