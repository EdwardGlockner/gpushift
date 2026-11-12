"""GPU detection via nvidia-smi.

``GPUSHIFT_FAKE_GPUS`` (comma-separated memory sizes in MiB, e.g. ``8192,6144``;
empty or ``0`` for none) replaces detection. Tests and CI use it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass

from ._proc import NO_WINDOW


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

    @property
    def free_perc(self) -> float:
        if not self.mem_total:
            return 0.0
        return 100.0 * (self.mem_total - self.mem_used) / self.mem_total

    def to_dict(self) -> dict:
        return asdict(self)


class GPUQueryError(RuntimeError):
    pass


def nvidia_smi_path() -> str | None:
    p = shutil.which("nvidia-smi")
    if p:
        return p
    if os.name == "nt":
        for c in (
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ):
            if os.path.exists(c):
                return c
    return None


def _fake() -> list[GPU] | None:
    fake = os.environ.get("GPUSHIFT_FAKE_GPUS")
    if fake is None:
        return None
    fake = fake.strip()
    if fake in ("", "0"):
        return []
    return [GPU(i, f"Fake GPU {i}", int(m), 0) for i, m in enumerate(fake.split(","))]


def query() -> list[GPU]:
    """Return all GPUs. ``[]`` if there is no NVIDIA driver.

    Raises GPUQueryError if nvidia-smi exists but fails."""
    fake = _fake()
    if fake is not None:
        return fake
    exe = nvidia_smi_path()
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise GPUQueryError(f"nvidia-smi failed: {e}") from e
    if out.returncode != 0:
        raise GPUQueryError(f"nvidia-smi failed: {(out.stderr or out.stdout).strip()}")
    return [g for g in map(parse_line, out.stdout.strip().splitlines()) if g is not None]


QUERY = "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit,fan.speed,clocks.sm"


def _num(s: str, kind=int):
    """A number from nvidia-smi, or None for "[N/A]", "[Not Supported]" etc."""
    try:
        return kind(float(s))
    except ValueError:
        return None


def parse_line(line: str) -> GPU | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return None
    try:
        g = GPU(int(parts[0]), parts[1], int(float(parts[2])), int(float(parts[3])))
    except ValueError:
        return None
    extra = parts[4:10] + [""] * (6 - len(parts[4:10]))
    g.util, g.temp, g.fan, g.clock = _num(extra[0]), _num(extra[1]), _num(extra[4]), _num(extra[5])
    g.power, g.power_limit = _num(extra[2], float), _num(extra[3], float)
    return g


def detect() -> list[GPU]:
    """Like query(), but returns [] instead of raising."""
    try:
        return query()
    except GPUQueryError:
        return []
