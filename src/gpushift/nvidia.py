"""NVIDIA GPUs via nvidia-smi."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

from ._proc import NO_WINDOW
from .gpus import GPU, GPUQueryError


def smi_path() -> str | None:
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


def available() -> bool:
    return smi_path() is not None


QUERY = "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit,fan.speed,clocks.sm"


def query() -> list[GPU]:
    """Raises GPUQueryError if nvidia-smi fails."""
    exe = smi_path()
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


def _num(s: str, kind=int):
    """A number, or None for "[N/A]", "[Not Supported]" etc."""
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


def per_process(pids=None) -> dict[int, int] | None:
    exe = smi_path()
    if not exe:
        return None
    out = subprocess.run([exe, "-q", "-x"], capture_output=True, text=True, timeout=15, creationflags=NO_WINDOW)
    if out.returncode != 0:
        return None
    used: dict[int, int] = {}
    for p in ET.fromstring(out.stdout).iter("process_info"):
        pid, mem = p.findtext("pid"), p.findtext("used_memory") or ""
        m = re.match(r"(\d+)\s*MiB", mem)
        if pid and m:
            used[int(pid)] = used.get(int(pid), 0) + int(m.group(1))
    return used


def visible_env(env: dict[str, str], gpus: list[int]) -> None:
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # match nvidia-smi's numbering
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
