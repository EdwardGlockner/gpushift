"""How much GPU memory each process uses.

* Windows: the "GPU Process Memory" performance counters (what Task Manager
  shows); nvidia-smi can't report per-process memory under WDDM.
* Linux: ``nvidia-smi -q -x``, which lists compute and graphics processes, or
  amdgpu.py without nvidia-smi.
* ``GPUSHIFT_FAKE_GPU_PROCS=<file.json>`` ({"pid": MiB, ...}) replaces both
  (tests use it).

Used to learn how much memory jobs need, so jobs with a known need can share a GPU.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from ._proc import NO_WINDOW


def per_process(pids=None) -> dict[int, int] | None:
    """pid -> MiB of dedicated GPU memory, summed over GPUs. None if this
    machine can't tell. ``pids`` limits the AMD reading."""
    fake = os.environ.get("GPUSHIFT_FAKE_GPU_PROCS")
    if fake is not None:
        try:
            with open(fake, encoding="utf-8") as f:
                return {int(k): int(v) for k, v in json.load(f).items()}
        except (OSError, ValueError):
            return {}
    try:
        if os.name == "nt":
            return _windows()
        from .gpus import nvidia_smi_path

        if not nvidia_smi_path() and sys.platform.startswith("linux"):
            from . import amdgpu

            return amdgpu.per_process(pids) if amdgpu.devices() else None
        return _nvidia_smi()
    except Exception:
        return None


def _nvidia_smi() -> dict[int, int] | None:
    from .gpus import nvidia_smi_path

    exe = nvidia_smi_path()
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


# ---------------------------------------------------------------- Windows (PDH)

_query = None


def _windows() -> dict[int, int] | None:
    import ctypes
    from ctypes import wintypes

    global _query
    pdh = ctypes.WinDLL("pdh")

    class Fmt(ctypes.Structure):
        _fields_ = [("status", wintypes.DWORD), ("value", ctypes.c_longlong)]

    class Item(ctypes.Structure):
        _fields_ = [("name", wintypes.LPWSTR), ("fmt", Fmt)]

    if _query is None:
        q, c = wintypes.HANDLE(), wintypes.HANDLE()
        if pdh.PdhOpenQueryW(None, None, ctypes.byref(q)) != 0:
            return None
        path = r"\GPU Process Memory(*)\Dedicated Usage"
        if pdh.PdhAddEnglishCounterW(q, path, None, ctypes.byref(c)) != 0:
            pdh.PdhCloseQuery(q)
            return None
        _query = (q, c)
    q, c = _query
    if pdh.PdhCollectQueryData(q) != 0:
        return None
    size, count = wintypes.DWORD(0), wintypes.DWORD(0)
    pdh.PdhGetFormattedCounterArrayW(c, 0x400, ctypes.byref(size), ctypes.byref(count), None)  # PDH_FMT_LARGE
    if not size.value:
        return {}
    buf = (ctypes.c_byte * size.value)()
    if pdh.PdhGetFormattedCounterArrayW(c, 0x400, ctypes.byref(size), ctypes.byref(count), buf) != 0:
        return None
    items = ctypes.cast(buf, ctypes.POINTER(Item))
    used: dict[int, int] = {}
    for i in range(count.value):
        m = re.match(r"pid_(\d+)_", items[i].name or "")
        mib = items[i].fmt.value // (1024 * 1024)
        if m and mib > 0:
            pid = int(m.group(1))
            used[pid] = used.get(pid, 0) + mib
    return used
