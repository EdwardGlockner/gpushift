"""CPU, memory and storage readings for the dashboard: load per core, RAM,
the programs using the most CPU, and each drive's space and disk traffic. The daemon samples every few seconds; a sample
covers the time since the previous one.

Per-program load comes from the "Process V2" performance counters on Windows
(one cheap query for all processes; psutil would open every process, which
takes over a second on a busy machine) and from psutil elsewhere."""

from __future__ import annotations

import os
import time

import psutil

TOP = 5
MAX_DISKS = 8
SKIP_FS = {"", "squashfs", "tmpfs", "devtmpfs", "overlay", "proc", "sysfs", "autofs"}


class Sampler:
    def __init__(self):
        self.procs: dict[int, psutil.Process] = {}
        self.pdh = _Pdh() if os.name == "nt" else None
        psutil.cpu_percent(percpu=True)  # start the first interval
        self._per_process()
        self.io_last = self._io()

    def sample(self, job_of: dict[int, int] | None = None) -> dict:
        """job_of: pid -> gpushift job id, to label job processes."""
        job_of = job_of or {}
        cores = psutil.cpu_percent(percpu=True)
        n = len(cores) or 1
        vm = psutil.virtual_memory()
        load: dict[str, float] = {}
        for pid, name, pct in self._per_process():
            pct /= n  # share of the whole machine
            if pct <= 0:
                continue
            if pid in job_of:
                name = f"job {job_of[pid]}"
            load[name] = load.get(name, 0.0) + pct
        top = sorted(((k, v) for k, v in load.items() if v >= 0.05), key=lambda kv: -kv[1])[:TOP]
        return {
            "at": time.time(),
            "load_pct": round(sum(cores) / n, 1),
            "cores_pct": cores,
            "mem_used_mb": (vm.total - vm.available) // 2**20,
            "mem_total_mb": vm.total // 2**20,
            "top": [{"name": k, "pct": round(v, 1)} for k, v in top],
            "disks": disks(),
            "io": self._io_rate(),
        }

    def _io(self) -> tuple[float, int, int] | None:
        try:
            c = psutil.disk_io_counters()
        except (OSError, RuntimeError):
            return None
        return None if c is None else (time.time(), c.read_bytes, c.write_bytes)

    def _io_rate(self) -> dict | None:
        """MB/s read and written across all disks since the last sample."""
        now, last = self._io(), self.io_last
        self.io_last = now
        if not now or not last or now[0] <= last[0]:
            return None
        dt = now[0] - last[0]
        return {
            "read_mb_s": round(max(0, now[1] - last[1]) / dt / 2**20, 1),
            "write_mb_s": round(max(0, now[2] - last[2]) / dt / 2**20, 1),
        }

    def _per_process(self) -> list[tuple[int, str, float]]:
        """(pid, name, % of one core) since the last call."""
        if self.pdh is not None:
            rows = self.pdh.read()
            if rows is not None:
                return rows
            self.pdh = None  # counters unavailable: fall back to psutil
        out, seen = [], {}
        for p in psutil.process_iter(["name"]):
            q = self.procs.get(p.pid, p)
            seen[p.pid] = q
            try:
                pct = q.cpu_percent(None)
            except psutil.Error:
                continue
            if p.pid:  # pid 0 is the idle process
                out.append((p.pid, (p.info["name"] or "?").removesuffix(".exe"), pct))
        self.procs = seen
        return out


def disks() -> list[dict]:
    """Space on each local drive (not CD drives, network shares or system mounts)."""
    out, seen = [], set()
    for p in psutil.disk_partitions(all=False):
        if p.fstype.lower() in SKIP_FS or "cdrom" in p.opts or p.device in seen:
            continue
        if p.mountpoint.startswith(("/snap", "/boot", "/var/lib/docker")):
            continue
        try:
            u = psutil.disk_usage(p.mountpoint)
        except OSError:  # card reader without a card, drive went away, ...
            continue
        seen.add(p.device)
        out.append(
            {
                "mount": p.mountpoint,
                "total_mb": u.total // 2**20,
                "used_mb": u.used // 2**20,
                "free_mb": u.free // 2**20,
            }
        )
    return out[:MAX_DISKS]


class _Pdh:
    r"""\Process(*)\% Processor Time, paired with \Process(*)\ID Process by
    instance name ("chrome", "chrome#1", ...)."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self.ct, self.wt = ctypes, wintypes
        self.lib = ctypes.WinDLL("pdh")
        self.q, self.cpu, self.pid = wintypes.HANDLE(), wintypes.HANDLE(), wintypes.HANDLE()
        self.ok = (
            self.lib.PdhOpenQueryW(None, None, ctypes.byref(self.q)) == 0
            and self.lib.PdhAddEnglishCounterW(self.q, r"\Process(*)\% Processor Time", None, ctypes.byref(self.cpu))
            == 0
            and self.lib.PdhAddEnglishCounterW(self.q, r"\Process(*)\ID Process", None, ctypes.byref(self.pid)) == 0
        )
        if self.ok:
            self.lib.PdhCollectQueryData(self.q)

    def _values(self, counter) -> dict[str, float] | None:
        ct, wt = self.ct, self.wt

        class Fmt(ct.Structure):
            _fields_ = (("status", wt.DWORD), ("value", ct.c_double))

        class Item(ct.Structure):
            _fields_ = (("name", wt.LPWSTR), ("fmt", Fmt))

        fmt = 0x200 | 0x8000  # PDH_FMT_DOUBLE | PDH_FMT_NOCAP100 (a process can use several cores)
        size, count = wt.DWORD(0), wt.DWORD(0)
        self.lib.PdhGetFormattedCounterArrayW(counter, fmt, ct.byref(size), ct.byref(count), None)
        if not size.value:
            return {}
        buf = (ct.c_byte * size.value)()
        if self.lib.PdhGetFormattedCounterArrayW(counter, fmt, ct.byref(size), ct.byref(count), buf) != 0:
            return None
        items = ct.cast(buf, ct.POINTER(Item))
        return {items[i].name: items[i].fmt.value for i in range(count.value)}

    def read(self) -> list[tuple[int, str, float]] | None:
        if not self.ok or self.lib.PdhCollectQueryData(self.q) != 0:
            return None
        cpu, pids = self._values(self.cpu), self._values(self.pid)
        if cpu is None or pids is None:
            return None
        out = []
        for inst, pct in cpu.items():
            pid = int(pids.get(inst, 0))
            if pid == 0:  # _Total, Idle
                continue
            out.append((pid, inst.split("#")[0], pct))
        return out
