"""Windows GPU readings for any vendor, via DXGI and performance counters."""

from __future__ import annotations

import re
from typing import NamedTuple

MiB = 2**20


class Adapter(NamedTuple):
    name: str
    vendor: int
    mem_total: int  # MiB
    luid: int


def adapters() -> list[Adapter]:
    import ctypes
    from ctypes import wintypes

    class LUID(ctypes.Structure):
        _fields_ = (("low", wintypes.DWORD), ("high", wintypes.LONG))

    class Desc1(ctypes.Structure):
        _fields_ = (
            ("description", wintypes.WCHAR * 128),
            ("vendor", wintypes.UINT),
            ("device", wintypes.UINT),
            ("subsys", wintypes.UINT),
            ("revision", wintypes.UINT),
            ("dedicated_video", ctypes.c_size_t),
            ("dedicated_system", ctypes.c_size_t),
            ("shared_system", ctypes.c_size_t),
            ("luid", LUID),
            ("flags", wintypes.UINT),
        )

    class GUID(ctypes.Structure):
        _fields_ = (("d1", wintypes.DWORD), ("d2", wintypes.WORD), ("d3", wintypes.WORD), ("d4", ctypes.c_ubyte * 8))

    def method(obj, index, *argtypes):
        vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtbl[index])

    iid = GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
    factory = ctypes.c_void_p()
    if ctypes.WinDLL("dxgi").CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) != 0:
        return []
    out = []
    try:
        enum = method(factory, 12, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))  # EnumAdapters1
        i = 0
        while True:
            adapter = ctypes.c_void_p()
            if enum(factory, i, ctypes.byref(adapter)) != 0:
                break
            d = Desc1()
            ok = method(adapter, 10, ctypes.POINTER(Desc1))(adapter, ctypes.byref(d)) == 0  # GetDesc1
            method(adapter, 2)(adapter)  # Release
            i += 1
            if ok and not d.flags & 2:  # 2: software adapter
                luid = (d.luid.high & 0xFFFFFFFF) << 32 | d.luid.low
                out.append(Adapter(d.description, d.vendor, d.dedicated_video // MiB, luid))
    finally:
        method(factory, 2)(factory)  # Release
    return out


class Counters:
    def __init__(self, *paths: str):
        import ctypes
        from ctypes import wintypes

        self.ct, self.wt = ctypes, wintypes
        self.lib = ctypes.WinDLL("pdh")
        self.q = wintypes.HANDLE()
        self.counters = []
        self.ok = self.lib.PdhOpenQueryW(None, None, ctypes.byref(self.q)) == 0
        for p in paths:
            c = wintypes.HANDLE()
            self.ok = self.ok and self.lib.PdhAddEnglishCounterW(self.q, p, None, ctypes.byref(c)) == 0
            self.counters.append(c)
        if self.ok:
            self.lib.PdhCollectQueryData(self.q)  # rates need two samples

    def read(self) -> list[dict[str, float]] | None:
        if not self.ok or self.lib.PdhCollectQueryData(self.q) != 0:
            return None
        out = [self._values(c) for c in self.counters]
        return None if any(v is None for v in out) else out

    def _values(self, counter) -> dict[str, float] | None:
        ct, wt = self.ct, self.wt

        class Fmt(ct.Structure):
            _fields_ = (("status", wt.DWORD), ("value", ct.c_double))

        class Item(ct.Structure):
            _fields_ = (("name", wt.LPWSTR), ("fmt", Fmt))

        fmt = 0x200 | 0x8000  # PDH_FMT_DOUBLE | PDH_FMT_NOCAP100
        size, count = wt.DWORD(0), wt.DWORD(0)
        self.lib.PdhGetFormattedCounterArrayW(counter, fmt, ct.byref(size), ct.byref(count), None)
        if not size.value:
            return {}
        buf = (ct.c_byte * size.value)()
        if self.lib.PdhGetFormattedCounterArrayW(counter, fmt, ct.byref(size), ct.byref(count), buf) != 0:
            return None
        items = ct.cast(buf, ct.POINTER(Item))
        return {items[i].name: items[i].fmt.value for i in range(count.value)}


def by_pid(values: dict[str, float]) -> dict[int, int]:
    used: dict[int, int] = {}
    for name, v in values.items():
        m = re.match(r"pid_(\d+)_", name)
        mib = int(v) // MiB
        if m and mib > 0:
            pid = int(m.group(1))
            used[pid] = used.get(pid, 0) + mib
    return used


def _luid(name: str) -> int | None:
    m = re.search(r"luid_0x([0-9a-fA-F]+)_0x([0-9a-fA-F]+)", name)
    return int(m.group(1), 16) << 32 | int(m.group(2), 16) if m else None


def by_luid(values: dict[str, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for name, v in values.items():
        luid = _luid(name)
        if luid is not None:
            out[luid] = out.get(luid, 0.0) + v
    return out


def busiest_engine(values: dict[str, float]) -> dict[int, float]:
    """Load per adapter: its busiest engine."""
    engines: dict[tuple[int, str], float] = {}
    for name, v in values.items():
        luid, m = _luid(name), re.search(r"_eng_(\d+)", name)
        if luid is not None and m:
            engines[luid, m.group(1)] = engines.get((luid, m.group(1)), 0.0) + v
    out: dict[int, float] = {}
    for (luid, _), v in engines.items():
        out[luid] = min(100.0, max(out.get(luid, 0.0), v))
    return out


_proc_mem: Counters | None = None


def process_memory() -> dict[int, int] | None:
    global _proc_mem
    if _proc_mem is None:
        _proc_mem = Counters(r"\GPU Process Memory(*)\Dedicated Usage")
    r = _proc_mem.read()
    return None if r is None else by_pid(r[0])
