"""Small cross-platform process helpers (spawning detached, suspending trees)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import psutil

IS_WINDOWS = os.name == "nt"
NO_WINDOW = 0x08000000 if IS_WINDOWS else 0  # CREATE_NO_WINDOW
NEW_GROUP = 0x00000200 if IS_WINDOWS else 0  # CREATE_NEW_PROCESS_GROUP
BREAKAWAY = 0x01000000 if IS_WINDOWS else 0  # CREATE_BREAKAWAY_FROM_JOB


def console_python() -> str:
    """A python executable that has a (hidden) console: under pythonw, child
    console programs would otherwise pop up their own windows."""
    exe = Path(sys.executable)
    if IS_WINDOWS and exe.name.lower() == "pythonw.exe":
        cand = exe.with_name("python.exe")
        if cand.exists():
            return str(cand)
    return str(exe)


def windowless_python() -> str:
    exe = Path(sys.executable)
    if IS_WINDOWS and exe.name.lower() == "python.exe":
        cand = exe.with_name("pythonw.exe")
        if cand.exists():
            return str(cand)
    return str(exe)


def spawn_detached(argv: list[str], **kw) -> subprocess.Popen:
    """Start a process that outlives its parent (and the terminal/agent that
    started it). On Windows, try to break away from the caller's job object so
    an agent's shell cleanup cannot kill it."""
    kw.setdefault("stdin", subprocess.DEVNULL)
    kw.setdefault("stdout", subprocess.DEVNULL)
    kw.setdefault("stderr", subprocess.DEVNULL)
    if IS_WINDOWS:
        base = NO_WINDOW | NEW_GROUP
        try:
            return subprocess.Popen(argv, creationflags=base | BREAKAWAY, **kw)
        except OSError:
            return subprocess.Popen(argv, creationflags=base, **kw)
    return subprocess.Popen(argv, start_new_session=True, **kw)


def get_proc(pid: int | None, ctime: float | None = None) -> psutil.Process | None:
    """The process with this pid, if it is alive and (when ctime is given) is
    the same process we started (guards against pid reuse)."""
    if not pid:
        return None
    try:
        p = psutil.Process(pid)
        if ctime and abs(p.create_time() - ctime) > 1.0:
            return None
        # Zombies only exist on POSIX; on Windows status() scans every process.
        if not IS_WINDOWS and p.status() == psutil.STATUS_ZOMBIE:
            return None
        return p
    except psutil.Error:
        return None


def alive(pid: int | None, ctime: float | None = None) -> bool:
    return get_proc(pid, ctime) is not None


def create_time(pid: int) -> float | None:
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error:
        return None


def tree(p: psutil.Process) -> list[psutil.Process]:
    try:
        return [p, *p.children(recursive=True)]
    except psutil.Error:
        return [p]


class Snapshot:
    """Parent links of all processes, read once. On Windows every psutil
    children()/parents() call scans the whole process list, so the daemon
    takes one snapshot per poll and answers all its tree questions from it."""

    def __init__(self):
        links = None
        fast = getattr(psutil._psplatform, "ppid_map", None)  # one scan; not public API, so optional
        if fast is not None:
            try:
                links = dict(fast())
            except Exception:
                links = None
        if links is None:
            links = {p.pid: p.info["ppid"] for p in psutil.process_iter(["ppid"])}
        self.parent: dict[int, int] = links
        self.kids: dict[int, list[int]] = {}
        for pid, ppid in links.items():
            if ppid != pid:
                self.kids.setdefault(ppid, []).append(pid)

    def tree(self, pid: int) -> list[int]:
        """pid and all its descendants."""
        out, todo, seen = [], [pid], set()
        while todo:
            p = todo.pop()
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
            todo.extend(self.kids.get(p, ()))
        return out

    def ancestors(self, pid: int) -> list[int]:
        out, seen = [], {pid}
        p = self.parent.get(pid)
        while p and p not in seen:
            seen.add(p)
            out.append(p)
            p = self.parent.get(p)
        return out


MiB = 2**20


def system_memory() -> tuple[int, int]:
    """(total, in use) RAM in MiB. GPUSHIFT_FAKE_RAM="total,used" (MiB)
    replaces the reading, for tests."""
    fake = os.environ.get("GPUSHIFT_FAKE_RAM")
    if fake:
        total, used = (int(x) for x in fake.split(","))
        return total, used
    vm = psutil.virtual_memory()
    return vm.total // MiB, (vm.total - vm.available) // MiB


def tree_memory(pids) -> tuple[int, int, int]:
    """(resident, private, peak private) RAM in MiB of these processes together.

    resident: everything in RAM, including pages of memory-mapped files and
    DLLs, which the OS can drop and reread whenever it needs the RAM.
    private: the part it can't drop, i.e. what a job really needs. On Windows,
    min(private commit, working set): CUDA commits far more than it touches,
    and mapped files fill the working set without being private. Elsewhere
    resident minus shared (macOS reports no shared memory, so there mapped
    files count as private: errs high). peak: the same from each process's recorded peaks
    (Windows; a sum of peaks reached at different times errs high, the safe
    side), else the current value."""
    res = priv = peak = 0
    for pid in pids:
        try:
            mi = psutil.Process(pid).memory_info()
        except psutil.Error:
            continue
        res += mi.rss
        if IS_WINDOWS:
            p = min(mi.private, mi.rss)
            peak += max(p, min(mi.peak_pagefile, mi.peak_wset))
        else:
            p = max(0, mi.rss - getattr(mi, "shared", 0))
            peak += p
        priv += p
    return res // MiB, priv // MiB, peak // MiB


def suspend_tree(pid: int | None, ctime: float | None = None) -> bool:
    p = get_proc(pid, ctime)
    if p is None:
        return False
    # Parent first so it cannot spawn new children while we work down the tree.
    procs = tree(p)
    for q in procs:
        try:
            q.suspend()
        except psutil.Error:
            pass
    # Once everything is paused, hand the frozen job's RAM back to the user.
    for q in procs:
        trim_memory(q.pid)
    return True


def trim_memory(pid: int) -> bool:
    """Ask Windows to move a process's resident pages out of RAM (file-backed
    pages are dropped, the rest go to the page file). Nothing is lost: pages
    fault back in when the process runs again. Locked pages (e.g. CUDA pinned
    buffers) simply stay. Best effort; a no-op off Windows."""
    if not IS_WINDOWS:
        return False
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    k32.K32EmptyWorkingSet.argtypes = (wintypes.HANDLE,)
    access = 0x0400 | 0x0100  # PROCESS_QUERY_INFORMATION | PROCESS_SET_QUOTA
    h = k32.OpenProcess(access, False, pid)
    if not h:
        return False
    try:
        return bool(k32.K32EmptyWorkingSet(h))
    finally:
        k32.CloseHandle(h)


def resume_tree(pid: int | None, ctime: float | None = None) -> bool:
    p = get_proc(pid, ctime)
    if p is None:
        return False
    for q in reversed(tree(p)):
        try:
            q.resume()
        except psutil.Error:
            pass
    return True


def kill(pid: int | None, ctime: float | None = None, timeout: float = 5.0) -> bool:
    """Kill this one process, leaving its children running."""
    p = get_proc(pid, ctime)
    if p is None:
        return False
    try:
        p.kill()
        p.wait(timeout)
    except (psutil.Error, psutil.TimeoutExpired):
        pass
    return True


def kill_tree(pid: int | None, ctime: float | None = None, timeout: float = 5.0) -> bool:
    p = get_proc(pid, ctime)
    if p is None:
        return False
    procs = tree(p)
    for q in reversed(procs):
        try:
            q.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=timeout)
    return True


def terminate(pid: int | None, ctime: float | None = None) -> bool:
    """Polite stop request: SIGTERM on POSIX. No-op on Windows (gpushift uses a
    stop file there, which the checkpoint helper polls)."""
    if IS_WINDOWS:
        return False
    p = get_proc(pid, ctime)
    if p is None:
        return False
    try:
        p.terminate()
        return True
    except psutil.Error:
        return False
