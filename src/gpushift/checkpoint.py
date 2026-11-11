"""gpushift.checkpoint: make a training script evictable. Dependency-free.

Jobs started by gpushift can always ``import gpushift.checkpoint``, even in an
environment where gpushift is not installed (gpushift puts this file on PYTHONPATH).

Usage::

    from gpushift import checkpoint as ckpt

    state = load(ckpt.latest_checkpoint("ckpts"))   # None on a fresh start
    for step in range(start, total):
        train_step()
        if step % 500 == 0 or ckpt.should_stop():
            save(f"ckpts/step{step}.pt")
        if ckpt.should_stop():
            break                                   # exit cleanly; gpushift requeues the job

When ``gpushift free`` needs the GPUs back, it asks the job to stop (SIGTERM on
Linux/macOS, a stop file on every platform). ``should_stop()`` turns True;
the script saves and exits; gpushift puts the job back at the front of the queue
with ``GPUSHIFT_RESUME=1`` so it continues from its latest checkpoint.

Calling ``should_stop()`` once marks the job as resumable, so gpushift evicts it
instead of freezing it. Scripts that never call it are frozen (their VRAM stays
allocated) during ``gpushift free``.
"""

from __future__ import annotations

import contextlib
import os
import signal
import threading
import time
from pathlib import Path

__all__ = [
    "atomic_path",
    "install",
    "job_id",
    "latest_checkpoint",
    "managed",
    "resuming",
    "should_stop",
]

_stop = False
_installed = False
_last_poll = 0.0


def managed() -> bool:
    """True when running as a gpushift job."""
    return "GPUSHIFT_JOB_ID" in os.environ


def job_id() -> int | None:
    v = os.environ.get("GPUSHIFT_JOB_ID")
    return int(v) if v else None


def _on_signal(signum, frame):
    global _stop
    _stop = True


def install() -> None:
    """Install the stop handlers and mark this job resumable. Idempotent;
    ``should_stop()`` calls it for you."""
    global _installed
    if _installed:
        return
    _installed = True
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(sig, _on_signal)
    d = os.environ.get("GPUSHIFT_JOB_DIR")
    if d:
        with contextlib.suppress(OSError):
            Path(d).mkdir(parents=True, exist_ok=True)
            Path(d, "resumable").touch()


def should_stop(poll_interval: float = 1.0) -> bool:
    """True once gpushift (or anyone sending SIGTERM) wants the job to stop.
    Cheap enough to call every step: the stop file is checked at most once per
    ``poll_interval`` seconds."""
    global _stop, _last_poll
    install()
    if _stop:
        return True
    f = os.environ.get("GPUSHIFT_STOP_FILE")
    if f:
        now = time.monotonic()
        if now - _last_poll >= poll_interval:
            _last_poll = now
            if os.path.exists(f):
                _stop = True
    return _stop


def resuming() -> bool:
    """True when gpushift restarted this job after an eviction."""
    return os.environ.get("GPUSHIFT_RESUME") == "1"


def latest_checkpoint(directory, pattern: str = "*") -> Path | None:
    """Newest file or directory in ``directory`` matching ``pattern`` (by
    modification time), or None. Temp files from ``atomic_path`` are ignored."""
    d = Path(directory)
    if not d.is_dir():
        return None
    cands = [p for p in d.glob(pattern) if not p.name.startswith(".") and not p.name.endswith(".gpushift-tmp")]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


@contextlib.contextmanager
def atomic_path(path):
    """Write a checkpoint without ever leaving a half-written file behind::

    with ckpt.atomic_path("ckpts/last.pt") as tmp:
        torch.save(state, tmp)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".gpushift-tmp")
    yield tmp
    os.replace(tmp, path)


def _reset_for_tests() -> None:
    global _stop, _installed, _last_poll
    _stop = False
    _installed = False
    _last_poll = 0.0
