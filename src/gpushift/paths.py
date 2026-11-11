"""Where gpushift keeps its config and state.

Config: ``~/.config/gpushift/config.toml`` (respects ``XDG_CONFIG_HOME``).
State (queue database, job logs, daemon log): a per-platform state dir.
``GPUSHIFT_HOME`` overrides both (used by tests and for side-by-side installs).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _home_override() -> Path | None:
    h = os.environ.get("GPUSHIFT_HOME")
    return Path(h) if h else None


def config_dir() -> Path:
    h = _home_override()
    if h:
        return h
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "gpushift"


def state_dir() -> Path:
    h = _home_override()
    if h:
        return h
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return (Path(base) if base else Path.home() / "AppData" / "Local") / "gpushift"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "gpushift"
    base = os.environ.get("XDG_STATE_HOME")
    return (Path(base) if base else Path.home() / ".local" / "state") / "gpushift"


def config_file() -> Path:
    return config_dir() / "config.toml"


def db_file() -> Path:
    return state_dir() / "gpushift.db"


def job_dir(job_id: int) -> Path:
    return state_dir() / "jobs" / str(job_id)


def log_file(job_id: int) -> Path:
    return job_dir(job_id) / "output.log"


def stop_file(job_id: int) -> Path:
    return job_dir(job_id) / "stop"


def resumable_marker(job_id: int) -> Path:
    return job_dir(job_id) / "resumable"


def daemon_lock() -> Path:
    return state_dir() / "daemon.lock"


def daemon_log() -> Path:
    return state_dir() / "daemon.log"


def pylib_dir() -> Path:
    """Directory appended to jobs' PYTHONPATH so ``import gpushift.checkpoint``
    works in any environment, even one without gpushift installed."""
    return state_dir() / "pylib"
