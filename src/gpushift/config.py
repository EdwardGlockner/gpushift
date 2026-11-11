"""Configuration: ~/.config/gpushift/config.toml, with defaults that scale to the
machine (number of GPUs, CPU-only)."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field

from . import gpus as gpus_mod
from . import paths

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


class ConfigError(ValueError):
    pass


@dataclass
class HookConfig:
    enabled: bool = True
    allow: list[str] = field(default_factory=list)  # extra regexes that are never blocked
    deny: list[str] = field(default_factory=list)  # extra regexes that are always blocked


# Programs that use the GPU all the time without being "heavy" GPU work.
DEFAULT_IGNORE = [
    "dwm.exe", "explorer.exe", "csrss.exe", "searchapp.exe", "searchhost.exe", "shellexperiencehost.exe",
    "startmenuexperiencehost.exe", "textinputhost.exe", "lockapp.exe", "applicationframehost.exe",
    "chrome.exe", "msedge.exe", "msedgewebview2.exe", "firefox.exe", "brave.exe", "opera.exe",
    "discord.exe", "code.exe", "slack.exe", "teams.exe", "ms-teams.exe", "spotify.exe", "steamwebhelper.exe",
    "xorg", "gnome-shell", "kwin_x11", "kwin_wayland", "plasmashell", "windowserver",
]  # fmt: skip


@dataclass
class AutoFreeConfig:
    """Free the GPUs while the user runs heavy GPU programs (games, Blender, ...)."""

    enabled: bool = True
    apps: list[str] = field(default_factory=list)  # always count these programs (any GPU memory)
    ignore: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORE))
    min_vram_mb: int = 1024  # other programs count once they use this much GPU memory
    after: int = 10  # seconds a program must keep using the GPU before gpushift frees it
    resume_after: int = 120  # seconds after the last such program exits before gpushift resumes
    include_agents: bool = False  # also count programs started by Claude Code sessions


@dataclass
class Config:
    gpus: list[int]  # GPU indices gpushift may use
    slots: int  # total slots (concurrency budget)
    shared_gpu: int | None  # GPU that --light jobs share
    gpu_free_perc: int = 75  # an unclaimed GPU counts as free if this % of its memory is free
    starvation_guard: bool = True
    fair_share: bool = True  # sessions take turns instead of strict submit order
    packing: bool = True  # jobs with a known GPU memory need may share a GPU
    free_mode_default: str = "full"  # full | soft
    evict_timeout: int = 300  # seconds a resumable job gets to checkpoint before it is frozen
    notify_desktop: str = "failures"  # all | failures | off
    ui_port: int = 8237  # web dashboard on 127.0.0.1; 0 turns it off
    hook: HookConfig = field(default_factory=HookConfig)
    auto_free: AutoFreeConfig = field(default_factory=AutoFreeConfig)
    detected: list[gpus_mod.GPU] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("detected")
        return d


def default_slots(n_gpus: int) -> int:
    if n_gpus:
        return 2 * n_gpus
    return max(1, (os.cpu_count() or 2) // 2)


def read_raw() -> dict:
    f = paths.config_file()
    if not f.exists():
        return {}
    try:
        with open(f, "rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{f}: {e}") from e


def load(detected: list[gpus_mod.GPU] | None = None) -> Config:
    raw = read_raw()
    if detected is None:
        detected = gpus_mod.detect()
    all_idx = [g.index for g in detected]

    vis = raw.get("gpus")
    if vis is None:
        vis = all_idx
    if not isinstance(vis, list) or not all(isinstance(i, int) for i in vis):
        raise ConfigError("gpus must be a list of GPU indices, e.g. gpus = [0, 1]")
    if detected:
        missing = [i for i in vis if i not in all_idx]
        if missing:
            raise ConfigError(f"gpus lists {missing}, but only {all_idx} were detected")

    slots = raw.get("slots", default_slots(len(vis)))
    if not isinstance(slots, int) or slots < 1:
        raise ConfigError("slots must be an integer >= 1")

    shared = raw.get("shared_gpu", vis[-1] if vis else None)
    if shared is not None and shared not in vis:
        raise ConfigError(f"shared_gpu = {shared} is not one of the usable GPUs {vis}")

    perc = raw.get("gpu_free_perc", 75)
    if not isinstance(perc, int) or not 0 <= perc <= 100:
        raise ConfigError("gpu_free_perc must be an integer between 0 and 100")

    mode = raw.get("free_mode_default", "full")
    if mode not in ("full", "soft"):
        raise ConfigError('free_mode_default must be "full" or "soft"')

    evict_timeout = raw.get("evict_timeout", 300)
    if not isinstance(evict_timeout, int) or evict_timeout < 1:
        raise ConfigError("evict_timeout must be a positive number of seconds")

    h = raw.get("hook", {}) or {}
    hook = HookConfig(
        enabled=bool(h.get("enabled", True)),
        allow=list(h.get("allow", [])),
        deny=list(h.get("deny", [])),
    )

    notify_desktop = (raw.get("notify", {}) or {}).get("desktop", "failures")
    if notify_desktop not in ("all", "failures", "off"):
        raise ConfigError('[notify] desktop must be "all", "failures" or "off"')

    ui_port = raw.get("ui_port", 8237)
    if not isinstance(ui_port, int) or not 0 <= ui_port <= 65535:
        raise ConfigError("ui_port must be a port number (0 turns the dashboard off)")

    af = raw.get("auto_free", {}) or {}
    d = AutoFreeConfig()
    auto_free = AutoFreeConfig(
        enabled=bool(af.get("enabled", d.enabled)),
        apps=[str(a).lower() for a in af.get("apps", d.apps)],
        ignore=[str(a).lower() for a in af.get("ignore", d.ignore)],
        min_vram_mb=af.get("min_vram_mb", d.min_vram_mb),
        after=af.get("after", d.after),
        resume_after=af.get("resume_after", d.resume_after),
        include_agents=bool(af.get("include_agents", d.include_agents)),
    )
    for k in ("min_vram_mb", "after", "resume_after"):
        v = getattr(auto_free, k)
        if not isinstance(v, int) or v < 0:
            raise ConfigError(f"[auto_free] {k} must be a whole number >= 0")
    return Config(
        gpus=list(vis),
        slots=slots,
        shared_gpu=shared,
        gpu_free_perc=perc,
        starvation_guard=bool(raw.get("starvation_guard", True)),
        fair_share=bool(raw.get("fair_share", True)),
        packing=bool(raw.get("packing", True)),
        free_mode_default=mode,
        evict_timeout=evict_timeout,
        notify_desktop=notify_desktop,
        ui_port=ui_port,
        hook=hook,
        auto_free=auto_free,
        detected=[g for g in detected if g.index in vis],
    )


def template(detected: list[gpus_mod.GPU]) -> str:
    idx = [g.index for g in detected]
    gpu_lines = (
        "\n".join(f"#   GPU {g.index}: {g.name} ({g.mem_total / 1024:.1f} GB)" for g in detected)
        or "#   (none: CPU-only mode)"
    )
    shared = idx[-1] if idx else None
    return f"""\
# gpushift configuration. Every setting is optional; the commented values are the
# defaults gpushift computed for this machine. Uncomment a line to override it.
#
# Detected GPUs:
{gpu_lines}

# GPU indices gpushift may schedule on (like CUDA_VISIBLE_DEVICES).
# gpus = {idx}

# Total slots: the machine's concurrency budget. --light/--cpu use 1,
# --one/--two use 2, --exclusive uses all. Default: 2 per GPU.
# slots = {default_slots(len(idx))}

# GPU that --light jobs share. Default: the last GPU.
# shared_gpu = {shared if shared is not None else '"none"'}

# An unclaimed GPU only counts as free if this % of its memory is free. Keep it
# below 100 minus what your desktop/display uses on that GPU.
# gpu_free_perc = 75

# true: a waiting multi-GPU/exclusive job holds back everything behind it,
# so big jobs cannot starve; jobs behind it may only use idle GPUs if their
# estimate says they finish before it could start. false: any job that fits
# may jump ahead.
# starvation_guard = true

# What `gpushift free` does by default: "full" (evict resumable jobs, freeze the
# rest) or "soft" (freeze everything, VRAM stays allocated).
# free_mode_default = "full"

# Seconds a resumable job gets to save a checkpoint and exit during `gpushift free`
# before it is frozen instead.
# evict_timeout = 300

# true: Claude sessions (and you) take turns, so one session's long list of
# jobs can't hold everyone else up. false: strict submit order.
# fair_share = true

# true: a job whose GPU memory need is known (--vram, or learned from earlier
# runs of the same command) may share a GPU with other such jobs if they fit.
# packing = true

# Port of the web dashboard (`gpushift ui`), on 127.0.0.1 only. 0 turns it off.
# ui_port = 8237

[notify]
# Desktop notifications when jobs end: "all", "failures" or "off".
# desktop = "failures"

[auto_free]
# Free the GPUs while you run heavy GPU programs yourself (games, Blender, ...),
# and resume when they have been closed for a while.
# enabled = true
# Programs that always count, whatever GPU memory they use:
# apps = ["blender.exe", "Overwatch.exe"]
# Other programs count once they use this much GPU memory (MB):
# min_vram_mb = 1024
# Seconds a program must keep going before gpushift frees, and seconds after the
# last one exits before gpushift resumes:
# after = 10
# resume_after = 120

[hook]
# The Claude Code hook that redirects direct GPU launches to `gpushift submit`.
# enabled = true
# Regexes for commands that must never be blocked / always be blocked.
# allow = ['^python scripts/quick_check\\.py']
# deny = ['my_trainer']
"""
