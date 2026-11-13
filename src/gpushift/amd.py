"""AMD GPUs on Linux (amdgpu driver) and Windows (DXGI). Indices follow HIP's order."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .gpus import GPU

LINUX = sys.platform.startswith("linux")

AMD = 0x1002
KFD = "class/kfd/kfd"
PCI_IDS = ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids", "/usr/share/pci.ids")
MiB = 1024 * 1024


def _read(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _int(p: Path) -> int | None:
    s = _read(p)
    try:
        return int(s, 0) if s else None
    except ValueError:
        return None


def _props(node: Path) -> dict[str, int]:
    out = {}
    for line in (_read(node / "properties") or "").splitlines():
        k, _, v = line.partition(" ")
        try:
            out[k] = int(v)
        except ValueError:
            pass
    return out


def devices(sys_root: str = "/sys") -> list[tuple[int, int | None, Path, dict[str, int]]]:
    """(index, gpu_id, PCI device dir, properties) per GPU."""
    nodes_dir = Path(sys_root, KFD, "topology", "nodes")
    try:
        nodes = sorted((n for n in nodes_dir.iterdir() if n.name.isdigit()), key=lambda n: int(n.name))
    except OSError:
        return []
    out = []
    for n in nodes:
        p = _props(n)
        if not p.get("simd_count"):  # CPU node
            continue
        loc, dom = p.get("location_id", 0), p.get("domain", 0)
        bdf = f"{dom:04x}:{loc >> 8 & 0xFF:02x}:{loc >> 3 & 0x1F:02x}.{loc & 7:x}"
        out.append((len(out), _int(n / "gpu_id"), Path(sys_root, "bus/pci/devices", bdf), p))
    return out


_names: dict[int, str | None] = {}


def _pci_name(device_id: int) -> str | None:
    if device_id in _names:
        return _names[device_id]
    name = None
    want = f"\t{device_id:04x}  "
    for f in PCI_IDS:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                in_amd = False
                for line in fh:
                    if not line.strip() or line.startswith("#"):
                        continue
                    if not line.startswith("\t"):
                        if in_amd:
                            break
                        in_amd = line.startswith(f"{AMD:04x}  ")
                    elif in_amd and line.startswith(want):
                        name = line[len(want) :].strip()
                        break
            break
        except OSError:
            continue
    _names[device_id] = name
    return name


def _name(dev: Path, props: dict[str, int]) -> str:
    product = _read(dev / "product_name")
    if product:
        return product
    pci = _pci_name(_int(dev / "device") or props.get("device_id", 0))
    if pci:
        if "[" in pci and pci.endswith("]"):
            pci = pci[pci.index("[") + 1 : -1]
        return f"AMD {pci}"
    v = props.get("gfx_target_version", 0)
    return f"AMD GPU (gfx{v // 10000}{v // 100 % 100:x}{v % 100:x})" if v else "AMD GPU"


def _hwmon(dev: Path) -> Path | None:
    try:
        return next(iter(sorted((dev / "hwmon").iterdir())), None)
    except OSError:
        return None


def available() -> bool:
    if os.name == "nt":
        return bool(_windows_adapters())
    return LINUX and bool(devices())


def query() -> list[GPU]:
    if os.name == "nt":
        return query_windows()
    return query_linux() if LINUX else []


def per_process(pids=None) -> dict[int, int] | None:
    return per_process_linux(pids) if LINUX and devices() else None


def visible_env(env: dict[str, str], gpus: list[int], windows: bool = os.name == "nt") -> None:
    ids = ",".join(map(str, gpus))
    if windows:
        env["HIP_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"] = ids or "-1"
        return
    # HIP filters again after ROCR, so it gets 0..n-1. -1 hides all GPUs.
    env["ROCR_VISIBLE_DEVICES"] = ids
    env["HIP_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(len(gpus)))) or "-1"


def _windows_adapters():
    from . import windows

    try:
        return [a for a in windows.adapters() if a.vendor == AMD]
    except OSError:
        return []


_counters = None


def query_windows() -> list[GPU]:
    # Assumes HIP numbers AMD GPUs in DXGI order.
    from . import windows

    global _counters
    found = _windows_adapters()
    if not found:
        return []
    if _counters is None:
        _counters = windows.Counters(
            r"\GPU Adapter Memory(*)\Dedicated Usage", r"\GPU Engine(*)\Utilization Percentage"
        )
    r = _counters.read()
    mem, load = (windows.by_luid(r[0]), windows.busiest_engine(r[1])) if r else ({}, {})
    return [
        GPU(
            i,
            a.name,
            a.mem_total,
            int(mem.get(a.luid, 0)) // MiB,
            util=round(load[a.luid]) if a.luid in load else None,
            vendor="amd",
        )
        for i, a in enumerate(found)
    ]


def query_linux(sys_root: str = "/sys") -> list[GPU]:
    out = []
    for idx, _, dev, props in devices(sys_root):
        g = GPU(
            idx,
            _name(dev, props),
            (_int(dev / "mem_info_vram_total") or 0) // MiB,
            (_int(dev / "mem_info_vram_used") or 0) // MiB,
            vendor="amd",
        )
        g.util = _int(dev / "gpu_busy_percent")
        hw = _hwmon(dev)
        if hw is not None:
            t = next((p for p in sorted(hw.glob("temp*_label")) if _read(p) == "edge"), hw / "temp1_label")
            temp = _int(hw / t.name.replace("_label", "_input"))
            g.temp = temp // 1000 if temp is not None else None
            power = _int(hw / "power1_input") or _int(hw / "power1_average")
            cap = _int(hw / "power1_cap")
            g.power = round(power / 1e6, 1) if power is not None else None
            g.power_limit = round(cap / 1e6, 1) if cap else None
            pwm, pwm_max = _int(hw / "pwm1"), _int(hw / "pwm1_max")
            g.fan = round(100 * pwm / pwm_max) if pwm is not None and pwm_max else None
            clock = _int(hw / "freq1_input")
            g.clock = clock // 1_000_000 if clock is not None and _read(hw / "freq1_label") == "sclk" else None
        out.append(g)
    return out


def _kib(v: str | None) -> int:
    """Size in KiB. No unit means bytes."""
    if not v:
        return 0
    num, _, unit = v.partition(" ")
    try:
        n = int(num)
    except ValueError:
        return 0
    return {"": n // 1024, "KiB": n, "MiB": n * 1024, "GiB": n * 1024 * 1024}.get(unit.strip(), 0)


def _fdinfo_kib(proc: Path) -> int:
    try:
        fds = os.listdir(proc / "fd")
    except OSError:
        return 0
    seen: set[tuple[str | None, str | None]] = set()
    total = 0
    for fd in fds:
        try:
            if not os.readlink(proc / "fd" / fd).startswith("/dev/dri/"):
                continue
        except OSError:
            continue
        info = {}
        for line in (_read(proc / "fdinfo" / fd) or "").splitlines():
            k, _, v = line.partition(":")
            info[k.strip()] = v.strip()
        if info.get("drm-driver") != "amdgpu":
            continue
        client = (info.get("drm-pdev"), info.get("drm-client-id"))
        if client in seen:
            continue
        seen.add(client)
        total += _kib(info.get("drm-resident-vram") or info.get("drm-memory-vram"))
    return total


def per_process_linux(pids=None, proc_root: str = "/proc", sys_root: str = "/sys") -> dict[int, int]:
    if pids is None:
        try:
            pids = [int(p) for p in os.listdir(proc_root) if p.isdigit()]
        except OSError:
            return {}
    gpu_ids = [gid for _, gid, _, _ in devices(sys_root) if gid is not None]
    kfd = Path(sys_root, KFD, "proc")
    used = {}
    for pid in pids:
        kib = _fdinfo_kib(Path(proc_root, str(pid)))
        kfd_kib = sum(_int(kfd / str(pid) / f"vram_{gid}") or 0 for gid in gpu_ids) // 1024
        mib = max(kib, kfd_kib) // 1024
        if mib:
            used[pid] = mib
    return used
