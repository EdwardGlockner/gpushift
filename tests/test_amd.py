"""AMD support, against a fake /sys and /proc."""

import os
from pathlib import Path

import pytest

from gpushift import amdgpu, runner


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def node(sys_root: Path, n: int, simd: int, location: int = 0, gpu_id: int = 0, gfx: int = 0) -> None:
    d = sys_root / "class/kfd/kfd/topology/nodes" / str(n)
    write(d / "gpu_id", f"{gpu_id}\n")
    write(
        d / "properties",
        f"cpu_cores_count {0 if simd else 8}\nsimd_count {simd}\ngfx_target_version {gfx}\n"
        f"vendor_id {4098 if simd else 0}\ndevice_id 0\nlocation_id {location}\ndomain 0\n",
    )


@pytest.fixture
def sysfs(tmp_path, monkeypatch):
    """A CPU node and two GPUs; the second has no sensors."""
    root = tmp_path / "sys"
    node(root, 0, 0)
    node(root, 1, 64, location=0x300, gpu_id=7773, gfx=120000)
    node(root, 2, 96, location=0x400, gpu_id=8001, gfx=90010)
    dev = root / "bus/pci/devices/0000:03:00.0"
    for f, v in {
        "device": "0x7590",
        "mem_info_vram_total": str(16304 * 2**20),
        "mem_info_vram_used": str(2336 * 2**20),
        "gpu_busy_percent": "39",
    }.items():
        write(dev / f, v + "\n")
    hw = dev / "hwmon/hwmon3"
    for f, v in {
        "temp1_label": "edge",
        "temp1_input": "55000",
        "temp2_label": "junction",
        "temp2_input": "70000",
        "power1_average": "42000000",
        "power1_cap": "182000000",
        "pwm1": "51",
        "pwm1_max": "255",
        "freq1_label": "sclk",
        "freq1_input": "1785000000",
    }.items():
        write(hw / f, v + "\n")
    write(root / "bus/pci/devices/0000:04:00.0/mem_info_vram_total", str(64 * 2**30))
    ids = tmp_path / "pci.ids"
    ids.write_text(
        "# comment\n"
        "1000  Broadcom\n\t7590  Not this one\n"
        "1002  Advanced Micro Devices, Inc. [AMD/ATI]\n"
        "\t7480  Navi 33 [Radeon RX 7600]\n"
        "\t7590  Navi 44 [Radeon RX 9050 / 9060 XT]\n"
        "\t\t1002 0e31  Subsystem\n"
        "1003  ULSI\n"
    )
    monkeypatch.setattr(amdgpu, "PCI_IDS", (str(tmp_path / "missing.ids"), str(ids)))
    monkeypatch.setattr(amdgpu, "_names", {})
    return root


def test_gpus_in_rocm_order_with_readings(sysfs):
    g0, g1 = amdgpu.query(str(sysfs))
    assert (g0.index, g0.name, g0.mem_total, g0.mem_used, g0.vendor) == (
        0,
        "AMD Radeon RX 9050 / 9060 XT",
        16304,
        2336,
        "amd",
    )
    assert (g0.util, g0.temp, g0.power, g0.power_limit, g0.fan, g0.clock) == (39, 55, 42.0, 182.0, 20, 1785)
    assert (g1.index, g1.name, g1.mem_total, g1.mem_used, g1.temp, g1.util) == (
        1,
        "AMD GPU (gfx90a)",
        65536,
        0,
        None,
        None,
    )


def test_no_amd_gpus(tmp_path):
    assert amdgpu.query(str(tmp_path)) == []
    assert amdgpu.per_process([1], str(tmp_path), str(tmp_path)) == {}


def fd(proc: Path, pid: int, n: int, target: str, info: str = "") -> None:
    d = proc / str(pid)
    (d / "fd").mkdir(parents=True, exist_ok=True)
    os.symlink(target, d / "fd" / str(n))
    write(d / "fdinfo" / str(n), info)


def amd_fd(client: int, kib: int, key: str = "drm-resident-vram") -> str:
    return f"pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t{client}\ndrm-pdev:\t0000:03:00.0\n{key}:\t{kib} KiB\n"


def test_per_process_memory(sysfs, tmp_path):
    proc = tmp_path / "proc"
    # Same client counts once.
    fd(proc, 100, 3, "/dev/dri/renderD128", amd_fd(40, 1024 * 1024))
    fd(proc, 100, 4, "/dev/dri/renderD128", amd_fd(40, 1024 * 1024))
    fd(proc, 100, 5, "/dev/dri/renderD128", amd_fd(41, 512 * 1024, key="drm-memory-vram"))
    fd(proc, 100, 6, "/dev/null")
    fd(proc, 200, 3, "/dev/dri/renderD129", "drm-driver:\ti915\ndrm-client-id:\t1\ndrm-resident-vram:\t9999 KiB\n")
    # KFD reports more than fdinfo.
    fd(proc, 300, 3, "/dev/dri/renderD128", amd_fd(50, 100 * 1024))
    write(sysfs / "class/kfd/kfd/proc/300/vram_7773", str(3 * 2**30))
    write(sysfs / "class/kfd/kfd/proc/300/vram_8001", str(1 * 2**30))
    used = amdgpu.per_process([100, 200, 300, 400], str(proc), str(sysfs))
    assert used == {100: 1536, 300: 4096}
    assert amdgpu.per_process([300], str(proc), str(sysfs)) == {300: 4096}
    assert amdgpu.per_process(None, str(proc), str(sysfs)) == used


JOB = {
    "id": 7,
    "env": {"CUDA_VISIBLE_DEVICES": "3", "HIP_VISIBLE_DEVICES": "2"},  # inherited
    "attempts": 0,
    "resume": 0,
    "gpus": [1, 2],
    "preset": "two",
    "n_gpus": 2,
}


def test_amd_job_sees_only_its_gpus(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUSHIFT_HOME", str(tmp_path))
    env = runner._job_env(JOB, "amd")
    assert (env["ROCR_VISIBLE_DEVICES"], env["HIP_VISIBLE_DEVICES"], env["CUDA_VISIBLE_DEVICES"]) == (
        "1,2",
        "0,1",
        "0,1",
    )
    cpu = runner._job_env({**JOB, "gpus": [], "preset": "cpu", "n_gpus": 0}, "amd")
    assert (cpu["ROCR_VISIBLE_DEVICES"], cpu["HIP_VISIBLE_DEVICES"], cpu["CUDA_VISIBLE_DEVICES"]) == ("", "-1", "-1")
    nv = runner._job_env(JOB, "nvidia")
    assert nv["CUDA_VISIBLE_DEVICES"] == "1,2" and "ROCR_VISIBLE_DEVICES" not in nv
