"""Write the README screenshots of the dashboard, in a light and a dark version.

    python demos/dashboard.py docs/img

Renders the real dashboard page (src/gpushift/ui.html) with made-up example data
in headless Chrome, Edge or Chromium, and writes gpushift-dashboard-light.png and
gpushift-dashboard-dark.png.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "src" / "gpushift" / "ui.html"
W, H = 1280, 1225
M, HR = 60, 3600


def iso(t: float | None) -> str | None:
    return None if t is None else time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t))


def job(now, id, tag, project, session, preset, command, *, gpus=(), state="queued", started=None, ran=0,
        takes=None, queued_ago=30 * M, vram=None, turn=None, after=()):  # fmt: skip
    return {
        "id": id, "state": state, "preset": preset, "gpus": list(gpus), "tag": tag, "project": project,
        "session": session, "command": command, "cwd": f"/home/ana/{project}", "submitted_at": iso(now - queued_ago),
        "started_at": iso(started), "ended_at": None, "frozen_at": None, "runtime_s": ran,
        "expected_duration_s": takes, "resumable": False, "restartable": False, "vram_now_mb": vram,
        "turn": turn, "waiting_for": list(after), "after": list(after), "note": None, "exit_code": None,
    }  # fmt: skip


def status(now: float) -> dict:
    a, b = "4f1e9c2a-0000", "b7d03e51-0000"
    running = [
        job(now, 41, "llama-sft", "llama-sft", a, "one", "torchrun --nproc_per_node 1 sft.py --config lora.yaml",
            gpus=[0], state="running", started=now - 2 * HR, ran=2 * HR, takes=5 * HR, vram=6900),
        job(now, 43, "resnet-sweep", "vision", b, "one", "python train.py --lr 3e-4 --epochs 90",
            gpus=[1], state="running", started=now - 50 * M, ran=50 * M, takes=2 * HR, vram=3900),
    ]  # fmt: skip
    queue = [
        job(now, 44, "resnet-eval", "vision", b, "light", "python eval.py --ckpt runs/sweep/last.pt",
            takes=10 * M, turn=1, after=[43], queued_ago=48 * M),
        job(now, 45, "tokenize", "llama-sft", a, "cpu", "python scripts/tokenize.py --shards 64",
            takes=25 * M, turn=2, queued_ago=20 * M),
        job(now, 46, "llama-eval", "llama-sft", a, "one", "python eval.py --tasks mmlu,gsm8k",
            takes=40 * M, turn=3, after=[41], queued_ago=12 * M),
        job(now, 47, "bench", None, None, "exclusive", "python bench.py --all", takes=15 * M, turn=4,
            queued_ago=5 * M),
    ]  # fmt: skip
    recent = []
    for id, tag, project, session, state, ran, ago in [
        (40, "llama-data", "llama-sft", a, "done", 18 * M, 2 * HR + 5 * M),
        (42, "resnet-smoke", "vision", b, "failed", 40, 55 * M),
    ]:
        r = job(now, id, tag, project, session, "one", "python run.py", ran=ran)
        r.update(
            state=state,
            ended_at=iso(now - ago),
            note="exit code 1 after 40s: CUDA out of memory" if state == "failed" else None,
        )
        recent.append(r)
    items = [
        {"id": 41, "gpus": [0], "start": now - 2 * HR, "end": now + 3 * HR, "state": "running"},
        {"id": 43, "gpus": [1], "start": now - 50 * M, "end": now + 70 * M, "state": "running"},
        {"id": 45, "gpus": [], "start": now, "end": now + 25 * M, "state": "queued"},
        {"id": 44, "gpus": [1], "start": now + 70 * M, "end": now + 80 * M, "state": "queued"},
        {"id": 46, "gpus": [0], "start": now + 3 * HR, "end": now + 3 * HR + 40 * M, "state": "queued"},
        {"id": 47, "gpus": [0, 1], "start": now + 3 * HR + 40 * M, "end": now + 3 * HR + 55 * M, "state": "queued"},
    ]
    gpu = dict(power_limit_w=151, fan_pct=54, shared=False)
    return {
        "schema": 1, "mode": "normal", "mode_since": None,
        "daemon": {"running": True, "pid": 1, "heartbeat_age_s": 0.4},
        "slots": {"total": 4, "used": 4},
        "gpus": [
            {**gpu, "index": 0, "name": "NVIDIA GeForce RTX 4090", "mem_total_mb": 24564, "mem_used_mb": 7400,
             "util_pct": 97, "temp_c": 71, "power_w": 402, "power_limit_w": 450, "clock_mhz": 2610, "jobs": [41]},
            {**gpu, "index": 1, "name": "NVIDIA GeForce RTX 3090", "mem_total_mb": 24576, "mem_used_mb": 4300,
             "util_pct": 88, "temp_c": 66, "power_w": 301, "power_limit_w": 350, "fan_pct": 61, "clock_mhz": 1890,
             "shared": True, "jobs": [43]},
        ],
        "running": running, "queue": queue, "recent": recent,
        "cpu": {
            "at": now, "load_pct": 38.0, "cores_pct": [62, 41, 35, 30, 48, 22, 27, 39],
            "mem_used_mb": 21500, "mem_total_mb": 65536,
            "top": [{"name": "job 41", "pct": 14.0}, {"name": "job 43", "pct": 11.0}, {"name": "code", "pct": 3.0},
                    {"name": "firefox", "pct": 2.0}, {"name": "Xorg", "pct": 1.0}],
            "disks": [{"mount": "/", "total_mb": 976000, "used_mb": 402000, "free_mb": 574000},
                      {"mount": "/data", "total_mb": 3815000, "used_mb": 2900000, "free_mb": 915000}],
            "io": {"read_mb_s": 212.4, "write_mb_s": 18.9},
        },
        "timeline": {"now": now, "items": items, "unplaced": []},
        "fair_share": False,
    }  # fmt: skip


def history() -> str:
    """Five minutes of made-up load for the sparklines."""
    import math

    def wave(base, amp, k):
        return [round(max(0, min(100, base + amp * math.sin(i / k) + (i * 7919 % 13 - 6))), 1) for i in range(150)]

    return json.dumps({0: wave(93, 4, 9), 1: wave(80, 9, 6), "cpu": wave(36, 8, 11)})


def page(dark: bool, now: float) -> str:
    html = PAGE.read_text(encoding="utf-8")
    # Pick the theme: the dark colours sit in a prefers-color-scheme block.
    html = html.replace("@media (prefers-color-scheme: dark)", "@media all" if dark else "@media not all")
    stub = (
        f"<script>const __S = {json.dumps(status(now))};"
        "window.fetch = async () => ({ ok: true, json: async () => __S });</script>"
    )
    fill = f"<script>Object.assign(history, {history()});</script>"
    return html.replace("<script>", stub + "<script>", 1).replace(
        "</script>\n</html>", "</script>\n" + fill + "\n</html>"
    )


def browser() -> str:
    names = ["chrome", "google-chrome", "chromium", "chromium-browser", "msedge"]
    found = [shutil.which(n) for n in names]
    found += [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for f in found:
        if f and os.path.exists(f):
            return f
    sys.exit("needs Chrome, Edge or Chromium")


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/img").resolve()  # Chrome needs an absolute path
    out.mkdir(parents=True, exist_ok=True)
    exe = browser()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Chrome may still hold a file
        for theme in ("light", "dark"):
            f = Path(tmp) / f"{theme}.html"
            f.write_text(page(theme == "dark", time.time()), encoding="utf-8")
            png = out / f"gpushift-dashboard-{theme}.png"
            subprocess.run(
                [exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={tmp}/profile",
                 f"--window-size={W},{H}", "--virtual-time-budget=4000", f"--screenshot={png}", f.as_uri()],
                check=True, capture_output=True, timeout=60,
            )  # fmt: skip
            print(png)


if __name__ == "__main__":
    main()
