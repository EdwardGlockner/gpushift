"""Desktop notifications (Windows toast, notify-send, macOS Notification
Center), and the messages that tell a Claude session its job ended.

``GPUSHIFT_NOTIFY_LOG=<file>`` writes desktop notifications to that file instead
of showing them (tests use it).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from ._proc import NO_WINDOW, spawn_detached

_PS_TOAST = r"""
$null = [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime]
$x = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$t = $x.GetElementsByTagName('text')
$null = $t.Item(0).AppendChild($x.CreateTextNode($env:GPUSHIFT_TOAST_TITLE))
$null = $t.Item(1).AppendChild($x.CreateTextNode($env:GPUSHIFT_TOAST_BODY))
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show([Windows.UI.Notifications.ToastNotification]::new($x))
"""


def desktop(title: str, body: str) -> None:
    """Show a desktop notification. Never raises."""
    log = os.environ.get("GPUSHIFT_NOTIFY_LOG")
    try:
        if log:
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"{title} | {body}\n")
            return
        if sys.platform == "win32":
            env = dict(os.environ, GPUSHIFT_TOAST_TITLE=title, GPUSHIFT_TOAST_BODY=body)
            spawn_detached(["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_TOAST], env=env)
        elif sys.platform == "darwin":
            script = "on run argv\ndisplay notification (item 2 of argv) with title (item 1 of argv)\nend run"
            spawn_detached(["osascript", "-e", script, title, body])
        elif shutil.which("notify-send"):
            subprocess.Popen(["notify-send", "--app-name=gpushift", title, body], creationflags=NO_WINDOW)
    except OSError:
        pass


def job_ended(j: dict) -> tuple[str, str]:
    """(title, body) describing a finished job."""
    what = f"job {j['id']} ({j['tag']})" if j.get("tag") else f"job {j['id']}"
    if j["state"] == "done":
        return f"gpushift: {what} finished", j["command"][:120]
    if j["state"] == "cancelled":
        return f"gpushift: {what} was cancelled", j["command"][:120]
    return f"gpushift: {what} failed", (j.get("note") or f"exit code {j.get('exit_code')}")[:200]


QUICK_SHARE = 0.1  # a job done in less than this share of its --eta probably did little


def _dur(sec: float) -> str:
    sec = int(max(sec, 0))
    if sec < 60:
        return f"{sec}s"
    h, m = divmod(sec // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def quick_finish(j: dict) -> str | None:
    """A warning when a job succeeded far sooner than its --eta: a script that
    skipped its work or stopped early also exits with 0."""
    eta, ran = j.get("eta_seconds"), j.get("total_runtime") or 0.0
    if j["state"] != "done" or not eta or ran >= QUICK_SHARE * eta:
        return None
    return (
        f"it ran only {_dur(ran)} of its {_dur(eta)} --eta. Check its log that it really did the work "
        "(it may have skipped it, found nothing to do, or stopped early and still exited with 0) "
        "before you use its results"
    )


IDLE_CORES = 0.05  # a job using less CPU than this (in cores) is doing nothing


def stall_text(info: dict, cores: float | None = None) -> str:
    """Why a running job looks stuck (info: control.stall), and whether it
    still uses the CPU (cores: CPU cores its processes use, if measured)."""
    times = info["ran_s"] / info["expected_s"]
    t = (
        f"it has run {_dur(info['ran_s'])}, {times:.0f}x its ~{_dur(info['expected_s'])} estimate, "
        f"and printed nothing for {_dur(info['quiet_s'])}"
    )
    if cores is None:
        return t
    if cores < IDLE_CORES:
        return t + "; it uses no CPU, so it is probably hung"
    return t + f"; it still keeps {cores:.1f} CPU cores busy, so it may be very slow rather than hung"


def stall_for_agent(jobs: list[dict]) -> str:
    """The note added to a Claude session when one of its jobs looks stuck."""
    lines = []
    for j in jobs:
        what = f"job {j['id']} ({j['tag']})" if j.get("tag") else f"job {j['id']}"
        lines.append(f"- {what}: {j['stall_note']}.")
    return (
        "Warning from the gpushift GPU queue: "
        + ("a job you submitted looks stuck" if len(jobs) == 1 else f"{len(jobs)} jobs you submitted look stuck")
        + ".\n"
        + "\n".join(lines)
        + "\nCheck its output with `gpushift logs <id>`. If it is stuck (or its settings make it far too slow), "
        "cancel it with `gpushift cancel <id>` so the queue can move on; gpushift never cancels it by itself."
    )


def after_clear(jobs: list[dict]) -> str:
    """The note added to a session started by /clear: the jobs the
    conversation before it submitted are now its own."""
    ended = [j for j in jobs if j["state"] in ("done", "failed", "cancelled")]
    open_ = [j for j in jobs if j not in ended]
    parts = []
    if open_:
        lines = []
        for j in open_:
            what = f"job {j['id']} ({j['tag']})" if j.get("tag") else f"job {j['id']}"
            lines.append(f"- {what}, {j['state']}: `{j['command'][:160]}`")
        parts.append(
            "Note from the gpushift GPU queue: before /clear, this conversation submitted "
            + ("a job that is" if len(open_) == 1 else f"{len(open_)} jobs that are")
            + " still queued or running. "
            + ("It is" if len(open_) == 1 else "They are")
            + " yours now, and you will be told when "
            + ("it ends" if len(open_) == 1 else "each one ends")
            + ".\n"
            + "\n".join(lines)
            + "\nIf you are waiting on "
            + ("it" if len(open_) == 1 else "them")
            + ", run `gpushift wait` in the background so you are woken up when the next one ends."
        )
    if ended:
        parts.append(
            "Before /clear, this conversation also submitted jobs that have ended since:\n"
            + "\n".join(for_agent(ended).splitlines()[1:])
        )
    return "\n\n".join(parts)


def for_agent(jobs: list[dict]) -> str:
    """The note added to a Claude session when its jobs end."""
    lines = []
    for j in jobs:
        title, body = job_ended(j)
        detail = "" if j["state"] == "done" else f": {body}"
        quick = quick_finish(j)
        detail += f". Warning: {quick}" if quick else ""
        lines.append(f"- {title.removeprefix('gpushift: ')}{detail}. Output: `gpushift logs {j['id']}`")
    head = (
        "Update from the gpushift GPU queue: a job you submitted has ended."
        if len(jobs) == 1
        else (f"Update from the gpushift GPU queue: {len(jobs)} jobs you submitted have ended.")
    )
    return head + "\n" + "\n".join(lines)
