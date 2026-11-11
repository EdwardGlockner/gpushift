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


def for_agent(jobs: list[dict]) -> str:
    """The note added to a Claude session when its jobs end."""
    lines = []
    for j in jobs:
        title, body = job_ended(j)
        detail = "" if j["state"] == "done" else f": {body}"
        lines.append(f"- {title.removeprefix('gpushift: ')}{detail}. Output: `gpushift logs {j['id']}`")
    head = (
        "Update from the gpushift GPU queue: a job you submitted has ended."
        if len(jobs) == 1
        else (f"Update from the gpushift GPU queue: {len(jobs)} jobs you submitted have ended.")
    )
    return head + "\n" + "\n".join(lines)
