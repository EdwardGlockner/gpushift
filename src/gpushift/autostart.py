"""Start the daemon at login: HKCU Run key on Windows, a systemd user service
on Linux, a LaunchAgent on macOS. None of these need admin rights."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import _proc

NAME = "gpushift"
UNIT = Path.home() / ".config" / "systemd" / "user" / "gpushift.service"
PLIST = Path.home() / "Library" / "LaunchAgents" / "com.gpushift.daemon.plist"


def _cmd() -> list[str]:
    return [_proc.windowless_python(), "-m", "gpushift", "daemon"]


def install() -> str:
    if sys.platform == "win32":
        import winreg

        cmd = subprocess.list2cmdline(_cmd())
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0, winreg.KEY_SET_VALUE
        ) as k:
            winreg.SetValueEx(k, NAME, 0, winreg.REG_SZ, cmd)
        return f"registered at login (HKCU\\...\\Run\\{NAME}): {cmd}"
    if sys.platform == "darwin":
        args = "".join(f"<string>{a}</string>" for a in _cmd())
        PLIST.parent.mkdir(parents=True, exist_ok=True)
        PLIST.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            f'<plist version="1.0"><dict><key>Label</key><string>com.gpushift.daemon</string>'
            f"<key>ProgramArguments</key><array>{args}</array>"
            "<key>RunAtLoad</key><true/><key>KeepAlive</key><true/></dict></plist>\n"
        )
        subprocess.run(["launchctl", "load", "-w", str(PLIST)], check=False)
        return f"installed LaunchAgent {PLIST}"
    exec_start = " ".join(f'"{a}"' if " " in a else a for a in _cmd())
    UNIT.parent.mkdir(parents=True, exist_ok=True)
    UNIT.write_text(
        "[Unit]\nDescription=gpushift GPU job queue\n\n"
        f"[Service]\nExecStart={exec_start}\nRestart=on-failure\nRestartSec=5\n\n"
        "[Install]\nWantedBy=default.target\n"
    )
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    r = subprocess.run(["systemctl", "--user", "enable", "--now", "gpushift.service"], capture_output=True, text=True)
    if r.returncode != 0:
        return f"wrote {UNIT}, but enabling failed: {r.stderr.strip()}"
    return (
        f"installed systemd user service {UNIT}. To keep the queue running after you log out, "
        "run: loginctl enable-linger $USER"
    )


def uninstall() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0, winreg.KEY_SET_VALUE
            ) as k:
                winreg.DeleteValue(k, NAME)
        except FileNotFoundError:
            return "not registered"
        return "removed from login startup"
    if sys.platform == "darwin":
        if PLIST.exists():
            subprocess.run(["launchctl", "unload", "-w", str(PLIST)], check=False)
            PLIST.unlink()
            return "removed LaunchAgent"
        return "not registered"
    if UNIT.exists():
        subprocess.run(["systemctl", "--user", "disable", "--now", "gpushift.service"], check=False)
        UNIT.unlink()
        return "removed systemd user service"
    return "not registered"


def status() -> str | None:
    """Description of the registered autostart, or None."""
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as k:
                return winreg.QueryValueEx(k, NAME)[0]
        except FileNotFoundError:
            return None
    if sys.platform == "darwin":
        return str(PLIST) if PLIST.exists() else None
    return str(UNIT) if UNIT.exists() else None
