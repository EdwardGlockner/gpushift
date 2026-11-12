"""gpushift command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import __version__, autostart, config, control, daemon, db, gpus, paths
from ._proc import NO_WINDOW
from .scheduler import PRESETS, PresetError, preset_requirements


class CLIError(Exception):
    pass


# ---------------------------------------------------------------- formatting


def parse_duration(s: str) -> int:
    """'40m', '1h30m', '90s', '2h', '1.5h'; a bare number means minutes."""
    s = s.strip().lower()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return int(float(s) * 60)
    total = 0.0
    pos = 0
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([dhms])", s):
        if m.start() != pos and s[pos : m.start()].strip():
            raise CLIError(f"cannot parse duration {s!r} (examples: 40m, 1h30m, 90s)")
        total += float(m.group(1)) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[m.group(2)]
        pos = m.end()
    if pos == 0 or s[pos:].strip():
        raise CLIError(f"cannot parse duration {s!r} (examples: 40m, 1h30m, 90s)")
    return int(total)


def parse_size(s: str) -> int:
    """'6G', '6GB', '500M', '1.5g'; a bare number means GB. Returns MiB."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([gm]i?b?)?\s*", s.lower())
    if not m:
        raise CLIError(f"cannot parse memory size {s!r} (examples: 6G, 500M)")
    n = float(m.group(1))
    return int(n if (m.group(2) or "g").startswith("m") else n * 1024)


def fmt_dur(sec: float | None) -> str:
    if sec is None:
        return "?"
    sec = int(max(sec, 0))
    if sec < 60:
        return f"{sec}s"
    m, _ = divmod(sec, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d:
        return f"{d}d{h:02d}h"
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m"


def fmt_clock(ts: float | None) -> str:
    if ts is None:
        return "?"
    lt = time.localtime(ts)
    if time.strftime("%Y%m%d", lt) == time.strftime("%Y%m%d"):
        return time.strftime("%H:%M", lt)
    return time.strftime("%a %H:%M", lt)


def fmt_start(ts: float | None, mode: str) -> str:
    if mode != "normal":
        return "paused"
    if ts is None:
        return "unknown"
    rel = ts - time.time()
    if rel < 30:
        return "now"
    return f"~{fmt_dur(rel)} ({fmt_clock(ts)})"


def iso(ts: float | None) -> str | None:
    return None if ts is None else time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ts))


def trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def job_public(con, j: dict) -> dict:
    runtime = control.ran(j)
    return {
        "id": j["id"],
        "state": j["state"],
        "preset": j["preset"],
        "n_gpus": j["n_gpus"],
        "slots": j["slots"],
        "gpus": j["gpus"] or [],
        "tag": j["tag"],
        "project": j["project"],
        "session": j["session"],
        "command": j["command"],
        "cwd": j["cwd"],
        "submitted_at": iso(j["submitted_at"]),
        "started_at": iso(j["started_at"]),
        "ended_at": iso(j["ended_at"]),
        "frozen_at": iso(j["frozen_at"]),
        "runtime_s": round(runtime),
        "exit_code": j["exit_code"],
        "eta_s": j["eta_seconds"],
        "expected_duration_s": (lambda d: None if d is None else round(d))(control.expected_duration(con, j)),
        "resumable": control.is_resumable(j),
        "restartable": bool(j["restartable"]),
        "vram_mb": j["vram_mb"],
        "peak_vram_mb": j["peak_vram_mb"],
        "evictions": j["evictions"],
        "resume": bool(j["resume"]),
        "after": j.get("after_ids") or [],
        "log": str(paths.log_file(j["id"])),
        "note": j["note"],
    }


# ---------------------------------------------------------------- context


def load_cfg() -> config.Config:
    try:
        return config.load()
    except config.ConfigError as e:
        raise CLIError(f"invalid config: {e}") from e


def project_name(cwd: str) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=NO_WINDOW,
        )
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip()).name
    except (OSError, subprocess.TimeoutExpired):
        pass
    return Path(cwd).name or cwd


def git_bash() -> str | None:
    """On Windows, the bash.exe of the Git Bash (MSYS2) session gpushift was called
    from, or None. Never WSL's bash in System32, which is a different system."""
    if os.name != "nt" or not os.environ.get("MSYSTEM"):
        return None
    exe = shutil.which("bash")
    if not exe or Path(exe).parent.name.lower() in ("system32", "windowsapps"):
        return None
    return exe


def windows_style(command: str) -> bool:
    """A shell string written for cmd.exe: it has backslash paths such as
    `.venv\\Scripts\\python.exe`, which bash would read as escapes."""
    return bool(re.search(r"[\w.]\\[\w.]", command))


def check_program_name(cmd: list[str]) -> None:
    """Catch a command that was half-quoted, e.g. `"python run.py a 1" --flag`:
    the quoted part would be taken as the name of one program."""
    prog = cmd[0]
    if len(cmd) < 2 or not re.search(r"\s", prog) or os.path.exists(prog) or shutil.which(prog):
        return
    raise CLIError(
        f"the program to run is {prog!r}, which contains spaces, so gpushift would look for a program with that whole "
        "name. Pass the command as separate words (gpushift submit -- python run.py a 1 --flag) or quote all of it "
        'as one string (gpushift submit -- "python run.py a 1 --flag").'
    )


def check_runnable(cmd: list[str], cwd: str) -> None:
    """Fail at submit time, not hours later, when the program or the Python
    script does not exist."""
    prog = cmd[0]
    if not (os.path.exists(os.path.join(cwd, prog)) or shutil.which(prog)):
        raise CLIError(f"program not found: {prog!r} (not on PATH and not a file in {cwd})")
    if not re.fullmatch(r"(python(\d+(\.\d+)?)?|py|pythonw)(\.exe)?", Path(prog).name, re.I):
        return
    args = iter(cmd[1:])
    for a in args:
        if a in ("-m", "-c", "-"):
            return
        if a in ("-W", "-X"):  # options that take a value
            next(args, None)
            continue
        if a.startswith("-"):
            continue
        if a.endswith(".py") and not os.path.exists(os.path.join(cwd, a)):
            raise CLIError(f"script not found: {a!r} (relative to {cwd})")
        return


def ensure_daemon(quiet: bool = False) -> bool:
    if daemon.is_running():
        return True
    ok = daemon.start_background()
    if not quiet:
        print(
            "started the gpushift daemon"
            if ok
            else "warning: could not start the gpushift daemon; run `gpushift doctor`",
            file=sys.stderr,
        )
    return ok


def get_job_or_fail(con, job_id: int) -> dict:
    j = db.get_job(con, job_id)
    if j is None:
        raise CLIError(f"job {job_id} not found")
    return j


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2))


# ---------------------------------------------------------------- commands


def cmd_submit(a) -> int:
    cmd = list(a.cmd or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise CLIError("nothing to run. Usage: gpushift submit --one -- python train.py")
    cfg = load_cfg()
    preset = a.preset or ("one" if cfg.gpus else "cpu")
    try:
        req = preset_requirements(preset, len(cfg.gpus), cfg.slots, gpus=a.gpus, slots=a.slots)
    except PresetError as e:
        raise CLIError(str(e)) from e
    if preset == "light" and cfg.shared_gpu is None:
        raise CLIError("--light needs a shared GPU, but none is configured")

    cwd = os.getcwd()
    check_program_name(cmd)
    shell = len(cmd) == 1 and bool(re.search(r"[\s&|<>;]", cmd[0]))
    if not shell:
        check_runnable(cmd, cwd)
    command = cmd[0] if len(cmd) == 1 else (subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd))
    if shell and not windows_style(cmd[0]) and (bash := git_bash()):
        # Written for the Git Bash it was submitted from (/c/... paths, cp,
        # 2>&1); cmd.exe would misread it, so run it in that bash.
        cmd, shell = [bash, "-c", cmd[0]], False
    project = project_name(cwd)
    con = db.connect()
    after = parse_after(con, a.after)
    with db.tx(con):
        pos = db.front_position(con) if a.priority else db.back_position(con)
        jid = db.insert_job(
            con,
            state="queued",
            prio=int(a.priority),
            preset=preset,
            n_gpus=req["n_gpus"],
            slots=req["slots"],
            exclusive=int(req["exclusive"]),
            shared=int(req["shared"]),
            position=pos,
            argv=cmd,
            command=command,
            shell=int(shell),
            cwd=cwd,
            env=control.filtered_env(),
            tag=a.tag or project,
            project=project,
            session=os.environ.get("CLAUDE_CODE_SESSION_ID"),
            eta_seconds=parse_duration(a.eta) if a.eta else None,
            resumable=int(a.resumable),
            restartable=int(a.restartable),
            vram_mb=parse_size(a.vram) if a.vram else None,
            after_ids=after or None,
        )
    paths.job_dir(jid).mkdir(parents=True, exist_ok=True)
    if not a.no_start:
        ensure_daemon(quiet=a.json)

    queue, starts = control.queue_and_starts(con, cfg)
    position = next((i + 1 for i, q in enumerate(queue) if q["id"] == jid), None)
    est = starts.get(jid)
    mode = db.get_mode(con)
    if a.json:
        _print_json(
            {
                "id": jid,
                "preset": preset,
                "position": position,
                "est_start": iso(est),
                "mode": mode,
                "log": str(paths.log_file(jid)),
            }
        )
        return 0
    desc = {
        "light": "shared GPU",
        "one": "1 GPU",
        "two": f"{req['n_gpus']} GPUs",
        "exclusive": "whole machine",
        "cpu": "CPU only",
    }[preset]
    print(
        f"job {jid} queued ({preset}: {desc}, {req['slots']} slot{'s' if req['slots'] != 1 else ''}) tag={a.tag or project}"
    )
    waits = next((q["waiting_for"] for q in queue if q["id"] == jid), [])
    if waits:
        print(f"position {position} of {len(queue)}; starts after job {', '.join(map(str, waits))} has finished")
    else:
        print(f"position {position} of {len(queue)}, estimated start: {fmt_start(est, mode)}")
    if mode != "normal":
        print("note: the queue is paused (free mode); it will start after `gpushift resume`")
    print(f"logs: gpushift logs {jid} -f")
    return 0


def parse_after(con, values: list[str] | None) -> list[int]:
    """--after 12 --after 13,14 -> [12, 13, 14], checking that each job exists
    and can still finish successfully."""
    ids: list[int] = []
    for v in values or []:
        for part in v.replace(" ", "").split(","):
            if not part:
                continue
            if not part.isdigit():
                raise CLIError(f"--after takes job ids, e.g. --after 12 or --after 12,13 (got {part!r})")
            ids.append(int(part))
    for i in dict.fromkeys(ids):
        j = db.get_job(con, i)
        if j is None:
            raise CLIError(f"--after {i}: there is no job {i}")
        if j["state"] in ("failed", "cancelled"):
            raise CLIError(f"--after {i}: job {i} {j['state']}, so this job would never run")
    return list(dict.fromkeys(ids))


def timeline(con, running: list[dict], queue: list[dict], placed: dict) -> dict:
    """What runs on which GPU and when: running jobs (end = expected end or
    None) and the queued jobs the forecast can place. Times are unix seconds."""
    now = time.time()
    items = []
    for j in running:
        left = control.time_left(con, j, now)
        items.append(
            {"id": j["id"], "gpus": j["gpus"] or [], "start": j["started_at"], "end": None if left is None else now + left,
             "state": j["state"]}
        )  # fmt: skip
    for j in queue:
        if j["id"] in placed:
            start, gpus, end = placed[j["id"]]
            items.append({"id": j["id"], "gpus": gpus, "start": start, "end": end, "state": "queued"})
    unplaced = [j["id"] for j in queue if j["id"] not in placed]
    return {"now": now, "items": items, "unplaced": unplaced}


def status_data(con, cfg) -> dict:
    mode = db.get_mode(con)
    running = db.active(con)
    placed: dict = {}
    queue, est = control.queue_and_starts(con, cfg, placed)
    live = {g.index: g for g in control.gpu_readings(con)}
    gpu_rows = []
    for i in cfg.gpus:
        g = live.get(i) or next((d for d in cfg.detected if d.index == i), None)
        gpu_rows.append(
            {
                "index": i,
                "name": g.name if g else None,
                "mem_total_mb": g.mem_total if g else None,
                "mem_used_mb": g.mem_used if g else None,
                "util_pct": g.util if g else None,
                "temp_c": g.temp if g else None,
                "power_w": g.power if g else None,
                "power_limit_w": g.power_limit if g else None,
                "fan_pct": g.fan if g else None,
                "clock_mhz": g.clock if g else None,
                "shared": i == cfg.shared_gpu,
                "jobs": [j["id"] for j in running if i in (j["gpus"] or [])],
            }
        )
    hb = db.get_kv(con, "daemon_heartbeat")
    q = []
    for n, j in enumerate(queue, 1):
        d = job_public(con, j)
        d["position"] = n
        d["turn"] = j.get("turn", n)  # place in queue order (est_start order can differ)
        d["waiting_for"] = j.get("waiting_for", [])
        d["est_start"] = iso(est.get(j["id"]))
        q.append(d)
    return {
        "schema": 1,
        "mode": mode,
        "mode_since": iso(db.get_kv(con, "mode_since")) if mode != "normal" else None,
        "auto_free": db.get_kv(con, "auto_free") if mode != "normal" else None,
        "daemon": {
            "running": daemon.is_running(),
            "pid": db.get_kv(con, "daemon_pid"),
            "heartbeat_age_s": None if hb is None else round(time.time() - hb, 1),
        },
        "slots": {"total": cfg.slots, "used": sum(j["slots"] for j in running)},
        "gpus": gpu_rows,
        "running": [job_public(con, j) for j in running],
        "queue": q,
        "recent": [job_public(con, j) for j in db.recent(con, 5)],
        "cpu": control.cpu_reading(con),
        "timeline": timeline(con, running, queue, placed),
        "_est": est,
    }


def cmd_status(a) -> int:
    cfg = load_cfg()
    con = db.connect()
    s = status_data(con, cfg)
    est = s.pop("_est")
    if a.json:
        _print_json(s)
        return 0
    mode = s["mode"]
    mode_txt = {
        "normal": "normal",
        "free": "FREE (queue paused, machine freed)",
        "soft": "FREE --soft (queue paused, jobs frozen)",
    }[mode]
    if mode != "normal" and s.get("auto_free"):
        apps = ", ".join(s["auto_free"].get("apps") or [])
        mode_txt = f"FREE (automatic: {apps} is using the GPU; resumes after it closes)"
    dm = "running" if s["daemon"]["running"] else "NOT RUNNING (`gpushift start`)"
    print(f"gpushift  mode: {mode_txt}  |  daemon: {dm}  |  slots: {s['slots']['used']}/{s['slots']['total']} used")
    print()
    by_id = {j["id"]: j for j in s["running"]}

    def job_line(j):
        st = "" if j["state"] == "running" else f" [{j['state'].upper()}]"
        exp = j["expected_duration_s"]
        if j["state"] == "running" and exp and j["runtime_s"] > exp:
            st = f" [over its {fmt_dur(exp)} estimate]"
        return f"job {j['id']:<4} {j['preset']:<9} {trunc(j['tag'] or '', 14):<14} {fmt_dur(j['runtime_s']):>6}{st}  {trunc(j['command'], 50)}"

    for g in s["gpus"]:
        mem = f"{g['mem_used_mb'] / 1024:.1f}/{g['mem_total_mb'] / 1024:.1f} GB" if g["mem_total_mb"] else ""
        name = trunc((g["name"] or "").replace("NVIDIA ", "").replace("GeForce ", ""), 16)
        label = f"GPU {g['index']} {name:<16} {mem:<13}" + ("(shared) " if g["shared"] else "         ")
        if not g["jobs"]:
            print(f"{label}idle")
        for k, jid in enumerate(g["jobs"]):
            print(f"{label if k == 0 else ' ' * len(label)}{job_line(by_id[jid])}")
    cpu_jobs = [j for j in s["running"] if not j["gpus"]]
    label = f"{'CPU':<47}"
    if cpu_jobs or not s["gpus"]:
        if not cpu_jobs:
            print(f"{label}idle")
        for k, j in enumerate(cpu_jobs):
            print(f"{label if k == 0 else ' ' * len(label)}{job_line(j)}")
    print()
    if s["queue"]:
        print(f"Queue ({len(s['queue'])} waiting)")
        print(f"  {'pos':>3}  {'id':<4} {'preset':<9} {'tag':<14} {'waiting':>7}  {'est. start':<20} command")
        raw = {q["id"]: q for q in db.queued(con)}
        for j in s["queue"]:
            waited = time.time() - raw[j["id"]]["submitted_at"] if j["id"] in raw else None
            flag = " (resume)" if j["resume"] else ""
            start = fmt_start(est.get(j["id"]), mode)
            if j["waiting_for"] and est.get(j["id"]) is None:
                start = "after job " + ",".join(map(str, j["waiting_for"]))
            print(
                f"  {j['position']:>3}  {j['id']:<4} {j['preset']:<9} {trunc(j['tag'] or '', 14):<14} {fmt_dur(waited):>7}  "
                f"{trunc(start, 20):<20} {trunc(j['command'], 40)}{flag}"
            )
    else:
        print("Queue empty")
    if s["recent"] and not a.short:
        print()
        print("Recent")
        raw = {r["id"]: r for r in db.recent(con, 5)}
        for j in s["recent"]:
            ended = raw[j["id"]]["ended_at"] if j["id"] in raw else None
            rc = "" if j["exit_code"] is None else f" (exit {j['exit_code']})"
            print(
                f"  job {j['id']:<4} {j['state']:<9}{rc:<9} {trunc(j['tag'] or '', 14):<14} ran {fmt_dur(j['runtime_s']):>6}, "
                f"ended {fmt_dur(time.time() - ended) if ended else '?'} ago  {trunc(j['command'], 36)}"
            )
    return 0


def cmd_show(a) -> int:
    con = db.connect()
    j = job_public(con, get_job_or_fail(con, a.id))
    if a.json:
        _print_json(j)
    else:
        for k, v in j.items():
            print(f"{k:>20}: {v}")
    return 0


def cmd_priority(a) -> int:
    con = db.connect()
    get_job_or_fail(con, a.id)
    msg = control.to_front(con, a.id)
    if "moved" not in msg:
        raise CLIError(msg)
    print(msg)
    return 0


def cmd_swap(a) -> int:
    con = db.connect()
    with db.tx(con):
        x, y = get_job_or_fail(con, a.a), get_job_or_fail(con, a.b)
        for j in (x, y):
            if j["state"] != "queued":
                raise CLIError(f"job {j['id']} is {j['state']}, not queued")
        queue = db.queued(con)
        order = [q["id"] for q in queue]
        i, k = order.index(x["id"]), order.index(y["id"])
        order[i], order[k] = order[k], order[i]
        problem = control.order_problem(queue, order)
        if problem:
            raise CLIError(problem)
        db.update(con, x["id"], position=y["position"], prio=y["prio"])
        db.update(con, y["id"], position=x["position"], prio=x["prio"])
    print(f"swapped jobs {a.a} and {a.b}")
    return 0


def cmd_cancel(a) -> int:
    con = db.connect()
    rc = 0
    for jid in a.ids:
        msg = control.cancel(con, jid)
        print(msg)
        if "not found" in msg:
            rc = 1
    return rc


def cmd_retry(a) -> int:
    con = db.connect()
    with db.tx(con):
        j = get_job_or_fail(con, a.id)
        keep = (
            "preset",
            "n_gpus",
            "slots",
            "exclusive",
            "shared",
            "argv",
            "command",
            "shell",
            "cwd",
            "env",
            "tag",
            "project",
            "session",
            "eta_seconds",
            "resumable",
            "restartable",
            "vram_mb",
        )
        jid = db.insert_job(
            con,
            state="queued",
            position=db.front_position(con) if a.priority else db.back_position(con),
            prio=int(a.priority),
            resume=int(a.resume),
            **{k: j[k] for k in keep},
        )
    paths.job_dir(jid).mkdir(parents=True, exist_ok=True)
    ensure_daemon()
    print(f"job {jid} queued (retry of job {a.id})")
    return 0


def _final(con, jid) -> dict | None:
    j = db.get_job(con, jid)
    return j if j and j["state"] in db.FINAL else None


def _emit(data: bytes) -> None:
    """Write job output. Logs written on Windows contain CRLF; normalize so the
    text-mode stdout does not turn them into CR CR LF."""
    if data:
        sys.stdout.write(data.replace(b"\r\n", b"\n").decode(errors="replace"))
        sys.stdout.flush()


def cmd_logs(a) -> int:
    con = db.connect()
    get_job_or_fail(con, a.id)
    f = paths.log_file(a.id)
    while not f.exists():
        if not a.follow or _final(con, a.id):
            print(f"(no output yet for job {a.id})")
            return 0
        time.sleep(0.5)
    with open(f, "rb") as fh:
        if a.tail:
            _emit(b"".join(fh.read().splitlines(keepends=True)[-a.tail :]))
        else:
            _emit(fh.read())
        if not a.follow:
            return 0
        try:
            while True:
                chunk = fh.read()
                if chunk:
                    _emit(chunk)
                elif _final(con, a.id):
                    _emit(fh.read())
                    return 0
                else:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            return 0


def cmd_wait(a) -> int:
    con = db.connect()
    get_job_or_fail(con, a.id)
    deadline = None if a.timeout is None else time.time() + parse_duration(a.timeout)
    while True:
        j = _final(con, a.id)
        if j:
            ev = j["evictions"] or 0
            extra = (
                f"; it was evicted {ev}x when the user freed the GPUs and resumed from its own checkpoint" if ev else ""
            )
            print(f"job {a.id} {j['state']} (exit code {j['exit_code']}{extra})")
            if j["state"] == "failed" and j["note"]:
                print(j["note"])
            with db.tx(con):  # the caller now knows; no need to notify the session again
                db.update(con, a.id, notified=1)
            return 0 if j["state"] == "done" else (j["exit_code"] or 1)
        if deadline and time.time() > deadline:
            print(f"job {a.id} still {db.get_job(con, a.id)['state']} (timed out waiting)")
            return 124
        time.sleep(1)


def _vram_summary(cfg) -> str:
    try:
        live = [g for g in gpus.query() if g.index in cfg.gpus]
    except gpus.GPUQueryError:
        return "(VRAM unknown)"
    return ", ".join(f"GPU {g.index} {g.mem_used / 1024:.1f}/{g.mem_total / 1024:.1f} GB" for g in live) or "(no GPUs)"


def cmd_free(a) -> int:
    cfg = load_cfg()
    con = db.connect()
    soft = a.soft or (cfg.free_mode_default == "soft" and not a.full)
    rep = control.free(con, soft)
    kind = "soft: everything frozen, VRAM stays allocated" if soft else "full"
    if rep["previous_mode"] != "normal":
        print(f"already in free mode ({rep['previous_mode']}); now {rep['mode']}")
    print(f"Free mode ON ({kind}). The queue is paused; nothing new will start.")
    by = {j["id"]: j for j in db.active(con)}

    def tag(jid):
        j = by.get(jid) or db.get_job(con, jid)
        return f"job {jid} ({j['tag']})"

    for jid in rep["evicting"]:
        print(f"  evicting:   {tag(jid)} - asked to checkpoint and exit")
    for jid in rep["restarting"]:
        print(f"  restarting: {tag(jid)} - stopped; it runs again from the front of the queue")
    for jid in rep["frozen"]:
        why = "" if soft else " (it can't checkpoint and isn't --restartable)"
        print(f"  frozen:     {tag(jid)} - paused, keeps its GPU memory{why}")
    for jid in rep["pending"]:
        print(f"  frozen:     {tag(jid)} - was just starting; it freezes itself")
    if not (rep["evicting"] or rep["restarting"] or rep["frozen"] or rep["pending"]):
        print("  no jobs were running")

    leaving = rep["evicting"] + rep["restarting"]
    if leaving and not a.no_wait:
        print(
            f"Waiting for jobs to leave the GPUs (up to {cfg.evict_timeout}s; Ctrl-C stops waiting, they still leave)..."
        )
        waiting = set(leaving)
        try:
            while waiting:
                time.sleep(1)
                control.enforce_evict_timeout(con, cfg.evict_timeout)
                for jid in list(waiting):
                    j = db.get_job(con, jid)
                    if j["state"] == "queued" and jid in rep["restarting"]:
                        print(f"  job {jid} stopped -> back at the front of the queue; it runs again on resume")
                        waiting.discard(jid)
                    elif j["state"] == "queued":
                        print(f"  job {jid} checkpointed and exited -> back at the front of the queue")
                        waiting.discard(jid)
                    elif j["state"] == "frozen":
                        print(f"  job {jid} did not exit within {cfg.evict_timeout}s -> frozen instead (VRAM held)")
                        waiting.discard(jid)
                    elif j["state"] in db.FINAL:
                        print(f"  job {jid} ended ({j['state']})")
                        waiting.discard(jid)
        except KeyboardInterrupt:
            print("  (stopped waiting)")
    print(f"VRAM in use now: {_vram_summary(cfg)}")
    _explain_frozen(con, cfg)
    print("Run `gpushift resume` to continue.")
    return 0


def _explain_frozen(con, cfg) -> None:
    """Say which GPUs frozen jobs keep memory on, and how to release it."""
    frozen = [j for j in db.active(con) if j["state"] == "frozen" and j["gpus"]]
    if not frozen:
        return
    try:
        live = {g.index: g for g in gpus.query()}
    except gpus.GPUQueryError:
        live = {}
    print("Frozen jobs keep their GPU memory:")
    for g in sorted({g for j in frozen for g in j["gpus"]}):
        mem = f" ({live[g].mem_used / 1024:.1f} GB in use)" if g in live else ""
        names = ", ".join(f"job {j['id']} ({j['tag']})" for j in frozen if g in j["gpus"])
        print(f"  GPU {g}{mem}: {names}")
    ids = " ".join(str(j["id"]) for j in frozen)
    print(f"To release it: gpushift cancel {ids}   (run it again later with: gpushift retry <id> --priority)")


def cmd_resume(a) -> int:
    con = db.connect()
    rep = control.resume(con)
    if rep["previous_mode"] == "normal" and not rep["resumed"]:
        print("not in free mode; nothing to resume")
        return 0
    print("Free mode OFF. The queue is running again.")
    for jid in rep["resumed"]:
        print(f"  resumed: job {jid}")
    for jid in rep["evicting"]:
        print(f"  job {jid} is still checkpointing; it will restart from the front of the queue")
    evicted = [j for j in db.queued(con) if j["resume"]]
    for j in evicted:
        print(f"  queued:  job {j['id']} will restart from its checkpoint")
    ensure_daemon()
    return 0


def cmd_start(a) -> int:
    if daemon.is_running():
        print("daemon already running")
        return 0
    if ensure_daemon(quiet=True):
        print("daemon started")
        return 0
    raise CLIError(f"daemon did not start; see {paths.daemon_log()}")


def cmd_stop(a) -> int:
    if daemon.stop():
        print("daemon stopped (running jobs keep running; queued jobs wait until it is started again)")
    else:
        print("daemon was not running")
    return 0


def cmd_daemon(a) -> int:
    return daemon.main()


def cmd_config(a) -> int:
    f = paths.config_file()
    if a.path:
        print(f)
        return 0
    if a.edit:
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(config.template(gpus.detect()))
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or ("notepad" if os.name == "nt" else "vi")
        return subprocess.call([*shlex.split(editor, posix=os.name != "nt"), str(f)])
    cfg = load_cfg()
    print(f"# {f}{'' if f.exists() else ' (not created yet; defaults shown)'}")
    if a.json:
        _print_json(cfg.to_dict())
    else:
        for k, v in cfg.to_dict().items():
            print(f"{k} = {v}")
    return 0


def cmd_init(a) -> int:
    print(f"gpushift {__version__} on {sys.platform}, python {sys.executable}")
    try:
        detected = gpus.query()
    except gpus.GPUQueryError as e:
        print(f"warning: {e}; continuing in CPU-only mode")
        detected = []
    if detected:
        for g in detected:
            print(f"  GPU {g.index}: {g.name}, {g.mem_total / 1024:.1f} GB ({g.mem_used / 1024:.1f} GB in use)")
    else:
        print("  no NVIDIA GPUs found -> CPU-only mode (only --cpu/--exclusive jobs)")
    f = paths.config_file()
    if f.exists() and not a.force:
        print(f"config: keeping existing {f}")
    else:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(config.template(detected))
        print(f"config: wrote {f}")
    cfg = load_cfg()
    for g in cfg.detected:
        if g.free_perc < cfg.gpu_free_perc:
            print(
                f"  note: GPU {g.index} is only {g.free_perc:.0f}% free right now (display/other apps); jobs wait until "
                f">= {cfg.gpu_free_perc}% is free. Lower gpu_free_perc in the config if this is normal for it."
            )
    db.connect()
    print(f"state: {paths.state_dir()}")
    if not a.no_autostart:
        try:
            print(f"autostart: {autostart.install()}")
        except Exception as e:
            print(f"autostart: failed ({e}); start the daemon manually with `gpushift start`")
    print("daemon: running" if ensure_daemon(quiet=True) else f"daemon: FAILED to start (see {paths.daemon_log()})")
    print(f"\nReady: {len(cfg.gpus)} GPU(s), {cfg.slots} slots, shared GPU for --light: {cfg.shared_gpu}.")
    print("Try: gpushift submit --cpu -- python -c \"print('hello from gpushift')\"   then   gpushift status")
    return 0


def cmd_doctor(a) -> int:
    problems = 0

    def check(ok: bool | None, label: str, detail: str = "") -> None:
        nonlocal problems
        tag = {True: "ok  ", False: "FAIL", None: "warn"}[ok]
        if ok is False:
            problems += 1
        print(f"[{tag}] {label}{(': ' + detail) if detail else ''}")

    check(True, "gpushift", f"{__version__}, python {sys.version.split()[0]} at {sys.executable}")
    smi = gpus.nvidia_smi_path()
    try:
        found = gpus.query()
        check(
            True if found else None,
            "GPUs",
            ", ".join(f"{g.index}:{g.name}" for g in found) or "none found (CPU-only mode)",
        )
    except gpus.GPUQueryError as e:
        check(False, "GPUs", str(e))
    if not smi and not os.environ.get("GPUSHIFT_FAKE_GPUS"):
        check(None, "nvidia-smi", "not found; install the NVIDIA driver to use GPUs")
    f = paths.config_file()
    try:
        cfg = config.load()
        check(
            True,
            "config",
            f"{f if f.exists() else 'defaults (no config file; run gpushift init)'}; gpus={cfg.gpus} slots={cfg.slots} shared_gpu={cfg.shared_gpu}",
        )
        for g in cfg.detected:
            if g.free_perc < cfg.gpu_free_perc:
                check(
                    None,
                    f"GPU {g.index} memory",
                    f"only {g.free_perc:.0f}% free (< gpu_free_perc={cfg.gpu_free_perc}); new jobs will not start on it",
                )
    except config.ConfigError as e:
        check(False, "config", str(e))
    try:
        con = db.connect()
        n = con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        check(True, "database", f"{paths.db_file()} ({n} jobs)")
        hb = db.get_kv(con, "daemon_heartbeat")
        running = daemon.is_running()
        if running:
            age = time.time() - hb if hb else None
            check(
                age is not None and age < 15,
                "daemon",
                f"running, last heartbeat {fmt_dur(age)} ago" if age is not None else "running, no heartbeat yet",
            )
        else:
            check(False, "daemon", "not running; start it with `gpushift start`")
        mode = db.get_mode(con)
        check(
            True if mode == "normal" else None,
            "mode",
            mode + ("" if mode == "normal" else " (queue paused; `gpushift resume`)"),
        )
    except Exception as e:
        check(False, "database", str(e))
    a_st = autostart.status()
    check(True if a_st else None, "autostart", a_st or "not registered (`gpushift init` registers it)")
    print("\nall good" if problems == 0 else f"\n{problems} problem(s)")
    return 0 if problems == 0 else 1


def cmd_hook(a) -> int:
    from . import hook

    return hook.main(a.event)


def cmd_autostart(a) -> int:
    print(autostart.install() if a.action == "on" else autostart.uninstall())
    return 0


def cmd_ui(a) -> int:
    cfg = load_cfg()
    if not cfg.ui_port:
        raise CLIError("the dashboard is turned off (ui_port = 0 in the config)")
    ensure_daemon()
    url = f"http://127.0.0.1:{cfg.ui_port}/"
    print(f"gpushift dashboard: {url}")
    if not a.no_open:
        import webbrowser

        webbrowser.open(url)
    return 0


# ---------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpushift", description="Shared GPU job queue. Submit jobs instead of launching them directly."
    )
    p.add_argument("--version", action="version", version=f"gpushift {__version__}")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    s = sub.add_parser(
        "submit",
        help="queue a job: gpushift submit --one -- python train.py",
        description="Queue a job. Presets: --light (share a GPU), --one (1 GPU), --two (all GPUs), "
        "--exclusive (whole machine), --cpu (no GPU).",
    )
    g = s.add_mutually_exclusive_group()
    for name in PRESETS:
        g.add_argument(f"--{name}", dest="preset", action="store_const", const=name)
    s.add_argument("--gpus", type=int, help="override the number of GPUs")
    s.add_argument("--slots", type=int, help="override the number of slots")
    s.add_argument("--tag", help="label (default: project/repo name)")
    s.add_argument("--eta", help="expected runtime, e.g. 40m, 2h, 1h30m")
    s.add_argument(
        "--resumable",
        action="store_true",
        help="job checkpoints on SIGTERM/stop file (auto-detected with gpushift.checkpoint)",
    )
    s.add_argument(
        "--vram",
        help="GPU memory the job needs per GPU, e.g. 6G or 500M. Jobs with a known need may share a GPU "
        "(gpushift also learns it from earlier runs of the same command)",
    )
    s.add_argument(
        "--restartable",
        action="store_true",
        help="safe to stop and run again (e.g. it skips work it already finished): `gpushift free` stops it "
        "and requeues it at the front instead of freezing it",
    )
    s.add_argument(
        "--priority", action="store_true", help="put the job at the front of the queue (only when the user asks)"
    )
    s.add_argument(
        "--after",
        action="append",
        metavar="JOB",
        help="start only after these jobs finished successfully, e.g. --after 12 or --after 12,13 "
        "(if one fails or is cancelled, this job is cancelled)",
    )
    s.add_argument("--no-start", action="store_true", help="do not auto-start the daemon")
    s.add_argument("--json", action="store_true")
    s.add_argument("cmd", nargs=argparse.REMAINDER, help="command to run (after --)")
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("status", aliases=["st", "queue", "q"], help="show GPUs, running jobs and the queue")
    s.add_argument("--json", action="store_true", help="stable machine-readable output (schema 1)")
    s.add_argument("--short", action="store_true", help="omit recent jobs")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("show", help="show one job's details")
    s.add_argument("id", type=int)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("priority", help="move a queued job to the front")
    s.add_argument("id", type=int)
    s.set_defaults(func=cmd_priority)

    s = sub.add_parser("swap", help="swap two queued jobs")
    s.add_argument("a", type=int)
    s.add_argument("b", type=int)
    s.set_defaults(func=cmd_swap)

    s = sub.add_parser("cancel", help="cancel queued or running jobs")
    s.add_argument("ids", type=int, nargs="+")
    s.set_defaults(func=cmd_cancel)

    s = sub.add_parser("retry", help="queue a copy of a finished/failed job")
    s.add_argument("id", type=int)
    s.add_argument("--priority", action="store_true")
    s.add_argument("--resume", action="store_true", help="set GPUSHIFT_RESUME=1 (continue from checkpoint)")
    s.set_defaults(func=cmd_retry)

    s = sub.add_parser("logs", help="print a job's output")
    s.add_argument("id", type=int)
    s.add_argument("-f", "--follow", action="store_true", help="keep printing until the job ends")
    s.add_argument("-n", "--tail", type=int, help="only the last N lines")
    s.set_defaults(func=cmd_logs)

    s = sub.add_parser("wait", help="block until a job ends; exit code mirrors the job")
    s.add_argument("id", type=int)
    s.add_argument("--timeout", help="give up after this long (e.g. 30m)")
    s.set_defaults(func=cmd_wait)

    s = sub.add_parser("free", help="pause the queue and free the GPUs (user only)")
    m = s.add_mutually_exclusive_group()
    m.add_argument("--soft", action="store_true", help="freeze everything (fast; VRAM stays allocated)")
    m.add_argument("--full", action="store_true", help="evict resumable jobs, freeze the rest (default)")
    s.add_argument("--no-wait", action="store_true", help="do not wait for evicted jobs to exit")
    s.set_defaults(func=cmd_free)

    s = sub.add_parser("resume", help="leave free mode: unfreeze jobs and restart the queue (user only)")
    s.set_defaults(func=cmd_resume)

    s = sub.add_parser("init", help="detect GPUs, write config, start the daemon, register autostart")
    s.add_argument("--force", action="store_true", help="overwrite an existing config")
    s.add_argument("--no-autostart", action="store_true")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("doctor", help="check the installation")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("config", help="show or edit the config")
    s.add_argument("--edit", action="store_true")
    s.add_argument("--path", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_config)

    s = sub.add_parser("start", help="start the daemon")
    s.set_defaults(func=cmd_start)
    s = sub.add_parser("stop", help="stop the daemon (jobs keep running)")
    s.set_defaults(func=cmd_stop)
    s = sub.add_parser("autostart", help="register/unregister the daemon at login")
    s.add_argument("action", choices=["on", "off"])
    s.set_defaults(func=cmd_autostart)
    s = sub.add_parser("ui", aliases=["dashboard"], help="open the web dashboard in your browser")
    s.add_argument("--no-open", action="store_true", help="only print the address")
    s.set_defaults(func=cmd_ui)

    s = sub.add_parser("daemon", help=argparse.SUPPRESS)
    s.set_defaults(func=cmd_daemon)
    s = sub.add_parser("hook", help=argparse.SUPPRESS)
    s.add_argument("event", nargs="?", default="pretooluse")
    s.set_defaults(func=cmd_hook)
    return p


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    # `gpushift submit [opts] -- cmd...`: split before argparse sees the command.
    cmd = None
    if argv and argv[0] == "submit" and "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1 :]
    p = build_parser()
    a = p.parse_args(argv)
    if cmd is not None:
        a.cmd = cmd
    if not getattr(a, "func", None):
        p.print_help()
        sys.exit(0)
    try:
        rc = a.func(a)
    except CLIError as e:
        print(f"gpushift: error: {e}", file=sys.stderr)
        rc = 2
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)
