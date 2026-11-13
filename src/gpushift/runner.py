"""Job runner: ``python -m gpushift.runner <job_id>``, started by the daemon.

Runs one job's command with its saved cwd/env, streams output to the job log,
and records the outcome in the database. It is a separate process so that
exit codes are recorded even if the daemon restarts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

from . import _proc, control, db, gpus, paths


def _vendor(con) -> str:
    readings = control.gpu_readings(con)
    return readings[0].vendor if readings else "nvidia"


def _job_env(j: dict, vendor: str = "nvidia") -> dict[str, str]:
    env = dict(j["env"])
    jid = j["id"]
    env["GPUSHIFT_JOB_ID"] = str(jid)
    env["GPUSHIFT_JOB_DIR"] = str(paths.job_dir(jid))
    env["GPUSHIFT_STOP_FILE"] = str(paths.stop_file(jid))
    env["GPUSHIFT_ATTEMPT"] = str(j["attempts"])
    env["GPUSHIFT_RESUME"] = "1" if j["resume"] else "0"
    env["GPUSHIFT_GPUS"] = ",".join(map(str, j["gpus"] or []))
    if j["gpus"] or j["preset"] == "cpu" or j["n_gpus"] == 0:
        gpus.visible_env(env, j["gpus"] or [], vendor)
    env.setdefault("PYTHONUNBUFFERED", "1")
    lib = str(paths.pylib_dir())
    pp = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{pp}{os.pathsep}{lib}" if pp else lib
    return env


def _popen_args(j: dict, env: dict) -> tuple[list[str] | str, bool]:
    argv = list(j["argv"])
    if j["shell"]:
        return (argv[0] if len(argv) == 1 else subprocess.list2cmdline(argv)), True
    # On Windows, CreateProcess searches the *parent's* PATH, not env's.
    path = env.get("PATH") or env.get("Path")
    exe = shutil.which(argv[0], path=path)
    if exe:
        argv[0] = exe
        if os.name == "nt" and exe.lower().endswith((".bat", ".cmd")):
            return subprocess.list2cmdline(argv), True
    return argv, False


def failure_note(log, rc: int, seconds: float) -> str:
    """'exit code 2 after 0s: <last line of output>', so a failure explains
    itself in `status`, notifications and the dashboard."""
    last = ""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            lines = [ln.strip() for ln in f.read().decode(errors="replace").splitlines()]
        last = next((ln for ln in reversed(lines) if ln and not ln.startswith("=== ")), "")
    except OSError:
        pass
    note = f"exit code {rc} after {seconds:.0f}s"
    return f"{note}: {last[:200]}" if last else note


def run(job_id: int) -> int:
    con = db.connect()
    me_ctime = _proc.create_time(os.getpid())
    log = paths.log_file(job_id)
    log.parent.mkdir(parents=True, exist_ok=True)
    control.clear_stop(job_id)

    # Pre-flight: the job may have been cancelled, or free mode switched on,
    # between scheduling and now.
    with db.tx(con):
        j = db.get_job(con, job_id)
        if j is None or j["state"] != "running":
            return 1
        if j["cancel_requested"]:
            db.update(con, job_id, state="cancelled", ended_at=time.time(), note="cancelled before start")
            return 0
        if db.get_mode(con) != "normal":
            control.requeue_front(con, job_id, resume=False, note="free mode started before the job did")
            return 0
        db.update(con, job_id, runner_pid=os.getpid(), runner_ctime=me_ctime, attempts=j["attempts"] + 1)
        j = db.get_job(con, job_id)

    env = _job_env(j, _vendor(con))
    args, shell = _popen_args(j, env)
    started = time.time()
    with open(log, "ab", buffering=0) as out:
        hdr = (
            f"=== gpushift job {job_id} attempt {j['attempts']} | {time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"preset={j['preset']} gpus={j['gpus'] or []} resume={bool(j['resume'])}\n"
            f"=== cwd: {j['cwd']}\n=== cmd: {j['command']}\n"
        )
        out.write(hdr.encode())
        try:
            kw = dict(
                cwd=j["cwd"], env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, shell=shell
            )
            if os.name == "nt":
                proc = subprocess.Popen(args, creationflags=_proc.NEW_GROUP, **kw)
            else:
                proc = subprocess.Popen(args, start_new_session=True, **kw)
        except OSError as e:
            out.write(f"=== gpushift: failed to start: {e}\n".encode())
            with db.tx(con):
                db.update(
                    con, job_id, state="failed", exit_code=127, ended_at=time.time(), note=f"failed to start: {e}"
                )
            return 127

        pctime = _proc.create_time(proc.pid)
        with db.tx(con):
            db.update(con, job_id, pid=proc.pid, pid_ctime=pctime, started_at=started)
            cur = db.get_job(con, job_id)
            mode = db.get_mode(con)
        if cur["cancel_requested"]:
            _proc.kill_tree(proc.pid, pctime)
        elif mode != "normal":
            # free started while we were launching: freeze right away.
            _proc.suspend_tree(proc.pid, pctime)
            with db.tx(con):
                db.update(con, job_id, where_state="running", state="frozen", frozen_at=time.time())

        rc = proc.wait()
        ended = time.time()
        with db.tx(con):
            cur = db.get_job(con, job_id)
            active = cur["state"] in db.ACTIVE
            this_run = control.ran(cur, ended) - cur["total_runtime"] if active else ended - started
            if cur["state"] not in db.ACTIVE or cur["runner_pid"] != os.getpid():
                outcome = cur["state"]
            elif cur["cancel_requested"]:
                outcome = "cancelled"
                db.update(
                    con,
                    job_id,
                    state="cancelled",
                    exit_code=rc,
                    ended_at=ended,
                    total_runtime=control.ran(cur, ended),
                    note="cancelled",
                )
            elif cur["state"] == "evicting":
                outcome = "evicted"
                db.update(con, job_id, exit_code=rc)
                note = (
                    "evicted by `gpushift free`; will resume from checkpoint"
                    if control.is_resumable(cur)
                    else "stopped by `gpushift free`; will run again (restartable)"
                )
                control.requeue_front(con, job_id, note=note)
            else:
                outcome = "done" if rc == 0 else "failed"
                db.update(
                    con,
                    job_id,
                    state=outcome,
                    exit_code=rc,
                    ended_at=ended,
                    total_runtime=control.ran(cur, ended),
                    note=None if rc == 0 else failure_note(log, rc, this_run),
                )
        control.clear_stop(job_id)
        out.write(f"=== gpushift: job {job_id} {outcome} (exit code {rc}) after {this_run:.0f}s\n".encode())
    return 0


def main() -> None:
    sys.exit(run(int(sys.argv[1])))


if __name__ == "__main__":
    main()
