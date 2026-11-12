"""The scheduler daemon: ``gpushift daemon``.

Every second: heartbeat, recover lost runners, enforce eviction timeouts,
and (in normal mode) start whatever the scheduler says may start. One daemon
per user, guaranteed by an OS file lock."""

from __future__ import annotations

import logging
import logging.handlers
import os
import shutil
import sys
import time
from pathlib import Path

import psutil

from . import _proc, config, control, cpustat, db, gpumem, gpus, notify, paths, scheduler

log = logging.getLogger("gpushift.daemon")


def _started_by_agent(p: psutil.Process) -> bool:
    """True if a Claude Code session started this process (an agent, not the user)."""
    try:
        return any(a.name().lower().startswith("claude") for a in p.parents())
    except psutil.Error:
        return False


TICK = float(os.environ.get("GPUSHIFT_TICK", "1.0"))
GPU_POLL = float(os.environ.get("GPUSHIFT_GPU_POLL", "5.0"))


# ---------------------------------------------------------------- locking


class Lock:
    """Non-blocking exclusive lock on a file, released when the process dies."""

    def __init__(self, path: Path):
        self.path = path
        self.fh = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")  # noqa: SIM115 - held open for the lifetime of the lock
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self.fh = fh
        return True

    def release(self) -> None:
        if self.fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self.fh.close()
        self.fh = None


def is_running() -> bool:
    lk = Lock(paths.daemon_lock())
    if lk.acquire():
        lk.release()
        return False
    return True


def start_background(wait: float = 5.0) -> bool:
    """Start the daemon detached (if not running). Returns True once it is up."""
    if is_running():
        return True
    _proc.spawn_detached([_proc.windowless_python(), "-m", "gpushift", "daemon"])
    deadline = time.time() + wait
    while time.time() < deadline:
        if is_running():
            return True
        time.sleep(0.1)
    return is_running()


def stop() -> bool:
    con = db.connect()
    pid = db.get_kv(con, "daemon_pid")
    ctime = db.get_kv(con, "daemon_ctime")
    if not is_running():
        return False
    # Only the daemon: runners are its children, and running jobs must survive
    # a stop. The next daemon adopts them.
    ok = _proc.kill(pid, ctime) if pid else False
    for _ in range(50):
        if not is_running():
            return True
        time.sleep(0.1)
    return ok


# ---------------------------------------------------------------- helpers


def install_pylib() -> None:
    """Expose gpushift.checkpoint to job processes in any Python environment."""
    d = paths.pylib_dir() / "gpushift"
    d.mkdir(parents=True, exist_ok=True)
    (d / "__init__.py").write_text('"""gpushift job-side helpers (checkpoint only)."""\n')
    shutil.copyfile(Path(__file__).with_name("checkpoint.py"), d / "checkpoint.py")


def _setup_logging() -> None:
    paths.state_dir().mkdir(parents=True, exist_ok=True)
    h = logging.handlers.RotatingFileHandler(paths.daemon_log(), maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    if sys.stderr is not None and sys.stderr.isatty():
        log.addHandler(logging.StreamHandler())


class Daemon:
    def __init__(self):
        self.con = db.connect()
        # mtime before load: an edit in between is then picked up by the next reload.
        self.cfg_mtime = self._cfg_mtime()
        self.cfg = config.load()
        self.gpu_state: list[gpus.GPU] = list(self.cfg.detected)
        self.last_gpu_poll = 0.0
        self.procs: dict[int, int] | None = None  # pid -> GPU MiB, from the last poll
        self.capacity: dict[int, int] | None = None  # MiB per GPU packed jobs may book
        self.user_since: float | None = None  # when heavy user GPU programs were first seen
        self.user_gone_since: float | None = None  # when the last one closed
        self.cpu = cpustat.Sampler()

    def _cfg_mtime(self) -> float:
        try:
            return paths.config_file().stat().st_mtime
        except FileNotFoundError:
            return 0.0

    def maybe_reload_config(self) -> None:
        m = self._cfg_mtime()
        if m == self.cfg_mtime:
            return
        self.cfg_mtime = m
        try:
            self.cfg = config.load()
            log.info("config reloaded: %s", self.cfg.to_dict())
        except config.ConfigError as e:
            log.error("config invalid, keeping previous: %s", e)

    def poll_gpus(self) -> None:
        now = time.time()
        if now - self.last_gpu_poll < GPU_POLL:
            return
        self.last_gpu_poll = now
        try:
            every = gpus.query()
            self.gpu_state = [g for g in every if g.index in self.cfg.gpus]
            # All GPUs, not just the usable ones: config.load() detects from this.
            db.set_kv(self.con, "gpu_state", [g.to_dict() for g in every])
            db.set_kv(self.con, "gpu_state_at", now)
        except gpus.GPUQueryError as e:
            log.warning("%s", e)
        self.procs = gpumem.per_process()
        self.measure_jobs()
        self.watch_user()
        self.poll_cpu()

    def poll_cpu(self) -> None:
        job_of = {}
        for j in db.active(self.con):
            p = _proc.get_proc(j["pid"], j["pid_ctime"])
            if p is not None:
                job_of.update({q.pid: j["id"] for q in _proc.tree(p)})
        try:
            db.set_kv(self.con, "cpu_state", self.cpu.sample(job_of))
        except Exception:  # a dashboard reading must never stop the scheduler
            log.exception("cpu reading failed")

    # -- the user's own GPU programs (games, Blender, ...)

    def user_gpu_programs(self) -> list[str]:
        """Names of heavy GPU programs running outside gpushift."""
        af = self.cfg.auto_free
        if not self.procs:
            return []
        ours: set[int] = set()
        for j in db.active(self.con):
            for pid, ctime in ((j["pid"], j["pid_ctime"]), (j["runner_pid"], j["runner_ctime"])):
                p = _proc.get_proc(pid, ctime)
                if p is not None:
                    ours.update(q.pid for q in _proc.tree(p))
        found = set()
        for pid, mib in self.procs.items():
            if pid in ours or mib <= 0:
                continue
            try:
                p = psutil.Process(pid)
                name = p.name()
            except psutil.Error:
                continue
            listed = name.lower() in af.apps
            if not listed and (name.lower() in af.ignore or mib < af.min_vram_mb):
                continue
            if not af.include_agents and _started_by_agent(p):
                continue
            found.add(name)
        return sorted(found)

    def watch_user(self) -> None:
        """Free the GPUs while the user runs heavy GPU programs; resume once
        they have been closed for a while (only after a free we did ourselves)."""
        af = self.cfg.auto_free
        if not af.enabled:
            return
        now = time.time()
        apps = self.user_gpu_programs()
        mode = db.get_mode(self.con)
        auto = db.get_kv(self.con, "auto_free")
        if apps:
            self.user_gone_since = None
            if self.user_since is None:
                self.user_since = now
            snoozed = db.get_kv(self.con, "auto_free_snooze")
            if mode == "normal" and not snoozed and now - self.user_since >= af.after:
                log.info("freeing the GPUs: %s is using the GPU", ", ".join(apps))
                control.free(self.con, soft=False, auto={"apps": apps, "since": now})
                notify.desktop(
                    "gpushift: GPUs freed for you",
                    f"{', '.join(apps)} is using the GPU. The queue waits until it has been closed for a while.",
                )
            elif auto and auto.get("apps") != apps:
                db.set_kv(self.con, "auto_free", {**auto, "apps": apps})
            return
        self.user_since = None
        if db.get_kv(self.con, "auto_free_snooze"):
            db.set_kv(self.con, "auto_free_snooze", None)
        if mode == "normal" or not auto:
            self.user_gone_since = None
            return
        if self.user_gone_since is None:
            self.user_gone_since = now
        if now - self.user_gone_since >= af.resume_after:
            log.info("resuming: %s closed", ", ".join(auto.get("apps") or []))
            control.resume(self.con, auto=True)
            self.user_gone_since = None
            notify.desktop("gpushift: queue resumed", f"{', '.join(auto.get('apps') or ['Your GPU program'])} closed.")

    def measure_jobs(self) -> None:
        """Record how much GPU memory each running job uses (and its peak),
        and how much is left for packed jobs."""
        if self.procs is None:
            self.capacity = None
            db.set_kv(self.con, "pack_capacity", None)
            return
        active = db.active(self.con)
        job_vram = {}
        for j in active:
            p = _proc.get_proc(j["pid"], j["pid_ctime"])
            if p is None:
                continue
            mib = sum(self.procs.get(q.pid, 0) for q in _proc.tree(p))
            job_vram[j["id"]] = mib
            if mib > (j["peak_vram_mb"] or 0) and j["state"] == "running":
                with db.tx(self.con):
                    db.update(self.con, j["id"], peak_vram_mb=mib)
        db.set_kv(self.con, "job_vram", job_vram)
        self.capacity = control.pack_capacity(self.gpu_state, job_vram, active)
        db.set_kv(self.con, "pack_capacity", self.capacity)

    def mem_ok(self) -> set[int] | None:
        return control.mem_ok(self.cfg, self.gpu_state)

    def schedule(self) -> None:
        machine = control.machine_from_config(self.cfg, self.mem_ok(), self.capacity)
        now = time.time()
        with db.tx(self.con):
            if db.get_mode(self.con) != "normal":
                return
            guard, fair = self.cfg.starvation_guard, self.cfg.fair_share
            running, queued, remaining, durations, res, waiting = control.decide(
                self.con, machine, guard, fair, self.cfg.reorder_slack, now
            )
            starts = scheduler.plan(
                [control.req_of(j) for j in queued if j["id"] not in waiting],
                running,
                machine,
                guard,
                durations=durations,
                remaining=remaining,
            )
            jumped = [jid for jid, _ in starts if jid in res.moved]
            if jumped:
                log.info(
                    "job %s runs ahead of job %s: saves %.0f min of waiting in total",
                    ", ".join(map(str, jumped)),
                    ", ".join(map(str, res.passed)),
                    res.gain / 60,
                )
            control.promise(self.con, res if jumped else scheduler.Reorder(res.order, res.est), now)
            for jid, g in starts:
                db.update(
                    self.con,
                    jid,
                    where_state="queued",
                    state="running",
                    gpus=g,
                    started_at=now,
                    runner_pid=None,
                    runner_ctime=None,
                    pid=None,
                    pid_ctime=None,
                    exit_code=None,
                    frozen_at=None,
                    paused_s=0.0,
                    evict_requested_at=None,
                )
        for jid, g in starts:
            try:
                p = _proc.spawn_detached([_proc.console_python(), "-m", "gpushift.runner", str(jid)])
                log.info("started job %s on gpus %s (runner pid %s)", jid, g, p.pid)
            except OSError as e:
                log.error("could not start runner for job %s: %s", jid, e)
                with db.tx(self.con):
                    db.update(self.con, jid, state="failed", ended_at=time.time(), note=f"could not start runner: {e}")

    def tick(self) -> None:
        db.set_kv(self.con, "daemon_heartbeat", time.time())
        self.maybe_reload_config()
        for jid in control.reconcile(self.con):
            log.warning("job %s: runner lost", jid)
        for jid in control.enforce_evict_timeout(self.con, self.cfg.evict_timeout):
            log.warning("job %s did not checkpoint in time; frozen", jid)
        for jid, dep, state in control.cancel_orphans(self.con):
            log.info("job %s cancelled: job %s, which it waited for, %s", jid, dep, state)
        self.poll_gpus()
        self.schedule()
        self.notify_ended()

    def notify_ended(self) -> None:
        """Desktop notification for jobs that ended since the last tick."""
        until = db.get_kv(self.con, "desktop_notified_until")
        if until is None:  # first run: don't announce old history
            db.set_kv(self.con, "desktop_notified_until", time.time())
            return
        rows = self.con.execute(
            "SELECT * FROM jobs WHERE state IN ('done','failed','cancelled') AND ended_at > ? ORDER BY ended_at",
            (until,),
        ).fetchall()
        if not rows:
            return
        for j in map(db.decode, rows):
            if self.cfg.notify_desktop == "all" or (self.cfg.notify_desktop == "failures" and j["state"] == "failed"):
                notify.desktop(*notify.job_ended(j))
        db.set_kv(self.con, "desktop_notified_until", rows[-1]["ended_at"])

    def run(self) -> None:
        log.info("gpushift daemon started (pid %s) gpus=%s slots=%s", os.getpid(), self.cfg.gpus, self.cfg.slots)
        if self.cfg.ui_port:
            from . import ui

            ui.serve(self.cfg.ui_port)
        while True:
            try:
                self.tick()
            except Exception:  # keep the queue alive no matter what
                log.exception("tick failed")
            time.sleep(TICK)


def main() -> int:
    lock = Lock(paths.daemon_lock())
    # Retry briefly: `is_running()` probes take the lock for a moment.
    for _ in range(20):
        if lock.acquire():
            break
        time.sleep(0.1)
    else:
        print("gpushift daemon is already running", file=sys.stderr)
        return 1
    _setup_logging()
    install_pylib()
    con = db.connect()
    db.set_kv(con, "daemon_pid", os.getpid())
    db.set_kv(con, "daemon_ctime", _proc.create_time(os.getpid()))
    db.set_kv(con, "daemon_started", time.time())
    try:
        Daemon().run()
    except KeyboardInterrupt:
        pass
    finally:
        lock.release()
    return 0
