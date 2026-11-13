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

from . import _proc, config, control, cpustat, db, gpumem, gpus, notify, paths, scheduler

log = logging.getLogger("gpushift.daemon")


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
        self.cpu: cpustat.Sampler | None = None  # made when a dashboard first asks (slow to set up on Windows)
        self.snap: _proc.Snapshot | None = None  # process tree, taken once per poll
        self.job_pids: dict[int, set[int]] = {}  # job id -> its processes (runner and job), per poll

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
        self.read_job_processes()
        self.procs = gpumem.per_process(set().union(*self.job_pids.values()))
        self.measure_jobs()
        self.poll_cpu()

    def read_job_processes(self) -> None:
        """One process snapshot per poll, and the processes of each active job
        (its runner and everything under it)."""
        try:
            self.snap = _proc.Snapshot()
        except Exception:  # never let a reading stop the scheduler
            log.exception("process snapshot failed")
            self.snap, self.job_pids = None, {}
            return
        self.job_pids = {}
        for j in db.active(self.con):
            pids: set[int] = set()
            for pid, ctime in ((j["runner_pid"], j["runner_ctime"]), (j["pid"], j["pid_ctime"])):
                if _proc.get_proc(pid, ctime) is not None:
                    pids.update(self.snap.tree(pid))
            self.job_pids[j["id"]] = pids

    def poll_cpu(self) -> None:
        """CPU, memory and disk readings. Only the dashboard shows them, so they
        are only taken while one is open."""
        from . import ui

        if not ui.being_viewed():
            return
        job_of = {}
        for jid, pids in self.job_pids.items():
            job_of.update(dict.fromkeys(pids, jid))
        try:
            if self.cpu is None:
                self.cpu = cpustat.Sampler()
            db.set_kv(self.con, "cpu_state", self.cpu.sample(job_of))
        except Exception:  # a dashboard reading must never stop the scheduler
            log.exception("cpu reading failed")

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
            pids = self.job_pids.get(j["id"])
            if not pids:
                continue
            mib = sum(self.procs.get(pid, 0) for pid in pids)
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
