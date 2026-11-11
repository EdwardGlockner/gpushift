"""Operations that change job/queue state. Shared by the CLI, the daemon and
job runners, so most of gpushift keeps working even when the daemon is down."""

from __future__ import annotations

import os
import statistics
import time

from . import _proc, db, paths
from .scheduler import Machine, Req, simulate

# ---------------------------------------------------------------- helpers


def req_of(j: dict) -> Req:
    """The scheduler's view of a job. ``j["vram_need"]`` (set by plan_inputs)
    makes it packable."""
    return Req(j["id"], j["n_gpus"], j["slots"], bool(j["exclusive"]), bool(j["shared"]), j.get("vram_need"))


PACK_MARGIN = 1.15  # learned memory needs get 15% ...
PACK_EXTRA_MB = 256  # ... plus 256 MiB of headroom


def vram_need(con, j: dict) -> int | None:
    """MiB per GPU the job needs: --vram, or learned from the peaks of earlier
    successful runs of the same command. None if unknown."""
    if j.get("vram_mb"):
        return int(j["vram_mb"])
    if j["n_gpus"] < 1 or j["exclusive"] or j["shared"]:
        return None
    rows = con.execute(
        "SELECT peak_vram_mb FROM jobs WHERE state='done' AND command=? AND project IS ? AND peak_vram_mb > 0 "
        "ORDER BY ended_at DESC LIMIT 5",
        (j["command"], j["project"]),
    ).fetchall()
    if not rows:
        return None
    peak = max(r["peak_vram_mb"] for r in rows) / j["n_gpus"]
    return int(peak * PACK_MARGIN + PACK_EXTRA_MB)


def is_resumable(j: dict) -> bool:
    return bool(j["resumable"]) or paths.resumable_marker(j["id"]).exists()


def request_stop(j: dict) -> None:
    f = paths.stop_file(j["id"])
    f.parent.mkdir(parents=True, exist_ok=True)
    f.touch()
    _proc.terminate(j["pid"], j["pid_ctime"])


def clear_stop(job_id: int) -> None:
    try:
        paths.stop_file(job_id).unlink()
    except FileNotFoundError:
        pass


def requeue_front(con, job_id: int, *, note: str, resume: bool = True, from_states=db.ACTIVE) -> bool:
    """Put an active job back at the head of the queue (eviction / lost runner)."""
    j = db.get_job(con, job_id)
    if j is None:
        return False
    now = time.time()
    runtime = j["total_runtime"] + (now - j["started_at"] if j["started_at"] else 0.0)
    return db.update(
        con,
        job_id,
        where_state=from_states,
        state="queued",
        position=db.front_position(con),
        resume=1 if resume else j["resume"],
        evictions=j["evictions"] + (1 if resume else 0),
        total_runtime=runtime,
        gpus=None,
        runner_pid=None,
        runner_ctime=None,
        pid=None,
        pid_ctime=None,
        started_at=None,
        frozen_at=None,
        evict_requested_at=None,
        note=note,
    )


# ---------------------------------------------------------------- cancel


def cancel(con, job_id: int) -> str:
    """Cancel a queued or active job. Returns a human-readable result."""
    with db.tx(con):
        j = db.get_job(con, job_id)
        if j is None:
            return f"job {job_id} not found"
        if j["state"] in db.FINAL:
            return f"job {job_id} already {j['state']}"
        if j["state"] == "queued":
            db.update(
                con,
                job_id,
                where_state="queued",
                state="cancelled",
                ended_at=time.time(),
                note="cancelled while queued",
            )
            return f"job {job_id} cancelled (was queued)"
        db.update(con, job_id, cancel_requested=1)
    killed = _proc.kill_tree(j["pid"], j["pid_ctime"])
    if not _proc.alive(j["runner_pid"], j["runner_ctime"]):
        # Nobody left to record the result; do it ourselves.
        with db.tx(con):
            db.update(con, job_id, where_state=db.ACTIVE, state="cancelled", ended_at=time.time(), note="cancelled")
    return f"job {job_id} cancelled" + (" (process killed)" if killed else "")


# ---------------------------------------------------------------- free / resume


def free(con, soft: bool, auto: dict | None = None) -> dict:
    """Stop the queue and free the machine.

    full: resumable jobs are asked to checkpoint and exit, restartable jobs are
          stopped (both are requeued at the front); others are frozen.
    soft: everything is frozen; VRAM stays allocated.
    auto: set when the daemon frees because the user runs heavy GPU programs
          ({"apps": [...]}); only such a free is resumed automatically."""
    now = time.time()
    with db.tx(con):
        prev = db.get_mode(con)
        db.set_kv(con, "mode", "soft" if soft else "free")
        if prev == "normal":
            db.set_kv(con, "mode_since", now)
        # A manual free (even during an automatic one) stays until the user resumes.
        db.set_kv(con, "auto_free", auto)
        jobs = db.active(con)
    rep = {
        "previous_mode": prev,
        "mode": "soft" if soft else "free",
        "evicting": [],
        "restarting": [],
        "frozen": [],
        "pending": [],
    }
    for j in jobs:
        jid = j["id"]
        if j["pid"] is None:
            # Runner has not started the process yet; it checks the mode and
            # freezes the job itself.
            rep["pending"].append(jid)
            continue
        evict = not soft and is_resumable(j)
        if not soft and not evict and j["restartable"] and j["state"] in ("running", "frozen"):
            # Safe to stop and run again: stop it now; its runner puts it back
            # at the front of the queue.
            with db.tx(con):
                ok = db.update(
                    con,
                    jid,
                    where_state=("running", "frozen"),
                    state="evicting",
                    evict_requested_at=now,
                    frozen_at=None,
                )
            if ok:
                _proc.kill_tree(j["pid"], j["pid_ctime"])
                rep["restarting"].append(jid)
            continue
        if j["state"] == "running":
            if evict:
                with db.tx(con):
                    ok = db.update(con, jid, where_state="running", state="evicting", evict_requested_at=now)
                if ok:
                    request_stop(j)
                    rep["evicting"].append(jid)
            else:
                _proc.suspend_tree(j["pid"], j["pid_ctime"])
                with db.tx(con):
                    if db.update(con, jid, where_state="running", state="frozen", frozen_at=now):
                        rep["frozen"].append(jid)
                    else:  # finished meanwhile
                        _proc.resume_tree(j["pid"], j["pid_ctime"])
        elif j["state"] == "frozen":
            if evict:  # soft -> full upgrade: wake it up so it can checkpoint
                with db.tx(con):
                    ok = db.update(
                        con, jid, where_state="frozen", state="evicting", evict_requested_at=now, frozen_at=None
                    )
                if ok:
                    request_stop(j)
                    _proc.resume_tree(j["pid"], j["pid_ctime"])
                    rep["evicting"].append(jid)
            else:
                rep["frozen"].append(jid)
        elif j["state"] == "evicting":
            rep["evicting"].append(jid)
    return rep


def resume(con, auto: bool = False) -> dict:
    """Leave free mode. A manual resume during an automatic free also tells
    the daemon not to free again until the user's GPU programs have closed."""
    with db.tx(con):
        prev = db.get_mode(con)
        db.set_kv(con, "mode", "normal")
        if db.get_kv(con, "auto_free") and not auto:
            db.set_kv(con, "auto_free_snooze", True)
        db.set_kv(con, "auto_free", None)
        jobs = db.active(con)
    rep = {"previous_mode": prev, "resumed": [], "evicting": []}
    for j in jobs:
        if j["state"] == "frozen":
            clear_stop(j["id"])
            with db.tx(con):
                ok = db.update(
                    con, j["id"], where_state="frozen", state="running", frozen_at=None, evict_requested_at=None
                )
            if ok:
                _proc.resume_tree(j["pid"], j["pid_ctime"])
                rep["resumed"].append(j["id"])
        elif j["state"] == "evicting":
            rep["evicting"].append(j["id"])
    return rep


def enforce_evict_timeout(con, timeout: int) -> list[int]:
    """Resumable jobs that did not exit in time get frozen instead."""
    now = time.time()
    frozen = []
    mode = db.get_mode(con)
    for j in db.active(con):
        if j["state"] != "evicting" or not j["evict_requested_at"]:
            continue
        if now - j["evict_requested_at"] < timeout:
            continue
        if mode == "normal":
            # User resumed meanwhile: let it keep running.
            clear_stop(j["id"])
            with db.tx(con):
                db.update(con, j["id"], where_state="evicting", state="running", evict_requested_at=None)
            continue
        _proc.suspend_tree(j["pid"], j["pid_ctime"])
        with db.tx(con):
            if db.update(
                con,
                j["id"],
                where_state="evicting",
                state="frozen",
                frozen_at=now,
                note=f"did not checkpoint within {timeout}s; frozen instead",
            ):
                frozen.append(j["id"])
            else:
                _proc.resume_tree(j["pid"], j["pid_ctime"])
    return frozen


# ---------------------------------------------------------------- lost runners


def reconcile(con, grace: float = 60.0) -> list[int]:
    """Handle active jobs whose runner process is gone (crash, reboot,
    killed). Resumable jobs are requeued at the front; others fail."""
    now = time.time()
    lost = []
    for j in db.active(con):
        if j["runner_pid"] is None:
            if now - (j["started_at"] or now) < grace:
                continue
        elif _proc.alive(j["runner_pid"], j["runner_ctime"]):
            continue
        _proc.kill_tree(j["pid"], j["pid_ctime"])
        with db.tx(con):
            cur = db.get_job(con, j["id"])
            if cur is None or cur["state"] not in db.ACTIVE or cur["runner_pid"] != j["runner_pid"]:
                continue  # runner finished it, or it was restarted, meanwhile
            if cur["cancel_requested"]:
                db.update(con, j["id"], state="cancelled", ended_at=now, note="cancelled")
            elif is_resumable(cur):
                requeue_front(con, j["id"], note="runner lost (crash/reboot); will resume from checkpoint")
            elif cur["restartable"]:
                requeue_front(con, j["id"], note="runner lost (crash/reboot); will run again (restartable)")
            else:
                db.update(
                    con,
                    j["id"],
                    state="failed",
                    ended_at=now,
                    note="runner lost (crash/reboot) and job is not resumable; `gpushift retry` to rerun",
                )
        lost.append(j["id"])
    return lost


# ---------------------------------------------------------------- estimates


def expected_duration(con, j: dict) -> float | None:
    if j["eta_seconds"]:
        return float(j["eta_seconds"])
    hist = db.history_runtimes(con, j["command"], j["project"])
    return statistics.median(hist) if hist else None


def time_left(con, j: dict, now: float | None = None) -> float | None:
    """Expected seconds until the job is done (for a queued job: its whole
    remaining run). Counts time it already ran before an eviction. None when
    unknown: no estimate, or it has already run past its estimate."""
    dur = expected_duration(con, j)
    if dur is None:
        return None
    now = time.time() if now is None else now
    ran = j["total_runtime"]
    if j["state"] in db.ACTIVE and j["started_at"]:
        ran += now - j["started_at"]
    left = dur - ran
    return left if left > 0 else None


FAIR_WINDOW = 24 * 3600  # how far back GPU time counts toward a session's share
UNKNOWN_JOB_SECONDS = 600.0  # what a job without an estimate counts as when taking turns


def session_usage(con, now: float) -> dict[str | None, float]:
    """GPU-seconds each session used recently (running jobs included)."""
    usage: dict[str | None, float] = {}
    rows = con.execute(
        "SELECT session, n_gpus, total_runtime, started_at, state FROM jobs "
        "WHERE state IN ('running','frozen','evicting') OR ended_at > ?",
        (now - FAIR_WINDOW,),
    )
    for r in rows:
        ran = r["total_runtime"] + (now - r["started_at"] if r["state"] in db.ACTIVE and r["started_at"] else 0.0)
        usage[r["session"]] = usage.get(r["session"], 0.0) + ran * max(1, r["n_gpus"])
    return usage


def ordered_queue(con, fair: bool = True, now: float | None = None) -> list[dict]:
    """Queued jobs in the order they get their turn.

    Jobs moved up with --priority and evicted jobs waiting to resume come
    first. With ``fair``, the rest take turns between sessions (your own
    terminal counts as one): the next job comes from the session that has used
    the least GPU time, counting what its earlier picks will use."""
    queued = db.queued(con)
    first = [j for j in queued if j["prio"] or j["resume"]]
    rest = [j for j in queued if not (j["prio"] or j["resume"])]
    if not fair or len({j["session"] for j in rest}) < 2:
        return first + rest
    now = time.time() if now is None else now
    usage = session_usage(con, now)
    lanes: dict[str | None, list[dict]] = {}
    for j in rest:
        lanes.setdefault(j["session"], []).append(j)
    order = []
    while lanes:
        # Least GPU time first; on a tie, whoever submitted earlier.
        s = min(lanes, key=lambda k: (usage.get(k, 0.0), lanes[k][0]["position"], lanes[k][0]["id"]))
        j = lanes[s].pop(0)
        if not lanes[s]:
            del lanes[s]
        order.append(j)
        usage[s] = usage.get(s, 0.0) + (time_left(con, j, now) or UNKNOWN_JOB_SECONDS) * max(1, j["n_gpus"])
    return first + order


def plan_inputs(con, now: float | None = None, fair: bool = True):
    """Running jobs, queued jobs (in turn order) and the time estimates the
    scheduler uses."""
    now = time.time() if now is None else now
    active, queued = db.active(con), ordered_queue(con, fair, now)
    for j in active + queued:
        j["vram_need"] = vram_need(con, j)
    running = [(req_of(j), j["gpus"] or []) for j in active]
    remaining = {j["id"]: time_left(con, j, now) for j in active}
    durations = {j["id"]: time_left(con, j, now) for j in queued}
    return running, queued, remaining, durations


def estimate_starts(con, machine: Machine, guard: bool, fair: bool = True) -> dict[int, float | None]:
    """job id -> estimated start (unix time) for queued jobs; None = unknown."""
    now = time.time()
    if db.get_mode(con) != "normal":
        return {j["id"]: None for j in db.queued(con)}
    running, queued, remaining, durations = plan_inputs(con, now, fair)
    off = simulate(
        [(req_of(j), durations[j["id"]]) for j in queued],
        [(r, g, remaining[r.id]) for r, g in running],
        machine,
        guard,
    )
    return {k: (now + v if v is not None else None) for k, v in off.items()}


def machine_from_config(cfg, mem_ok: set[int] | None = None, capacity: dict[int, int] | None = None) -> Machine:
    return Machine(list(cfg.gpus), cfg.shared_gpu, cfg.slots, mem_ok, capacity if cfg.packing else None)


def pack_capacity(live, job_vram: dict[int, int], active: list[dict]) -> dict[int, int] | None:
    """MiB per GPU that packed jobs may book: total memory minus what programs
    outside gpushift use (measured use minus gpushift jobs' own use). None if unknown."""
    if not live:
        return None
    ours: dict[int, float] = {}
    for j in active:
        if j["gpus"] and j["id"] in job_vram:
            for g in j["gpus"]:
                ours[g] = ours.get(g, 0.0) + job_vram[j["id"]] / len(j["gpus"])
    return {g.index: int(g.mem_total - max(0.0, g.mem_used - ours.get(g.index, 0.0))) for g in live}


def mem_ok(cfg, live) -> set[int] | None:
    """GPUs whose free memory passes gpu_free_perc (None if unknown)."""
    if not live:
        return None
    known = {g.index: g for g in live}
    return {i for i in cfg.gpus if i not in known or known[i].free_perc >= cfg.gpu_free_perc}


def live_machine(cfg) -> Machine:
    from . import gpus

    try:
        live = gpus.query()
    except gpus.GPUQueryError:
        live = []
    return machine_from_config(cfg, mem_ok(cfg, live), _daemon_capacity())


def _daemon_capacity() -> dict[int, int] | None:
    """The packing capacity the daemon measured last (so `status` estimates
    match what the daemon will do)."""
    con = db.connect()
    cap = db.get_kv(con, "pack_capacity")
    return {int(k): v for k, v in cap.items()} if cap else None


def filtered_env() -> dict[str, str]:
    """The submitter's environment, minus per-session secrets."""
    drop_prefixes = ("CLAUDE_CODE_MESSAGING_",)
    return {k: v for k, v in os.environ.items() if not k.startswith(drop_prefixes)}
