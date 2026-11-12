"""Operations that change job/queue state. Shared by the CLI, the daemon and
job runners, so most of gpushift keeps working even when the daemon is down."""

from __future__ import annotations

import math
import os
import statistics
import time

from . import _proc, db, paths
from .scheduler import Machine, Req, reorder, simulate

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


def ran(j: dict, now: float | None = None) -> float:
    """Seconds the job has actually run, over all attempts. Time spent frozen
    doesn't count."""
    total = j["total_runtime"]
    if j["state"] in db.ACTIVE and j["started_at"]:
        now = time.time() if now is None else now
        frozen = now - j["frozen_at"] if j.get("frozen_at") else 0.0
        total += max(0.0, now - j["started_at"] - (j.get("paused_s") or 0.0) - frozen)
    return total


def unfreeze(j: dict, now: float) -> dict:
    """Fields that end a frozen spell: the time it lasted moves to paused_s."""
    if not j.get("frozen_at"):
        return {"frozen_at": None}
    return {"frozen_at": None, "paused_s": (j.get("paused_s") or 0.0) + now - j["frozen_at"]}


def to_front(con, job_id: int) -> str:
    """Move a queued job to the front of the queue (--priority / `gpushift priority`)."""
    with db.tx(con):
        queue = db.queued(con)
        if job_id not in [q["id"] for q in queue]:
            cur = db.get_job(con, job_id)
            return f"job {job_id} is {cur['state'] if cur else 'not found'}, not queued"
        problem = order_problem(queue, [job_id] + [q["id"] for q in queue if q["id"] != job_id])
        if problem:
            return problem
        db.update(con, job_id, position=db.front_position(con), prio=1)
    return f"job {job_id} moved to the front of the queue"


def requeue_front(con, job_id: int, *, note: str, resume: bool = True, from_states=db.ACTIVE) -> bool:
    """Put an active job back at the head of the queue (eviction / lost runner)."""
    j = db.get_job(con, job_id)
    if j is None:
        return False
    runtime = ran(j)
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
        paused_s=0.0,
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
            cur = db.get_job(con, job_id)
            db.update(
                con,
                job_id,
                where_state=db.ACTIVE,
                state="cancelled",
                ended_at=time.time(),
                total_runtime=ran(cur),
                note="cancelled",
            )
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
                    **unfreeze(j, now),
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
                        con, jid, where_state="frozen", state="evicting", evict_requested_at=now, **unfreeze(j, now)
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
    now = time.time()
    for j in jobs:
        if j["state"] == "frozen":
            clear_stop(j["id"])
            with db.tx(con):
                ok = db.update(
                    con, j["id"], where_state="frozen", state="running", evict_requested_at=None, **unfreeze(j, now)
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
                db.update(con, j["id"], state="cancelled", ended_at=now, total_runtime=ran(cur, now), note="cancelled")
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
                    total_runtime=ran(cur, now),
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
    left = dur - ran(j, now)
    return left if left > 0 else None


FAIR_WINDOW = 24 * 3600  # how far back GPU time counts toward a session's share
UNKNOWN_JOB_SECONDS = 600.0  # what a job without an estimate counts as when taking turns


def session_usage(con, now: float) -> dict[str | None, float]:
    """GPU-seconds each session used recently (running jobs included)."""
    usage: dict[str | None, float] = {}
    rows = con.execute(
        "SELECT session, n_gpus, total_runtime, started_at, frozen_at, paused_s, state FROM jobs "
        "WHERE state IN ('running','frozen','evicting') OR ended_at > ?",
        (now - FAIR_WINDOW,),
    )
    for r in rows:
        usage[r["session"]] = usage.get(r["session"], 0.0) + ran(dict(r), now) * max(1, r["n_gpus"])
    return usage


def ordered_queue(con, fair: bool = False, now: float | None = None) -> list[dict]:
    """Queued jobs in the order they get their turn.

    Jobs moved up with --priority and evicted jobs waiting to resume come
    first. With ``fair``, the rest take turns between sessions (your own
    terminal counts as one): the next job comes from the session that has used
    the least GPU time, counting what its earlier picks will use."""
    queued = db.queued(con)
    if not fair:
        return queued  # queue position (--priority and evicted jobs get the front position)
    first = [j for j in queued if j["prio"] or j["resume"]]
    rest = [j for j in queued if not (j["prio"] or j["resume"])]
    if len({j["session"] for j in rest}) < 2:
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


def waiting_for(con, queued: list[dict]) -> dict[int, set[int]]:
    """Queued job id -> the jobs it still waits for (--after): those that are
    queued or running. Jobs that wait for nothing are left out."""
    deps = {j["id"]: set(j.get("after_ids") or []) for j in queued if j.get("after_ids")}
    ids = set().union(*deps.values()) if deps else set()
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    open_ = {
        r["id"]
        for r in con.execute(
            f"SELECT id FROM jobs WHERE id IN ({marks}) AND state NOT IN ('done','failed','cancelled')", list(ids)
        )
    }
    return {i: d & open_ for i, d in deps.items() if d & open_}


def cancel_orphans(con) -> list[tuple[int, int, str]]:
    """Cancel queued jobs that wait for a job which failed or was cancelled:
    they would wait forever. Returns (job, the job it waited for, its state)."""
    out = []
    with db.tx(con):
        for j in db.queued(con):
            for dep in j.get("after_ids") or []:
                d = con.execute("SELECT state FROM jobs WHERE id=?", (dep,)).fetchone()
                if d is None or d["state"] in ("failed", "cancelled"):
                    state = d["state"] if d else "gone"
                    db.update(
                        con,
                        j["id"],
                        where_state="queued",
                        state="cancelled",
                        ended_at=time.time(),
                        note=f"not run: job {dep}, which it waited for, {state if state != 'gone' else 'no longer exists'}",
                    )
                    out.append((j["id"], dep, state))
                    break
    return out


def plan_inputs(con, now: float | None = None, fair: bool = False):
    """Running jobs, queued jobs (in queue order), the time estimates the
    scheduler uses, and what queued jobs still wait for (--after)."""
    now = time.time() if now is None else now
    active, queued = db.active(con), ordered_queue(con, fair, now)
    for j in active + queued:
        j["vram_need"] = vram_need(con, j)
    running = [(req_of(j), j["gpus"] or []) for j in active]
    remaining = {j["id"]: time_left(con, j, now) for j in active}
    durations = {j["id"]: time_left(con, j, now) for j in queued}
    return running, queued, remaining, durations, waiting_for(con, queued)


def decide(con, machine: Machine, guard: bool, fair: bool = False, slack: float = 0.0, now: float | None = None):
    """The order the queue runs in: queue order, possibly with one job moved to
    the front when that saves waiting overall (see scheduler.reorder).

    Returns (running, queued in that order, remaining, durations, Reorder,
    waiting) where waiting maps jobs that may not start yet to what they wait for."""
    now = time.time() if now is None else now
    running, queued, remaining, durations, waiting = plan_inputs(con, now, fair)
    args, kw = _reorder_args(con, queued, running, remaining, durations, machine, guard, slack, now, waiting)
    res = reorder(*args, **kw)
    by_id = {j["id"]: j for j in queued}
    return running, [by_id[i] for i in res.order], remaining, durations, res, waiting


def _reorder_args(con, queued, running, remaining, durations, machine, guard, slack, now, waiting):
    promised = db.get_kv(con, "reorder_limits") or {}
    return (
        [(req_of(j), durations[j["id"]]) for j in queued],
        [(r, g, remaining[r.id]) for r, g in running],
        machine,
        guard,
    ), dict(
        slack=slack,
        pinned={j["id"] for j in queued if j["prio"] or j["resume"]},
        limits={int(k): v - now for k, v in promised.items()},
        after=waiting,
    )


def queue_and_starts(con, cfg, plan_out: dict | None = None) -> tuple[list[dict], dict[int, float | None]]:
    """Queued jobs in the order they are expected to start, and job id ->
    estimated start (unix time; None = unknown). Each queued job gets
    ``waiting_for`` (ids, --after). plan_out, if given, is filled with job id ->
    (start, gpus, end or None) in unix time for every job the forecast places."""
    if db.get_mode(con) != "normal":
        queued = ordered_queue(con, cfg.fair_share)
        waiting = waiting_for(con, queued)
        for j in queued:
            j["waiting_for"] = sorted(waiting.get(j["id"], ()))
        return queued, {j["id"]: None for j in queued}
    now = time.time()
    running, queued, remaining, durations, waiting = plan_inputs(con, now, cfg.fair_share)
    machine = live_machine(cfg, con)
    args, kw = _reorder_args(
        con, queued, running, remaining, durations, machine, cfg.starvation_guard, cfg.reorder_slack, now, waiting
    )
    out: dict = {}
    off = simulate(*args, **kw, out=out)
    if plan_out is not None:
        for i, (start, gpus, end) in out.items():
            plan_out[i] = (now + start, gpus, None if end is None else now + end)
    est = {k: (now + v if v is not None else None) for k, v in off.items()}
    for j in queued:
        j["waiting_for"] = sorted(waiting.get(j["id"], ()))
    turn = {j["id"]: n for n, j in enumerate(queued)}
    for j in queued:
        j["turn"] = turn[j["id"]] + 1
    queued.sort(key=lambda j: (est[j["id"]] is None, est[j["id"]] or 0.0, turn[j["id"]]))
    return queued, est


def order_problem(queue: list[dict], order: list[int]) -> str | None:
    """Why this queue order is not allowed, or None: a job may not be put
    ahead of a queued job it waits for (--after)."""
    pos = {i: n for n, i in enumerate(order)}
    for j in queue:
        for dep in j.get("after_ids") or []:
            if dep in pos and j["id"] in pos and pos[dep] > pos[j["id"]]:
                return f"job {j['id']} waits for job {dep} (--after), so it can't go ahead of it"
    return None


def move(con, job_id: int, before: int | None) -> str:
    """Move a queued job to just before another queued job (None: to the end).
    A job moved up is pinned like --priority (later reordering never pushes it
    back); a job moved down loses that."""
    with db.tx(con):
        queue = db.queued(con)
        ids = [j["id"] for j in queue]
        if job_id not in ids:
            j = db.get_job(con, job_id)
            return f"job {job_id} is {j['state'] if j else 'not found'}, not queued"
        if before is not None and before not in ids:
            return f"job {before} is not queued"
        if before == job_id:
            return f"job {job_id} stays where it is"
        old = ids.index(job_id)
        rest = [j for j in queue if j["id"] != job_id]
        at = len(rest) if before is None else next(i for i, j in enumerate(rest) if j["id"] == before)
        order = [j["id"] for j in rest]
        order.insert(at, job_id)
        problem = order_problem(queue, order)
        if problem:
            return problem
        lo = rest[at - 1]["position"] if at > 0 else None
        hi = rest[at]["position"] if at < len(rest) else None
        if lo is not None and hi is not None and hi - lo < 1e-6:
            # Positions have been halved too often: renumber the queue first.
            for n, j in enumerate(rest):
                j["position"] = float(n)
                db.update(con, j["id"], position=j["position"])
            lo, hi = (float(at - 1) if at > 0 else None), (float(at) if at < len(rest) else None)
        if lo is None and hi is None:
            pos = 0.0
        elif lo is None:
            pos = hi - 1.0
        elif hi is None:
            pos = lo + 1.0
        else:
            pos = (lo + hi) / 2
        up = at < old
        db.update(con, job_id, position=pos, prio=1 if up else 0)
    where = "the end" if before is None else f"before job {before}"
    return f"job {job_id} moved to {where}"


def promise(con, res, now: float) -> None:
    """Remember the latest start each job was promised when a job jumped ahead,
    so later reorders cannot push it back further. Forget jobs no longer queued."""
    old = {int(k): v for k, v in (db.get_kv(con, "reorder_limits") or {}).items()}
    queued = set(res.order)
    new = {k: v for k, v in old.items() if k in queued}
    for i, off in (res.limits or {}).items():
        new[i] = min(new.get(i, math.inf), now + off)
    if new != old:
        db.set_kv(con, "reorder_limits", {str(k): v for k, v in new.items()} or None)


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


GPU_READING_MAX_AGE = 10.0  # seconds a reading stored by the daemon counts as current


def gpu_readings(con=None) -> list:
    """Current GPU readings. Taken from the daemon's last poll when it is
    recent, so status and the dashboard don't run nvidia-smi (~0.1 s, and
    work for the driver) on every refresh."""
    from . import gpus

    con = con or db.connect()
    at = db.get_kv(con, "gpu_state_at")
    stored = db.get_kv(con, "gpu_state")
    if at and stored is not None and time.time() - at < GPU_READING_MAX_AGE:
        return [gpus.GPU(**g) for g in stored]
    try:
        return gpus.query()
    except gpus.GPUQueryError:
        return []


def cpu_reading(con) -> dict | None:
    """The daemon's last CPU/RAM reading (see cpustat), if it is recent."""
    r = db.get_kv(con, "cpu_state")
    return r if r and time.time() - r.get("at", 0) < GPU_READING_MAX_AGE else None


def live_machine(cfg, con=None) -> Machine:
    con = con or db.connect()
    return machine_from_config(cfg, mem_ok(cfg, gpu_readings(con)), _daemon_capacity(con))


def _daemon_capacity(con) -> dict[int, int] | None:
    """The packing capacity the daemon measured last (so `status` estimates
    match what the daemon will do)."""
    cap = db.get_kv(con, "pack_capacity")
    return {int(k): v for k, v in cap.items()} if cap else None


def filtered_env() -> dict[str, str]:
    """The submitter's environment, minus per-session secrets."""
    drop_prefixes = ("CLAUDE_CODE_MESSAGING_",)
    return {k: v for k, v in os.environ.items() if not k.startswith(drop_prefixes)}
