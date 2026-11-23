"""The queue database (SQLite, WAL). The CLI, the daemon and job runners all
read and write it; state changes use BEGIN IMMEDIATE transactions with
conditional updates so concurrent writers cannot clobber each other."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from pathlib import Path

from . import paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state TEXT NOT NULL,              -- queued running frozen evicting done failed cancelled
    preset TEXT NOT NULL,
    n_gpus INTEGER NOT NULL,
    slots INTEGER NOT NULL,
    exclusive INTEGER NOT NULL DEFAULT 0,
    shared INTEGER NOT NULL DEFAULT 0,
    position REAL NOT NULL,
    argv TEXT NOT NULL,               -- JSON list
    command TEXT NOT NULL,            -- display string
    shell INTEGER NOT NULL DEFAULT 0,
    cwd TEXT NOT NULL,
    env TEXT NOT NULL,                -- JSON object
    tag TEXT,
    project TEXT,
    session TEXT,
    eta_seconds INTEGER,
    resumable INTEGER NOT NULL DEFAULT 0,
    restartable INTEGER NOT NULL DEFAULT 0,
    prio INTEGER NOT NULL DEFAULT 0,  -- moved to the front with --priority / gpushift priority
    notified INTEGER NOT NULL DEFAULT 0,  -- the submitting session was told it ended
    vram_mb INTEGER,                  -- GPU memory the job needs per GPU (--vram, or learned)
    peak_vram_mb INTEGER,             -- most GPU memory it was seen using
    ram_mb INTEGER,                   -- RAM the job needs (--ram)
    peak_ram_mb INTEGER,              -- most RAM it was seen using (all its processes)
    submitted_at REAL NOT NULL,
    started_at REAL,
    ended_at REAL,
    gpus TEXT,                        -- JSON list of assigned GPU indices
    runner_pid INTEGER,
    runner_ctime REAL,
    pid INTEGER,
    pid_ctime REAL,
    exit_code INTEGER,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    evictions INTEGER NOT NULL DEFAULT 0,
    resume INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    frozen_at REAL,
    paused_s REAL NOT NULL DEFAULT 0, -- seconds the current run spent frozen
    after_ids TEXT,                   -- JSON list: jobs that must finish first (--after)
    only_gpus TEXT,                   -- JSON list: the GPUs it may run on (--gpu); NULL = any
    stall_note TEXT,                  -- why the daemon thinks it may be stuck (set once)
    stall_told INTEGER NOT NULL DEFAULT 0,  -- its session was told it may be stuck
    evict_requested_at REAL,
    total_runtime REAL NOT NULL DEFAULT 0,
    note TEXT
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
-- The Claude Code process each session runs in. /clear starts a new session
-- in the same process, which is how its jobs find their way to the new one.
CREATE TABLE IF NOT EXISTS sessions (
    session TEXT PRIMARY KEY,
    claude_pid INTEGER NOT NULL,
    claude_ctime REAL NOT NULL,
    seen_at REAL NOT NULL,
    cleared_into TEXT                 -- the session /clear replaced it with
);
"""

ACTIVE = ("running", "frozen", "evicting")
FINAL = ("done", "failed", "cancelled")
JSON_FIELDS = ("argv", "env", "gpus", "after_ids", "only_gpus")


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or paths.db_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=30, isolation_level=None, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    migrate(con)
    return con


# Columns added after the first release: (name, declaration).
ADDED_COLUMNS = [
    ("restartable", "INTEGER NOT NULL DEFAULT 0"),
    ("prio", "INTEGER NOT NULL DEFAULT 0"),
    ("notified", "INTEGER NOT NULL DEFAULT 0"),
    ("vram_mb", "INTEGER"),
    ("peak_vram_mb", "INTEGER"),
    ("paused_s", "REAL NOT NULL DEFAULT 0"),
    ("after_ids", "TEXT"),
    ("ram_mb", "INTEGER"),
    ("peak_ram_mb", "INTEGER"),
    ("only_gpus", "TEXT"),
    ("stall_note", "TEXT"),
    ("stall_told", "INTEGER NOT NULL DEFAULT 0"),
]


def migrate(con: sqlite3.Connection) -> None:
    """Add columns that older databases lack."""
    have = {r["name"] for r in con.execute("PRAGMA table_info(jobs)")}
    for name, decl in ADDED_COLUMNS:
        if name in have:
            continue
        try:
            con.execute(f"ALTER TABLE jobs ADD COLUMN {name} {decl}")
        except sqlite3.OperationalError as e:  # another process added it first
            if "duplicate column" not in str(e):
                raise


@contextlib.contextmanager
def tx(con: sqlite3.Connection):
    con.execute("BEGIN IMMEDIATE")
    try:
        yield con
    except BaseException:
        con.execute("ROLLBACK")
        raise
    else:
        con.execute("COMMIT")


def get_kv(con, key: str, default=None):
    row = con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def set_kv(con, key: str, value) -> None:
    con.execute(
        "INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value)),
    )


def get_mode(con) -> str:
    return get_kv(con, "mode", "normal")


def decode(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    for k in JSON_FIELDS:
        if d.get(k) is not None:
            d[k] = json.loads(d[k])
    return d


def get_job(con, job_id: int) -> dict | None:
    return decode(con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())


def queued(con) -> list[dict]:
    return [decode(r) for r in con.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY position, id")]


def active(con) -> list[dict]:
    return [
        decode(r)
        for r in con.execute(
            "SELECT * FROM jobs WHERE state IN ('running','frozen','evicting') ORDER BY started_at, id"
        )
    ]


def recent(con, limit: int = 5) -> list[dict]:
    return [
        decode(r)
        for r in con.execute(
            "SELECT * FROM jobs WHERE state IN ('done','failed','cancelled') ORDER BY ended_at DESC, id DESC LIMIT ?",
            (limit,),
        )
    ]


def front_position(con) -> float:
    row = con.execute("SELECT MIN(position) AS p FROM jobs WHERE state='queued'").fetchone()
    return (row["p"] if row["p"] is not None else 0.0) - 1.0


def back_position(con) -> float:
    row = con.execute("SELECT MAX(position) AS p FROM jobs").fetchone()
    return (row["p"] if row["p"] is not None else 0.0) + 1.0


def update(con, job_id: int, *, where_state: str | tuple | None = None, **fields) -> bool:
    """Update a job. With where_state, only if it is still in that state.
    Returns whether a row changed."""
    for k in JSON_FIELDS:
        if k in fields and fields[k] is not None:
            fields[k] = json.dumps(fields[k])
    sets = ", ".join(f"{k}=?" for k in fields)
    sql = f"UPDATE jobs SET {sets} WHERE id=?"
    params = [*fields.values(), job_id]
    if where_state is not None:
        states = (where_state,) if isinstance(where_state, str) else tuple(where_state)
        sql += f" AND state IN ({','.join('?' * len(states))})"
        params += list(states)
    return con.execute(sql, params).rowcount > 0


def insert_job(con, **fields) -> int:
    for k in JSON_FIELDS:
        if k in fields and fields[k] is not None:
            fields[k] = json.dumps(fields[k])
    fields.setdefault("submitted_at", time.time())
    cols = ", ".join(fields)
    qs = ", ".join("?" * len(fields))
    cur = con.execute(f"INSERT INTO jobs({cols}) VALUES({qs})", list(fields.values()))
    return cur.lastrowid


def history_runtimes(con, command: str, project: str | None, limit: int = 10) -> list[float]:
    rows = con.execute(
        "SELECT total_runtime FROM jobs WHERE state='done' AND command=? AND project IS ? "
        "AND total_runtime > 0 ORDER BY ended_at DESC LIMIT ?",
        (command, project, limit),
    ).fetchall()
    return [r["total_runtime"] for r in rows]


def history_runs(con, command: str, project: str | None, limit: int = 10) -> list[tuple[float, list[int]]]:
    """(runtime, GPUs it ran on) of recent successful runs, newest first."""
    rows = con.execute(
        "SELECT total_runtime, gpus FROM jobs WHERE state='done' AND command=? AND project IS ? "
        "AND total_runtime > 0 ORDER BY ended_at DESC LIMIT ?",
        (command, project, limit),
    ).fetchall()
    return [(r["total_runtime"], json.loads(r["gpus"]) if r["gpus"] else []) for r in rows]


def eta_runs(con, limit: int) -> list[tuple[str | None, float, int]]:
    """(project, runtime, --eta) of recent successful runs that had an --eta
    and were never evicted, newest first."""
    rows = con.execute(
        "SELECT project, total_runtime, eta_seconds FROM jobs WHERE state='done' AND eta_seconds > 0 "
        "AND evictions=0 AND total_runtime > 0 ORDER BY ended_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [(r["project"], r["total_runtime"], r["eta_seconds"]) for r in rows]


def single_gpu_runs(con, limit: int) -> list[tuple[str, str | None, int, float]]:
    """(command, project, GPU, runtime) of recent successful one-GPU runs that
    stayed on one GPU (never evicted), newest first."""
    rows = con.execute(
        "SELECT command, project, gpus, total_runtime FROM jobs WHERE state='done' AND n_gpus=1 "
        "AND evictions=0 AND total_runtime > 0 AND gpus IS NOT NULL ORDER BY ended_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    out = []
    for r in rows:
        g = json.loads(r["gpus"])
        if len(g) == 1:
            out.append((r["command"], r["project"], g[0], r["total_runtime"]))
    return out


SESSION_KEEP_S = 30 * 86400  # forget sessions not seen for this long


def known_session(con, session: str) -> bool:
    return con.execute("SELECT 1 FROM sessions WHERE session=?", (session,)).fetchone() is not None


def remember_session(con, session: str, pid: int, ctime: float) -> None:
    """Record that ``session`` runs in the Claude process (pid, ctime) (call inside a tx)."""
    now = time.time()
    con.execute(
        "INSERT INTO sessions(session, claude_pid, claude_ctime, seen_at) VALUES(?, ?, ?, ?) "
        "ON CONFLICT(session) DO UPDATE SET claude_pid=excluded.claude_pid, claude_ctime=excluded.claude_ctime, "
        "seen_at=excluded.seen_at",
        (session, pid, ctime, now),
    )
    con.execute("DELETE FROM sessions WHERE seen_at < ?", (now - SESSION_KEEP_S,))


def hand_over(con, session: str, pid: int, ctime: float) -> tuple[str | None, list[dict]]:
    """After /clear: ``session`` is new in the Claude process (pid, ctime).
    Moves the jobs of the session that ran there before (unfinished, or ended
    but not yet reported) to it. Returns (old session, moved jobs); call inside a tx."""
    row = con.execute(
        "SELECT session FROM sessions WHERE claude_pid=? AND ABS(claude_ctime - ?) < 1.0 AND session != ? "
        "AND cleared_into IS NULL ORDER BY seen_at DESC LIMIT 1",
        (pid, ctime, session),
    ).fetchone()
    remember_session(con, session, pid, ctime)
    if row is None:
        return None, []
    old = row["session"]
    con.execute("UPDATE sessions SET cleared_into=? WHERE session=?", (session, old))
    jobs = [
        decode(r)
        for r in con.execute(
            "SELECT * FROM jobs WHERE session=? AND (state NOT IN ('done','failed','cancelled') OR notified=0) "
            "ORDER BY id",
            (old,),
        )
    ]
    for j in jobs:
        update(con, j["id"], session=session)
        j["session"] = session
    return old, jobs


def successor(con, session: str | None) -> str | None:
    """The session that /clear (once or several times) replaced ``session`` with, if any."""
    seen = set()
    nxt = None
    while session and session not in seen:
        seen.add(session)
        row = con.execute("SELECT cleared_into FROM sessions WHERE session=?", (session,)).fetchone()
        session = row["cleared_into"] if row else None
        nxt = session or nxt
    return nxt


def take_stalled(con, session: str | None, job_id: int | None = None) -> list[dict]:
    """This session's running jobs (or just job ``job_id``) the daemon flagged
    as possibly stuck and nobody was told about; marks them as told (call
    inside a tx)."""
    sql = "SELECT * FROM jobs WHERE state='running' AND stall_note IS NOT NULL AND stall_told=0 AND "
    rows = con.execute(
        sql + ("id=?" if job_id is not None else "session IS ?") + " ORDER BY id",
        (job_id if job_id is not None else session,),
    ).fetchall()
    jobs = [decode(r) for r in rows]
    for j in jobs:
        update(con, j["id"], stall_told=1)
    return jobs
