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
    evict_requested_at REAL,
    total_runtime REAL NOT NULL DEFAULT 0,
    note TEXT
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""

# History: one row per job state change, mode change, daemon start/stop and
# queue move. Triggers write the first two, so changes made by any process, or
# by an older gpushift, are logged. Columns the triggers read must exist in
# every release.
_NOW = "(julianday('now') - 2440587.5) * 86400.0"
EVENTS = f"""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    t REAL NOT NULL,
    job_id INTEGER,                   -- NULL for mode and daemon events
    kind TEXT NOT NULL,               -- state mode daemon moved retry
    old TEXT,
    new TEXT,
    gpus TEXT,                        -- JSON list
    runtime REAL,                     -- total_runtime, when this change updated it
    exit_code INTEGER,
    note TEXT                         -- the job's note, when this change set a new one
);
CREATE INDEX IF NOT EXISTS events_t ON events(t);
CREATE INDEX IF NOT EXISTS events_job ON events(job_id);
CREATE TRIGGER IF NOT EXISTS events_job_added_v1 AFTER INSERT ON jobs BEGIN
    INSERT INTO events(t, job_id, kind, new, note) VALUES ({_NOW}, NEW.id, 'state', NEW.state, NEW.note);
END;
CREATE TRIGGER IF NOT EXISTS events_job_state_v1 AFTER UPDATE OF state ON jobs
WHEN OLD.state IS NOT NEW.state BEGIN
    INSERT INTO events(t, job_id, kind, old, new, gpus, runtime, exit_code, note) VALUES (
        {_NOW}, NEW.id, 'state', OLD.state, NEW.state, NEW.gpus,
        CASE WHEN NEW.total_runtime IS NOT OLD.total_runtime THEN NEW.total_runtime END,
        NEW.exit_code,
        CASE WHEN NEW.note IS NOT OLD.note THEN NEW.note END
    );
END;
CREATE TRIGGER IF NOT EXISTS events_mode_added_v1 AFTER INSERT ON kv
WHEN NEW.key = 'mode' AND NEW.value IS NOT '"normal"' BEGIN
    INSERT INTO events(t, kind, old, new) VALUES ({_NOW}, 'mode', 'normal', trim(NEW.value, '"'));
END;
CREATE TRIGGER IF NOT EXISTS events_mode_v1 AFTER UPDATE OF value ON kv
WHEN NEW.key = 'mode' AND OLD.value IS NOT NEW.value BEGIN
    INSERT INTO events(t, kind, old, new) VALUES ({_NOW}, 'mode', trim(OLD.value, '"'), trim(NEW.value, '"'));
END;
"""

ACTIVE = ("running", "frozen", "evicting")
FINAL = ("done", "failed", "cancelled")
JSON_FIELDS = ("argv", "env", "gpus", "after_ids")


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or paths.db_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=30, isolation_level=None, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    migrate(con)
    con.executescript(EVENTS)
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


def log_event(con, kind: str, job_id: int | None = None, *, new=None, note=None, t: float | None = None) -> None:
    con.execute(
        f"INSERT INTO events(t, job_id, kind, new, note) VALUES(coalesce(?, {_NOW}), ?, ?, ?, ?)",
        (t, job_id, kind, new, note),
    )


def events(con, job_id: int | None = None, limit: int | None = None) -> list[dict]:
    """Events in time order: one job's, or everyone's. limit keeps the latest."""
    where, args = ("WHERE job_id=?", [job_id]) if job_id is not None else ("", [])
    rows = con.execute(
        f"SELECT * FROM (SELECT * FROM events {where} ORDER BY t DESC, id DESC LIMIT ?) ORDER BY t, id",
        [*args, -1 if limit is None else limit],
    )
    return [decode(r) for r in rows]


def history_runtimes(con, command: str, project: str | None, limit: int = 10) -> list[float]:
    rows = con.execute(
        "SELECT total_runtime FROM jobs WHERE state='done' AND command=? AND project IS ? "
        "AND total_runtime > 0 ORDER BY ended_at DESC LIMIT ?",
        (command, project, limit),
    ).fetchall()
    return [r["total_runtime"] for r in rows]
