import json
import sqlite3

from conftest import PY, counter, ticker

from gpushift import control, db


def _job(con, **kw) -> int:
    fields = dict(
        state="queued", preset="one", n_gpus=1, slots=2, position=1.0, argv=["x"], command="x", cwd=".", env={}
    )
    return db.insert_job(con, **{**fields, **kw})


def _changes(con, job_id=None) -> list[tuple]:
    return [(e["old"], e["new"]) for e in db.events(con, job_id)]


def test_every_job_state_change_is_an_event(tmp_path):
    con = db.connect(tmp_path / "q.db")
    jid = _job(con)
    db.update(con, jid, state="running", gpus=[1])
    db.update(con, jid, state="running", peak_vram_mb=900)  # not a state change
    db.update(con, jid, state="queued", gpus=None, total_runtime=5.0, note="evicted")
    db.update(con, jid, state="running", gpus=[0])
    db.update(con, jid, state="done", exit_code=0, total_runtime=9.0)

    evs = db.events(con, jid)
    assert _changes(con, jid) == [
        (None, "queued"),
        ("queued", "running"),
        ("running", "queued"),
        ("queued", "running"),
        ("running", "done"),
    ]
    assert [e["gpus"] for e in evs] == [None, [1], None, [0], [0]]
    assert [e["runtime"] for e in evs] == [None, None, 5.0, None, 9.0]
    # The note is kept only on the change that set it, not on later ones.
    assert [e["note"] for e in evs] == [None, None, "evicted", None, None]
    assert evs[-1]["exit_code"] == 0


def test_free_and_resume_are_events(tmp_path):
    con = db.connect(tmp_path / "q.db")
    control.resume(con)  # already normal
    control.free(con, soft=True)
    control.free(con, soft=False)
    control.free(con, soft=False)
    control.resume(con)
    assert _changes(con) == [("normal", "soft"), ("soft", "free"), ("free", "normal")]
    db.set_kv(con, "daemon_heartbeat", 1.0)
    assert len(db.events(con)) == 3


def test_queue_moves_are_events(tmp_path):
    con = db.connect(tmp_path / "q.db")
    a, b, c = (_job(con, position=float(k)) for k in range(3))
    control.to_front(con, c)
    control.move(con, c, None)
    control.move(con, b, a)
    notes = [(e["job_id"], e["note"]) for e in db.events(con) if e["kind"] == "moved"]
    assert notes == [(c, "moved to the front"), (c, "moved to the end"), (b, f"moved before job {a}")]


def test_events_come_in_time_order_and_limit_keeps_the_latest(tmp_path):
    con = db.connect(tmp_path / "q.db")
    db.log_event(con, "daemon", new="started", t=200.0)
    db.log_event(con, "daemon", new="stopped", t=100.0)
    db.log_event(con, "daemon", new="started", t=300.0)
    assert [e["t"] for e in db.events(con)] == [100.0, 200.0, 300.0]
    assert [e["t"] for e in db.events(con, limit=2)] == [200.0, 300.0]


def test_database_from_before_the_event_log(tmp_path):
    f = tmp_path / "old.db"
    old = sqlite3.connect(f)
    old.executescript(db.SCHEMA)
    old.execute(
        "INSERT INTO jobs (state, preset, n_gpus, slots, position, argv, command, cwd, env, submitted_at) "
        "VALUES ('queued', 'one', 1, 2, 1.0, '[]', 'x', '.', '{}', 1.0)"
    )
    old.commit()
    old.close()
    con = db.connect(f)
    assert db.events(con) == []
    db.update(con, 1, state="running")
    assert _changes(con) == [("queued", "running")]
    assert db.get_job(con, 1)["command"] == "x"


# ---------------------------------------------------------------- with a real queue


def _events(q, job_id=None) -> list[dict]:
    args = ["events", "--json"] + ([str(job_id)] if job_id is not None else [])
    out = json.loads(q.gpushift(*args).stdout)
    assert out["schema"] == 1
    return out["events"]


def _steps(evs) -> list[tuple]:
    return [(e["from"], e["to"]) for e in evs if e["kind"] == "state"]


def test_free_and_resume_with_a_job_on_each_gpu(q, tmp_path):
    res = q.submit("--one", *counter(tmp_path / "c", 150))
    plain = q.submit("--one", *ticker())
    q.wait_for(lambda: q.job(res)["pid"] and q.job(plain)["pid"], what="pids")
    q.wait_resumable(res)
    q.gpushift("free")
    q.gpushift("resume")
    q.wait_state(res, "done", timeout=40)
    q.gpushift("cancel", str(plain))
    q.wait_state(plain, "cancelled")

    r, p = _events(q, res), _events(q, plain)
    assert _steps(r) == [
        (None, "queued"),
        ("queued", "running"),
        ("running", "evicting"),
        ("evicting", "queued"),
        ("queued", "running"),
        ("running", "done"),
    ]
    assert "evicted" in r[3]["note"] and r[3]["runtime_s"] is not None
    assert _steps(p) == [
        (None, "queued"),
        ("queued", "running"),
        ("running", "frozen"),
        ("frozen", "running"),
        ("running", "cancelled"),
    ]
    assert {r[1]["gpus"][0], p[1]["gpus"][0]} == {0, 1}
    modes = [(e["from"], e["to"]) for e in _events(q) if e["kind"] == "mode"]
    assert modes == [("normal", "free"), ("free", "normal")]


def test_reboot_shows_when_the_daemon_was_down(q):
    jid = q.submit("--one", *ticker())
    q.wait_for(lambda: q.job(jid)["pid"], what="pid")
    q.kill_everything()
    q.start()
    q.wait_state(jid, "failed")

    daemon = [e["to"] for e in _events(q) if e["kind"] == "daemon"]
    assert daemon == ["started", "stopped", "started"]
    assert "runner lost" in _events(q, jid)[-1]["note"]
    text = q.gpushift("events", str(jid)).stdout
    assert "queued -> running on GPU" in text and "running -> failed" in text


def test_retry_and_swap_are_events(q):
    bad = q.submit("--cpu", "--", PY, "-c", "raise SystemExit(1)")
    q.wait_state(bad, "failed")
    new = int(q.gpushift("retry", str(bad)).stdout.split()[1])
    assert [e["note"] for e in _events(q, new) if e["kind"] == "retry"] == [f"retry of job {bad}"]

    q.gpushift("free")
    a, b = (q.submit("--cpu", "--", PY, "-c", "pass") for _ in range(2))
    q.gpushift("swap", str(a), str(b))
    assert [e["note"] for e in _events(q, a) if e["kind"] == "moved"] == [f"swapped with job {b}"]
    q.gpushift("resume")
