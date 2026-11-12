import os
import signal

import pytest

from gpushift import checkpoint, cli, config


@pytest.mark.parametrize(
    "s,sec",
    [("40m", 2400), ("1h30m", 5400), ("90s", 90), ("2h", 7200), ("1.5h", 5400), ("10", 600), ("1d", 86400)],
)
def test_parse_duration(s, sec):
    assert cli.parse_duration(s) == sec


@pytest.mark.parametrize("s", ["abc", "10x", "m10", "1h junk"])
def test_parse_duration_bad(s):
    with pytest.raises(cli.CLIError):
        cli.parse_duration(s)


def test_fmt_dur():
    assert cli.fmt_dur(5) == "5s"
    assert cli.fmt_dur(125) == "2m"
    assert cli.fmt_dur(3700) == "1h01m"
    assert cli.fmt_dur(None) == "?"


def test_config_defaults_scale(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUSHIFT_HOME", str(tmp_path))
    from gpushift.gpus import GPU

    two = [GPU(0, "a", 8192, 0), GPU(1, "b", 6144, 0)]
    c = config.load(two)
    assert (c.gpus, c.slots, c.shared_gpu) == ([0, 1], 4, 1)
    four = [GPU(i, "x", 1, 0) for i in range(4)]
    c = config.load(four)
    assert (c.slots, c.shared_gpu) == (8, 3)
    c = config.load([])
    assert c.gpus == [] and c.shared_gpu is None and c.slots >= 1


def test_config_file_and_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUSHIFT_HOME", str(tmp_path))
    from gpushift.gpus import GPU

    two = [GPU(0, "a", 8192, 0), GPU(1, "b", 6144, 0)]
    (tmp_path / "config.toml").write_text(config.template(two))
    assert config.load(two).slots == 4  # template is all comments
    (tmp_path / "config.toml").write_text("slots = 6\nshared_gpu = 0\n[hook]\nallow = ['x']\n")
    c = config.load(two)
    assert (c.slots, c.shared_gpu, c.hook.allow) == (6, 0, ["x"])
    (tmp_path / "config.toml").write_text("shared_gpu = 5\n")
    with pytest.raises(config.ConfigError):
        config.load(two)
    (tmp_path / "config.toml").write_text("gpus = [0, 3]\n")
    with pytest.raises(config.ConfigError):
        config.load(two)


def test_checkpoint_stop_file(tmp_path, monkeypatch):
    checkpoint._reset_for_tests()
    stop = tmp_path / "stop"
    monkeypatch.setenv("GPUSHIFT_STOP_FILE", str(stop))
    monkeypatch.setenv("GPUSHIFT_JOB_DIR", str(tmp_path / "job"))
    assert not checkpoint.should_stop(poll_interval=0)
    assert (tmp_path / "job" / "resumable").exists()
    stop.touch()
    assert checkpoint.should_stop(poll_interval=0)
    checkpoint._reset_for_tests()


@pytest.mark.skipif(os.name == "nt", reason="SIGTERM to self is not catchable on Windows")
def test_checkpoint_sigterm(monkeypatch):
    checkpoint._reset_for_tests()
    monkeypatch.delenv("GPUSHIFT_STOP_FILE", raising=False)
    old = signal.getsignal(signal.SIGTERM)
    try:
        assert not checkpoint.should_stop()
        os.kill(os.getpid(), signal.SIGTERM)
        assert checkpoint.should_stop()
    finally:
        signal.signal(signal.SIGTERM, old)
        checkpoint._reset_for_tests()


def test_latest_checkpoint_and_atomic(tmp_path):
    assert checkpoint.latest_checkpoint(tmp_path / "missing") is None
    assert checkpoint.latest_checkpoint(tmp_path) is None
    a = tmp_path / "step1.pt"
    a.write_text("1")
    os.utime(a, (1, 1))
    with checkpoint.atomic_path(tmp_path / "step2.pt") as tmp:
        tmp.write_text("2")
    (tmp_path / "step3.pt.gpushift-tmp").write_text("partial")
    assert checkpoint.latest_checkpoint(tmp_path).name == "step2.pt"
    assert checkpoint.latest_checkpoint(tmp_path, "step1*").name == "step1.pt"


def test_resuming(monkeypatch):
    monkeypatch.setenv("GPUSHIFT_RESUME", "1")
    assert checkpoint.resuming()
    monkeypatch.setenv("GPUSHIFT_RESUME", "0")
    assert not checkpoint.resuming()


@pytest.mark.parametrize(
    "command, expected",
    [
        (r"python\.venv\Scripts\python.exe train.py && echo ok", True),
        (r"C:\envs\ml\python.exe train.py", True),
        (r".\run.bat --fast", True),
        ("python /c/Users/me/train.py > log.txt 2>&1 && cp a.csv out/", False),
        ("python C:/Users/me/train.py && python eval.py", False),
        ('echo "a \\"quoted\\" word" && python train.py', False),
    ],
)
def test_windows_style(command, expected):
    assert cli.windows_style(command) is expected


def test_half_quoted_command_is_rejected():
    with pytest.raises(cli.CLIError, match="contains spaces"):
        cli.check_program_name(["python run.py exp_a 1", "mode=static"])
    cli.check_program_name(["python", "run.py", "exp_a", "1"])  # separate words
    cli.check_program_name(["python run.py exp_a 1"])  # one shell string


def test_time_left_unknown_once_overdue():
    from gpushift import control

    j = {"eta_seconds": 600, "total_runtime": 0.0, "state": "running", "started_at": 1000.0}
    assert control.time_left(None, j, now=1100.0) == 500.0
    assert control.time_left(None, j, now=1700.0) is None  # past its estimate: unknown
    resumed = {"eta_seconds": 600, "total_runtime": 400.0, "state": "queued", "started_at": None}
    assert control.time_left(None, resumed, now=0.0) == 200.0


def test_time_spent_frozen_does_not_count_as_run_time():
    from gpushift import control

    j = {"total_runtime": 50.0, "state": "frozen", "started_at": 1000.0, "frozen_at": 1100.0, "paused_s": 0.0}
    assert control.ran(j, now=1700.0) == 150.0  # 100 s run before the freeze, plus 50 s from an earlier attempt
    j |= control.unfreeze(j, 1700.0)
    j["state"] = "running"
    assert j["paused_s"] == 600.0 and j["frozen_at"] is None
    assert control.ran(j, now=1800.0) == 250.0
    assert control.time_left(None, {**j, "eta_seconds": 1000}, now=1800.0) == 750.0


def test_dashboard_log_shows_the_end_and_the_last_state_of_progress_bars(tmp_path, monkeypatch):
    from gpushift import paths, ui

    monkeypatch.setattr(paths, "log_file", lambda jid: tmp_path / f"{jid}.log")
    (tmp_path / "7.log").write_bytes(b"start\r\nstep 1/3\rstep 2/3\rstep 3/3\r\ndone\n")
    assert ui.log_tail(7)["text"] == "start\nstep 3/3\ndone\n"
    monkeypatch.setattr(ui, "LOG_TAIL", 10)
    r = ui.log_tail(7)
    assert r["truncated"] and r["text"] == "done\n"
    assert ui.log_tail(8) == {"id": 8, "path": str(tmp_path / "8.log"), "text": "", "truncated": False}


def test_old_database_gets_new_columns(tmp_path):
    import sqlite3

    from gpushift import db

    f = tmp_path / "old.db"
    old = sqlite3.connect(f)
    old.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY, state TEXT NOT NULL)")
    old.execute("INSERT INTO jobs (id, state) VALUES (1, 'queued')")
    old.commit()
    old.close()
    con = db.connect(f)
    assert con.execute("SELECT restartable FROM jobs WHERE id=1").fetchone()[0] == 0


def _queue_db(tmp_path, jobs):
    """A queue database with (session, eta_seconds, prio) jobs submitted in order."""
    from gpushift import db

    con = db.connect(tmp_path / "q.db")
    for k, (session, eta, prio) in enumerate(jobs, 1):
        db.insert_job(
            con, state="queued", preset="one", n_gpus=1, slots=2, position=float(k), argv=["x"], command=f"x{k}",
            cwd=".", env={}, session=session, eta_seconds=eta, prio=prio,
        )  # fmt: skip
    return con


def test_sessions_take_turns(tmp_path):
    from gpushift import control

    # Session a queued three jobs before b and c queued one each.
    con = _queue_db(tmp_path, [("a", 600, 0), ("a", 600, 0), ("a", 600, 0), ("b", 600, 0), ("c", 600, 0)])
    order = [j["id"] for j in control.ordered_queue(con, fair=True, now=0.0)]
    assert order == [1, 4, 5, 2, 3]
    # Without fair sharing: submit order.
    assert [j["id"] for j in control.ordered_queue(con, fair=False)] == [1, 2, 3, 4, 5]


def test_priority_jobs_go_first_and_long_jobs_count_more(tmp_path):
    from gpushift import control

    con = _queue_db(tmp_path, [("a", 7200, 0), ("a", 60, 0), ("b", 60, 0), ("b", 60, 0), ("c", 60, 1)])
    order = [j["id"] for j in control.ordered_queue(con, fair=True, now=0.0)]
    # c's job was prioritized; a's first job is 2 h, so b gets two turns before a's next one.
    assert order == [5, 1, 3, 4, 2]


def test_dragging_jobs_in_the_queue(tmp_path):
    from gpushift import control, db

    con = _queue_db(tmp_path, [("a", 600, 0)] * 4)
    order = lambda: [j["id"] for j in control.ordered_queue(con)]  # noqa: E731
    assert control.move(con, 4, before=2) == "job 4 moved to before job 2"
    assert order() == [1, 4, 2, 3]
    assert db.get_job(con, 4)["prio"] == 1  # moved up: pinned like --priority
    control.move(con, 1, before=None)
    assert order() == [4, 2, 3, 1]
    assert db.get_job(con, 1)["prio"] == 0
    control.move(con, 3, before=4)
    assert order() == [3, 4, 2, 1]
    for _ in range(60):  # halving the gap again and again eventually renumbers the queue
        control.move(con, 2, before=4)
        control.move(con, 4, before=2)
    assert order() == [3, 4, 2, 1]
    assert "not queued" in control.move(con, 99, before=None)
    assert "not queued" in control.move(con, 2, before=99)


def test_a_job_cannot_be_moved_ahead_of_a_job_it_waits_for(tmp_path):
    from gpushift import control, db

    con = _queue_db(tmp_path, [("a", 600, 0)] * 4)
    db.update(con, 3, after_ids=[2])  # job 3: --after 2
    order = lambda: [j["id"] for j in control.ordered_queue(con)]  # noqa: E731
    assert "can't go ahead" in control.move(con, 3, before=2)
    assert "can't go ahead" in control.move(con, 3, before=1)
    assert "can't go ahead" in control.move(con, 2, before=None)  # the job it waits for, behind it
    assert "can't go ahead" in control.to_front(con, 3)
    assert order() == [1, 2, 3, 4]
    # Moving both, or moving them around each other in the allowed order, is fine.
    assert "moved" in control.move(con, 2, before=1)
    assert "moved" in control.move(con, 3, before=1)
    assert order() == [2, 3, 1, 4]
    assert "moved" in control.to_front(con, 2)


def test_cpu_reading():
    import time

    from gpushift import cpustat

    s = cpustat.Sampler()
    sum(i * i for i in range(2_000_000))  # some CPU work to show up
    time.sleep(0.2)
    r = s.sample()
    assert 0 <= r["load_pct"] <= 100 and len(r["cores_pct"]) >= 1
    assert 0 < r["mem_used_mb"] <= r["mem_total_mb"]
    assert len(r["top"]) <= cpustat.TOP and all(t["pct"] > 0 and t["name"] for t in r["top"])
    assert r["disks"] and all(0 <= d["free_mb"] <= d["total_mb"] for d in r["disks"])
    assert r["io"] is None or (r["io"]["read_mb_s"] >= 0 and r["io"]["write_mb_s"] >= 0)
