"""End-to-end tests against a real daemon (fake GPUs, real processes)."""

import json
import os
import subprocess
import time

import pytest
from conftest import PY, counter, ticker


def test_run_logs_and_env(q):
    jid = q.submit(
        "--one",
        "--",
        PY,
        "-c",
        "import os; print('cvd=' + os.environ['CUDA_VISIBLE_DEVICES'], os.environ['GPUSHIFT_JOB_ID'])",
    )
    q.wait_state(jid, "done")
    assert f"cvd=0 {jid}" in q.log(jid)
    cpu = q.submit("--cpu", "--", PY, "-c", "import os; print('cvd=[' + os.environ['CUDA_VISIBLE_DEVICES'] + ']')")
    q.wait_state(cpu, "done")
    assert "cvd=[]" in q.log(cpu)
    bad = q.submit("--cpu", "--", PY, "-c", "raise SystemExit(3)")
    q.wait_state(bad, "failed")
    assert q.job(bad)["exit_code"] == 3


def test_logs_have_no_blank_lines_between_output_lines(q):
    jid = q.submit("--cpu", "--", PY, "-c", "print('a'); print('b'); print('c')")
    q.wait_state(jid, "done")
    out = q.gpushift("logs", str(jid), "-n", "4").stdout
    assert "a\nb\nc\n" in out


def test_shell_string_command(q):
    jid = q.submit("--cpu", "--", "echo first && echo second")
    q.wait_state(jid, "done")
    log = q.log(jid)
    assert "first" in log and "second" in log


def test_status_json(q):
    import json

    q.submit("--one", *ticker())
    s = json.loads(q.gpushift("status", "--json").stdout)
    assert s["schema"] == 1 and s["mode"] == "normal"
    assert [g["index"] for g in s["gpus"]] == [0, 1]
    assert s["gpus"][1]["shared"] is True


def test_cancel_queued_and_running(q):
    a = q.submit("--two", *ticker())
    b = q.submit("--two", *ticker())
    q.wait_state(a, "running")
    q.wait_for(lambda: q.job(a)["pid"], what="pid")
    assert q.state(b) == "queued"
    q.gpushift("cancel", str(b))
    assert q.state(b) == "cancelled"
    q.gpushift("cancel", str(a))
    q.wait_state(a, "cancelled")


def test_priority_and_swap(q):
    blocker = q.submit("--exclusive", *ticker())
    q.wait_state(blocker, "running")
    a, b, c = (q.submit("--cpu", "--", PY, "-c", "pass") for _ in range(3))
    q.gpushift("priority", str(c))
    q.gpushift("swap", str(a), str(b))
    from gpushift import db

    assert [j["id"] for j in db.queued(q.con)] == [c, b, a]


def test_free_full_evicts_resumable_and_freezes_rest(q, tmp_path):
    res = q.submit("--one", *counter(tmp_path / "c", 150))
    plain = q.submit("--one", *ticker())
    later = q.submit("--cpu", "--", PY, "-c", "print('ran')")
    for j in (res, plain):
        q.wait_state(j, "running")
    q.wait_for(lambda: q.job(res)["pid"] and q.job(plain)["pid"], what="pids")
    q.wait_resumable(res)

    out = q.gpushift("free").stdout
    assert "checkpointed and exited" in out
    assert q.state(res) == "queued" and q.job(res)["resume"] == 1
    assert q.state(plain) == "frozen"

    # frozen really means frozen: no new output
    size = len(q.log(plain))
    time.sleep(1.0)
    assert len(q.log(plain)) == size
    # queue is paused
    assert q.state(later) == "queued"

    out = q.gpushift("resume").stdout
    assert "resumed: job" in out
    q.wait_state(res, "done", timeout=40)
    log = q.log(res)
    assert "checkpointed at step" in log and "resumed at step" in log and "finished at step 150" in log
    assert q.job(res)["evictions"] == 1
    q.wait_state(later, "done", timeout=30)
    time.sleep(0.5)
    assert len(q.log(plain)) > size  # unfrozen


def test_free_soft_then_upgrade_to_full(q, tmp_path):
    res = q.submit("--one", "--resumable", *counter(tmp_path / "c", 300))
    q.wait_resumable(res)
    q.gpushift("free", "--soft")
    assert q.state(res) == "frozen"
    q.gpushift("free")  # upgrade: wakes it up so it can checkpoint
    assert q.state(res) == "queued"
    assert "checkpointed at step" in q.log(res)
    q.gpushift("resume")
    q.wait_state(res, "running", "done")


def test_double_free_and_double_resume(q):
    q.submit("--one", *ticker())
    q.gpushift("free", "--soft")
    out = q.gpushift("free", "--soft").stdout
    assert "already in free mode" in out
    q.gpushift("resume")
    assert "nothing to resume" in q.gpushift("resume").stdout


def test_submit_during_free_waits(q):
    q.gpushift("free", "--soft")
    jid = q.submit("--cpu", "--", PY, "-c", "pass")
    time.sleep(1.0)
    assert q.state(jid) == "queued"
    q.gpushift("resume")
    q.wait_state(jid, "done")


def test_job_dies_while_frozen(q):
    from gpushift import _proc

    jid = q.submit("--one", *ticker())
    q.wait_for(lambda: q.job(jid)["pid"], what="pid")
    q.gpushift("free", "--soft")
    assert q.state(jid) == "frozen"
    j = q.job(jid)
    _proc.kill_tree(j["pid"], j["pid_ctime"])
    q.wait_state(jid, "failed")
    assert "nothing to resume" not in q.gpushift("resume").stdout  # mode was free -> resumes the queue


def test_after_waits_for_a_job_and_is_cancelled_when_it_fails(q):
    import json

    train = q.submit("--one", "--", PY, "-c", "import time; time.sleep(3); print('trained')")
    ev = q.submit("--light", "--after", str(train), "--", PY, "-c", "print('evaluated')")
    doomed = q.submit("--cpu", "--", PY, "-c", "import time; time.sleep(2); raise SystemExit(1)")
    never = q.submit("--cpu", "--after", f"{train},{doomed}", "--", PY, "-c", "print('never')")
    q.wait_state(train, "running")
    s = json.loads(q.gpushift("status", "--json").stdout)
    assert next(j for j in s["queue"] if j["id"] == ev)["waiting_for"] == [train]
    assert q.state(ev) == "queued"  # the light GPU is free, but it waits
    q.wait_state(doomed, "failed")
    q.wait_state(never, "cancelled")
    assert f"job {doomed}" in q.job(never)["note"]
    q.wait_state(train, "done")
    q.wait_state(ev, "done")
    assert q.job(ev)["started_at"] >= q.job(train)["ended_at"]
    # Waiting for a job that already failed is refused at submit time.
    r = q.gpushift("submit", "--cpu", "--after", str(doomed), "--", PY, "-c", "pass", check=False)
    assert r.returncode != 0 and "would never run" in r.stderr
    r = q.gpushift("submit", "--cpu", "--after", "9999", "--", PY, "-c", "pass", check=False)
    assert r.returncode != 0 and "no job 9999" in r.stderr


def test_time_frozen_is_not_run_time(q):
    import json

    jid = q.submit("--one", *ticker())
    q.wait_for(lambda: q.job(jid)["pid"], what="pid")
    q.gpushift("free", "--soft")
    time.sleep(3)
    q.gpushift("resume")
    assert q.job(jid)["paused_s"] >= 2.5
    s = json.loads(q.gpushift("status", "--json").stdout)
    j = next(r for r in s["running"] if r["id"] == jid)
    assert j["runtime_s"] < time.time() - q.job(jid)["started_at"] - 2.5
    q.gpushift("cancel", str(jid))
    q.wait_state(jid, "cancelled")
    assert q.job(jid)["total_runtime"] < time.time() - q.job(jid)["started_at"] - 2.5


def test_eviction_timeout_falls_back_to_freeze(q):
    q.write_config("evict_timeout = 1\n")
    # claims to be resumable but never checks for the stop request (and ignores SIGTERM)
    stubborn = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nprint('ready' + '-now', flush=True)\nwhile True: time.sleep(0.05)"
    jid = q.submit("--one", "--resumable", "--", PY, "-c", stubborn)
    q.wait_for(lambda: "ready-now" in q.log(jid), what="job started")
    q.wait_for(lambda: q.job(jid)["pid"], what="pid")
    out = q.gpushift("free").stdout
    assert "frozen instead" in out
    assert q.state(jid) == "frozen"
    q.gpushift("resume")
    assert q.state(jid) == "running"


def test_reboot_while_free(q, tmp_path):
    res = q.submit("--one", "--resumable", *counter(tmp_path / "c", 30))
    plain = q.submit("--one", *ticker())
    q.wait_for(lambda: q.job(res)["pid"] and q.job(plain)["pid"], what="pids")
    q.gpushift("free", "--soft")
    q.kill_everything()  # "reboot"
    q.start()
    q.wait_state(res, "queued")
    q.wait_state(plain, "failed")
    assert "runner lost" in q.job(plain)["note"]
    from gpushift import db

    assert db.get_mode(q.con) == "soft"  # free mode survives the restart
    time.sleep(1.0)
    assert q.state(res) == "queued"
    q.gpushift("resume")
    q.wait_state(res, "done", timeout=30)


def test_stop_leaves_running_jobs_alone(q):
    import psutil

    jid = q.submit("--one", *ticker())
    q.wait_for(lambda: q.job(jid)["pid"], what="pid")
    pid = q.job(jid)["pid"]
    q.gpushift("stop")
    time.sleep(1.0)
    assert psutil.pid_exists(pid)
    q.start()
    time.sleep(2.0)  # the new daemon adopts it instead of marking it lost
    assert q.state(jid) == "running"
    q.gpushift("cancel", str(jid))
    q.wait_state(jid, "cancelled")


def test_starvation_guard_two_gpu_job_not_starved(q):
    # Long enough that it is still running when the assertions below look.
    first = q.submit("--one", "--", PY, "-c", "import time; time.sleep(8)")
    q.wait_state(first, "running")
    big = q.submit("--two", "--", PY, "-c", "pass")
    smalls = [q.submit("--one", "--", PY, "-c", "import time; time.sleep(2)") for _ in range(3)]
    # GPU 1 is free, but the --two job at the head holds back the --one jobs.
    time.sleep(1.0)
    assert all(q.state(s) == "queued" for s in smalls)
    q.wait_state(big, "done", timeout=30)
    started_big = q.job(big)["started_at"] or q.job(big)["ended_at"]
    for s in smalls:
        q.wait_state(s, "running", "done", timeout=20)
        assert q.job(s)["started_at"] >= started_big - 0.01


def test_retry(q):
    bad = q.submit("--cpu", "--", PY, "-c", "raise SystemExit(1)")
    q.wait_state(bad, "failed")
    out = q.gpushift("retry", str(bad)).stdout
    new = int(out.split()[1])
    q.wait_state(new, "failed")
    assert q.job(new)["command"] == q.job(bad)["command"]


def _git_bash():
    from gpushift import cli

    old = os.environ.get("MSYSTEM")
    os.environ["MSYSTEM"] = "MINGW64"
    try:
        return cli.git_bash()
    finally:
        if old is None:
            del os.environ["MSYSTEM"]
        else:
            os.environ["MSYSTEM"] = old


@pytest.mark.skipif(os.name != "nt" or not _git_bash(), reason="needs Windows with Git Bash")
def test_shell_string_from_git_bash_runs_in_bash(q, tmp_path):
    # What Claude Code's Bash tool on Windows submits: /c/... paths, a
    # redirect, && and cp. cmd.exe cannot run this; the bash it came from can.
    def msys(p):
        s = str(p).replace("\\", "/")
        return f"/{s[0].lower()}{s[2:]}"

    src, dst = tmp_path / "src.txt", tmp_path / "dst.txt"
    script = tmp_path / "probe.py"
    src.write_text("hi")
    script.write_text("print('probe ran')")
    cmd = f"python {msys(script)} > {msys(tmp_path / 'probe.log')} 2>&1 && cp {msys(src)} {msys(dst)}"
    env = dict(q.env, MSYSTEM="MINGW64", PATH=os.path.dirname(PY) + os.pathsep + os.environ["PATH"])
    r = subprocess.run(
        [PY, "-m", "gpushift", "submit", "--json", "--no-start", "--cpu", "--", cmd],
        env=env,
        capture_output=True,
        text=True,
    )
    jid = json.loads(r.stdout)["id"]
    q.wait_state(jid, "done", "failed")
    assert q.state(jid) == "done", q.log(jid)
    assert (tmp_path / "probe.log").read_text().strip() == "probe ran"
    assert dst.read_text() == "hi"
    assert q.job(jid)["command"] == cmd  # shown as typed


@pytest.mark.skipif(os.name != "nt" or not _git_bash(), reason="needs Windows with Git Bash")
def test_windows_style_string_from_git_bash_runs_in_cmd(q, tmp_path):
    # Backslash paths are written for cmd.exe; bash would read them as escapes.
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "probe.py").write_text("open('ran.txt', 'w').write('yes')")
    env = dict(q.env, MSYSTEM="MINGW64", PATH=os.path.dirname(PY) + os.pathsep + os.environ["PATH"])
    r = subprocess.run(
        [PY, "-m", "gpushift", "submit", "--json", "--no-start", "--cpu", "--", r"python sub\probe.py && echo done"],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    jid = json.loads(r.stdout)["id"]
    q.wait_state(jid, "done", "failed")
    assert q.state(jid) == "done", q.log(jid)
    assert (tmp_path / "ran.txt").read_text() == "yes"


def test_half_quoted_command_is_rejected_at_submit(q):
    r = q.gpushift("submit", "--cpu", "--", "python run.py exp_a 1", "mode=static", check=False)
    assert r.returncode == 2
    assert "contains spaces" in r.stderr


def test_free_restarts_restartable_job(q):
    jid = q.submit("--one", "--restartable", *ticker())
    q.wait_for(lambda: q.job(jid)["pid"], what="pid")
    out = q.gpushift("free").stdout
    assert "restarting:" in out
    assert q.state(jid) == "queued"  # stopped and back at the front, not frozen
    assert "will run again" in q.job(jid)["note"]
    q.gpushift("resume")
    q.wait_for(lambda: q.job(jid)["attempts"] == 2 and q.job(jid)["pid"], what="second run")
    assert q.state(jid) == "running"


def test_free_says_which_gpu_frozen_jobs_hold(q):
    jid = q.submit("--one", *ticker())
    q.wait_for(lambda: q.job(jid)["pid"], what="pid")
    out = q.gpushift("free").stdout
    assert "Frozen jobs keep their GPU memory:" in out
    assert "GPU 0" in out and f"gpushift cancel {jid}" in out
    q.gpushift("resume")


def test_short_job_uses_idle_gpu_while_two_gpu_job_waits(q):
    long = q.submit("--one", "--eta", "60s", *ticker(n=400))
    q.wait_state(long, "running")
    big = q.submit("--two", "--eta", "10s", "--", PY, "-c", "pass")
    short = q.submit("--one", "--eta", "5s", "--", PY, "-c", "import time; time.sleep(1)")
    slow = q.submit("--one", "--eta", "2h", "--", PY, "-c", "pass")
    q.wait_state(short, "running", "done")  # ends long before GPU 0 frees: may use GPU 1
    assert q.job(short)["gpus"] == [1]
    time.sleep(1.0)
    assert q.state(big) == "queued"
    assert q.state(slow) == "queued"  # would still be running when the --two job could start


def test_missing_script_or_program_is_rejected_at_submit(q):
    r = q.gpushift("submit", "--cpu", "--", PY, "no_such_script.py", check=False)
    assert r.returncode == 2 and "script not found" in r.stderr
    r = q.gpushift("submit", "--cpu", "--", "no-such-program-xyz", "--flag", check=False)
    assert r.returncode == 2 and "program not found" in r.stderr


def test_failure_note_shows_last_output_line(q):
    jid = q.submit("--cpu", "--", PY, "-c", "import sys; print('loading data'); sys.exit('CUDA out of memory')")
    q.wait_state(jid, "failed")
    assert "CUDA out of memory" in q.job(jid)["note"]


def _hook_notify(q, session, event="PostToolUse"):
    payload = json.dumps({"session_id": session, "hook_event_name": event, "tool_name": "Bash"})
    return subprocess.run(
        [PY, "-m", "gpushift", "hook", "notify"], input=payload, env=q.env, capture_output=True, text=True, timeout=60
    )


def test_session_is_told_when_its_job_ends(q):
    env = dict(q.env, CLAUDE_CODE_SESSION_ID="sess-a")
    r = subprocess.run(
        [PY, "-m", "gpushift", "submit", "--json", "--no-start", "--cpu", "--", PY, "-c", "raise SystemExit('boom')"],
        env=env, capture_output=True, text=True,
    )  # fmt: skip
    jid = json.loads(r.stdout)["id"]
    assert _hook_notify(q, "sess-a").stdout == ""  # still running: nothing to say
    q.wait_state(jid, "failed")
    out = json.loads(_hook_notify(q, "sess-a").stdout)["hookSpecificOutput"]
    assert out["hookEventName"] == "PostToolUse"
    assert f"job {jid}" in out["additionalContext"] and "boom" in out["additionalContext"]
    assert _hook_notify(q, "sess-a").stdout == ""  # told once
    assert _hook_notify(q, "sess-b").stdout == ""  # other sessions are not told


def test_desktop_notification_for_failed_job(q, tmp_path):
    log = tmp_path / "toasts.txt"
    q.kill_daemon()
    q.env["GPUSHIFT_NOTIFY_LOG"] = str(log)
    q.start()
    time.sleep(0.5)
    ok = q.submit("--cpu", "--", PY, "-c", "pass")
    bad = q.submit("--cpu", "--", PY, "-c", "raise SystemExit('bad input')")
    q.wait_state(ok, "done")
    q.wait_state(bad, "failed")
    q.wait_for(lambda: log.exists() and "bad input" in log.read_text(), what="notification")
    assert f"job {ok} " not in log.read_text()  # default: failures only


def test_jobs_with_known_memory_share_a_gpu(q):
    time.sleep(1.0)  # let the daemon measure the GPUs once
    jobs = [q.submit("--one", "--vram", "3G", *ticker()) for _ in range(3)]
    for j in jobs:
        q.wait_state(j, "running")
    gpus = [q.job(j)["gpus"] for j in jobs]
    assert gpus == [[0], [0], [1]]  # 2 x 3 GB fit on the 8 GB GPU; the third goes to GPU 1


def test_memory_need_is_learned_from_earlier_runs(q):
    from gpushift import control

    cmd = ["--", PY, "-c", "import time; time.sleep(3)"]
    first = q.submit("--one", *cmd)
    q.wait_for(lambda: q.job(first)["pid"], what="pid")
    q.set_gpu_procs({q.job(first)["pid"]: 2000})
    q.wait_state(first, "done", timeout=30)
    assert q.job(first)["peak_vram_mb"] == 2000
    second = q.submit("--one", *cmd)
    need = control.vram_need(q.con, q.job(second))
    assert need == int(2000 * control.PACK_MARGIN + control.PACK_EXTRA_MB)


AUTO_FREE_CONFIG = "[auto_free]\nafter = 1\nresume_after = 1\ninclude_agents = true\n"


def _user_program(q):
    """A program the user runs that holds 3 GB of GPU memory."""
    p = subprocess.Popen([PY, "-c", "import time; time.sleep(120)"])
    q.set_gpu_procs({p.pid: 3000})
    return p


def _mode(q):
    from gpushift import db

    return db.get_mode(q.con)


def test_gpus_are_freed_while_the_user_runs_a_heavy_gpu_program(q):
    q.write_config(AUTO_FREE_CONFIG)
    jid = q.submit("--one", *ticker())
    q.wait_for(lambda: q.job(jid)["pid"], what="pid")
    prog = _user_program(q)
    try:
        q.wait_for(lambda: _mode(q) == "free", 15, "automatic free")
        assert q.state(jid) == "frozen"
        s = json.loads(q.gpushift("status", "--json").stdout)
        assert s["auto_free"]["apps"]
    finally:
        prog.kill()
    q.set_gpu_procs({})
    q.wait_for(lambda: _mode(q) == "normal", 15, "automatic resume")
    q.wait_state(jid, "running")


def test_manual_resume_during_auto_free_is_respected(q):
    q.write_config(AUTO_FREE_CONFIG)
    prog = _user_program(q)
    try:
        q.wait_for(lambda: _mode(q) == "free", 15, "automatic free")
        q.gpushift("resume")
        time.sleep(2.5)
        assert _mode(q) == "normal"  # not freed again while that program keeps running
    finally:
        prog.kill()
        q.set_gpu_procs({})


def test_manual_free_is_not_resumed_automatically(q):
    q.write_config(AUTO_FREE_CONFIG)
    q.gpushift("free")
    time.sleep(2.5)
    assert _mode(q) == "free"


def test_dashboard(q, tmp_path):
    import socket
    import urllib.error
    import urllib.request

    with socket.socket() as s:  # a free port
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    q.write_config(f"ui_port = {port}\n")
    q.kill_daemon()
    q.start()
    base = f"http://127.0.0.1:{port}"
    jid = q.submit("--one", *ticker())
    q.wait_state(jid, "running")

    def get(path):
        q.wait_for(lambda: _reachable(base), 10, "dashboard")
        return urllib.request.urlopen(base + path, timeout=10).read()

    assert b"<title>gpushift</title>" in get("/")
    s = json.loads(get("/api/status"))
    assert [j["id"] for j in s["running"]] == [jid]

    def post(path, headers):
        req = urllib.request.Request(base + path, data=b"{}", headers=headers, method="POST")
        try:
            return urllib.request.urlopen(req, timeout=10).status
        except urllib.error.HTTPError as e:
            return e.code

    assert post(f"/api/cancel/{jid}", {}) == 403  # no X-Gpushift header: refused
    assert q.state(jid) == "running"
    assert post(f"/api/cancel/{jid}", {"X-Gpushift": "1"}) == 200
    q.wait_state(jid, "cancelled")


def _reachable(base):
    import urllib.request

    try:
        urllib.request.urlopen(base + "/api/status", timeout=2)
        return True
    except OSError:
        return False


def test_short_whole_machine_job_runs_before_a_long_one_gpu_job(q):
    q.gpushift("free")  # queue both before the daemon decides
    long = q.submit("--one", "--eta", "4h", *ticker())
    short = q.submit("--exclusive", "--eta", "1m", *ticker(20))
    q.gpushift("resume")
    q.wait_state(short, "running")
    assert q.state(long) == "queued"
    q.wait_state(short, "done")
    q.wait_state(long, "running")
    q.gpushift("cancel", str(long))
    from gpushift import paths

    assert f"job {short} runs ahead of job {long}" in paths.daemon_log().read_text()
