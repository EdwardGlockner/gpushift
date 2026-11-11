import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
JOBS = Path(__file__).resolve().parent / "jobs"
PY = sys.executable


class Queue:
    """A real gpushift daemon on an isolated GPUSHIFT_HOME with two fake GPUs."""

    def __init__(self, home: Path, monkeypatch):
        self.home = home
        home.mkdir(parents=True, exist_ok=True)
        # Per-process GPU memory the daemon sees: {pid: MiB}; tests fill it in.
        self.gpu_procs = home / "gpu_procs.json"
        self.set_gpu_procs({})
        self.env = dict(os.environ)
        self.env.update(
            GPUSHIFT_HOME=str(home),
            GPUSHIFT_FAKE_GPUS="8192,6144",
            GPUSHIFT_FAKE_GPU_PROCS=str(self.gpu_procs),
            GPUSHIFT_TICK="0.2",
            GPUSHIFT_GPU_POLL="0.5",
        )
        for k, v in self.env.items():
            if k.startswith("GPUSHIFT_"):
                monkeypatch.setenv(k, v)
        self.daemon = None

    def set_gpu_procs(self, used: dict[int, int]) -> None:
        import json

        tmp = self.gpu_procs.with_suffix(".tmp")
        tmp.write_text(json.dumps({str(k): v for k, v in used.items()}))
        os.replace(tmp, self.gpu_procs)

    # -- processes
    def start(self):
        self.daemon = subprocess.Popen(
            [PY, "-m", "gpushift", "daemon"],
            env=self.env,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        from gpushift import daemon

        self.wait_for(daemon.is_running, 10, "daemon start")

    def kill_daemon(self):
        if self.daemon and self.daemon.poll() is None:
            self.daemon.kill()
            self.daemon.wait()

    def gpushift(self, *args, check=True) -> subprocess.CompletedProcess:
        r = subprocess.run(
            [PY, "-m", "gpushift", *args], env=self.env, cwd=ROOT, capture_output=True, text=True, timeout=120
        )
        if check and r.returncode != 0:
            raise AssertionError(f"gpushift {' '.join(args)} failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
        return r

    def submit(self, *args) -> int:
        import json

        r = self.gpushift("submit", "--json", "--no-start", *args)
        return json.loads(r.stdout)["id"]

    # -- state
    @property
    def con(self):
        from gpushift import db

        return db.connect()

    def job(self, jid):
        from gpushift import db

        return db.get_job(self.con, jid)

    def state(self, jid):
        return self.job(jid)["state"]

    def wait_for(self, pred, timeout=20, what="condition"):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if pred():
                return
            time.sleep(0.1)
        raise AssertionError(f"timed out waiting for {what}")

    def wait_state(self, jid, *states, timeout=20):
        self.wait_for(lambda: self.state(jid) in states, timeout, f"job {jid} -> {states} (is {self.state(jid)})")

    def log(self, jid) -> str:
        from gpushift import paths

        f = paths.log_file(jid)
        return f.read_text(errors="replace") if f.exists() else ""

    def kill_everything(self):
        """Simulate a reboot: daemon, runners and jobs all die."""
        from gpushift import _proc, db

        self.kill_daemon()
        for j in db.active(self.con):
            _proc.kill_tree(j["runner_pid"], j["runner_ctime"])
            _proc.kill_tree(j["pid"], j["pid_ctime"])

    def cleanup(self):
        try:
            self.kill_everything()
        except Exception:
            pass


@pytest.fixture
def q(tmp_path, monkeypatch):
    queue = Queue(tmp_path / "gpushift", monkeypatch)
    queue.start()
    yield queue
    queue.cleanup()


def counter(out: Path, steps: int) -> list[str]:
    """A resumable job (uses gpushift.checkpoint)."""
    return ["--", PY, str(JOBS / "counter.py"), str(out), str(steps)]


def ticker(n: int = 600, dt: float = 0.05) -> list[str]:
    """A non-resumable job that prints a line every dt seconds."""
    return ["--", PY, "-c", f"import time\nfor i in range({n}):\n    print(i, flush=True); time.sleep({dt})"]
