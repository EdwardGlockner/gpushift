"""The web dashboard: a small HTTP server on 127.0.0.1 that the daemon runs
in a thread. `gpushift ui` opens it.

GET  /              the page (ui.html)
GET  /api/status    `gpushift status --json` plus per-job GPU memory
POST /api/free      {"soft": bool}
POST /api/resume
POST /api/cancel/<id>, /api/priority/<id>, /api/retry/<id>

POSTs must carry the header ``X-Gpushift: 1``; browsers can't add it to requests
from other sites, so other web pages can't drive the queue. The Host header
must be localhost, which stops DNS-rebinding tricks. Like the hook, this is a
guardrail for a single-user machine, not a security boundary.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config, control, db

log = logging.getLogger("gpushift.daemon")
PAGE = Path(__file__).with_name("ui.html")


def status_payload() -> dict:
    from .cli import status_data

    cfg = config.load()
    con = db.connect()
    s = status_data(con, cfg)
    s.pop("_est", None)
    vram = db.get_kv(con, "job_vram") or {}
    for j in s["running"]:
        j["vram_now_mb"] = vram.get(str(j["id"]))
    s["fair_share"] = cfg.fair_share
    return s


def _act(path: str, body: dict) -> dict:
    con = db.connect()
    if path == "/api/free":
        rep = control.free(con, soft=bool(body.get("soft")))
        return {"ok": True, "report": rep}
    if path == "/api/resume":
        return {"ok": True, "report": control.resume(con)}
    m = re.fullmatch(r"/api/(cancel|priority|retry)/(\d+)", path)
    if not m:
        return {"ok": False, "error": "unknown action"}
    action, jid = m.group(1), int(m.group(2))
    if action == "cancel":
        return {"ok": True, "message": control.cancel(con, jid)}
    with db.tx(con):
        j = db.get_job(con, jid)
        if j is None:
            return {"ok": False, "error": f"job {jid} not found"}
        if action == "priority":
            if j["state"] != "queued":
                return {"ok": False, "error": f"job {jid} is {j['state']}, not queued"}
            db.update(con, jid, position=db.front_position(con), prio=1)
            return {"ok": True, "message": f"job {jid} moved to the front"}
        keep = ("preset", "n_gpus", "slots", "exclusive", "shared", "argv", "command", "shell", "cwd", "env", "tag",
                "project", "session", "eta_seconds", "resumable", "restartable", "vram_mb")  # fmt: skip
        new = db.insert_job(con, state="queued", position=db.back_position(con), **{k: j[k] for k in keep})
    return {"ok": True, "message": f"job {new} queued (retry of job {jid})"}


class Handler(BaseHTTPRequestHandler):
    server_version = "gpushift"

    def log_message(self, fmt, *args):  # keep the daemon log quiet
        pass

    def _local(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in ("127.0.0.1", "localhost", "[::1]")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if not self._local():
            return self._json(403, {"error": "forbidden"})
        try:
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/status":
                return self._json(200, status_payload())
            return self._json(404, {"error": "not found"})
        except Exception as e:
            log.exception("ui: GET %s failed", self.path)
            return self._json(500, {"error": str(e)})

    def do_POST(self):
        if not self._local() or self.headers.get("X-Gpushift") != "1":
            return self._json(403, {"error": "forbidden"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            res = _act(self.path, body)
            return self._json(200 if res.get("ok") else 400, res)
        except Exception as e:
            log.exception("ui: POST %s failed", self.path)
            return self._json(500, {"ok": False, "error": str(e)})


def serve(port: int) -> ThreadingHTTPServer | None:
    """Start the dashboard in a background thread. None if the port is taken."""
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        log.warning("dashboard not started on port %s: %s", port, e)
        return None
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name="gpushift-ui", daemon=True).start()
    log.info("dashboard on http://127.0.0.1:%s", port)
    return srv
