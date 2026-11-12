"""The web dashboard: a small HTTP server on 127.0.0.1 that the daemon runs
in a thread. `gpushift ui` opens it.

GET  /              the page (ui.html)
GET  /api/status    `gpushift status --json` plus per-job GPU memory
GET  /api/log/<id>  the end of a job's log
POST /api/free      {"soft": bool}
POST /api/resume
POST /api/cancel/<id>, /api/priority/<id>
POST /api/move/<id>  {"before": <id> | null}  (null: to the end)

POSTs must carry the header ``X-Gpushift: 1``; browsers can't add it to requests
from other sites, so other web pages can't drive the queue. The Host header
must be localhost, which stops DNS-rebinding tricks. Like the hook, this is a
guardrail for a single-user machine, not a security boundary.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config, control, db, paths

log = logging.getLogger("gpushift.daemon")
PAGE = Path(__file__).with_name("ui.html")
RECENT = 10  # finished jobs the dashboard lists
LOG_TAIL = 64 * 1024  # bytes of a job's log the dashboard shows


def log_tail(job_id: int) -> dict:
    """The last lines of a job's log (at most LOG_TAIL bytes)."""
    f = paths.log_file(job_id)
    try:
        with open(f, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - LOG_TAIL))
            text = fh.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return {"id": job_id, "path": str(f), "text": "", "truncated": False}
    lines = text.split("\n")
    if size > LOG_TAIL:
        lines = lines[1:]  # the first line is cut off
    # Progress bars redraw a line with \r: keep what the line shows last.
    lines = [ln.rstrip("\r").rsplit("\r", 1)[-1] for ln in lines]
    text = "\n".join(lines)
    return {"id": job_id, "path": str(f), "text": text, "truncated": size > LOG_TAIL}


def status_payload() -> dict:
    from .cli import job_public, status_data

    cfg = config.load()
    con = db.connect()
    s = status_data(con, cfg)
    s.pop("_est", None)
    s["recent"] = [job_public(con, j) for j in db.recent(con, RECENT)]
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
    m = re.fullmatch(r"/api/move/(\d+)", path)
    if m:
        before = body.get("before")
        msg = control.move(con, int(m.group(1)), None if before is None else int(before))
        return {"ok": "moved" in msg, "message": msg, "error": msg}
    m = re.fullmatch(r"/api/(cancel|priority)/(\d+)", path)
    if not m:
        return {"ok": False, "error": "unknown action"}
    action, jid = m.group(1), int(m.group(2))
    if action == "cancel":
        return {"ok": True, "message": control.cancel(con, jid)}
    msg = control.to_front(con, jid)
    return {"ok": "moved" in msg, "message": msg, "error": msg}


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
            m = re.fullmatch(r"/api/log/(\d+)", self.path)
            if m:
                return self._json(200, log_tail(int(m.group(1))))
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


class Server(ThreadingHTTPServer):
    # On Windows, SO_REUSEADDR lets a second process bind the same port and
    # take over the dashboard; ask for the port exclusively instead.
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        # Not HTTPServer.server_bind: it looks up the host's name, which can
        # hang for many seconds on macOS (and the daemon waits for this).
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def serve(port: int) -> ThreadingHTTPServer | None:
    """Start the dashboard in a background thread. None if the port is taken."""
    try:
        srv = Server(("127.0.0.1", port), Handler)
    except OSError as e:
        log.warning("dashboard not started on port %s: %s", port, e)
        return None
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name="gpushift-ui", daemon=True).start()
    log.info("dashboard on http://127.0.0.1:%s", port)
    return srv
