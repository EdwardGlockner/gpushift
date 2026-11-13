"""Claude Code PreToolUse hook: redirect direct GPU launches to `gpushift submit`.

Wired up by the plugin as ``gpushift hook pretooluse`` for the Bash and
PowerShell tools. Reads the hook JSON on stdin; exit code 2 blocks the call
and the stderr message is shown to the agent.

Design rule: false positives (blocking harmless commands) are worse than
misses. Anything unparseable is allowed. Escape hatches:
  * append ``# no-queue`` to the command, or prefix it with ``GPUSHIFT_NO_QUEUE=1``
  * ``[hook] allow = [regex, ...]`` in the config (``deny`` forces a block)
  * ``[hook] enabled = false``
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from dataclasses import dataclass

TOOLS = ("Bash", "PowerShell")

LAUNCHERS = {"torchrun", "deepspeed", "horovodrun", "colossalai"}
LAUNCH_MODULES = {"torch.distributed.run", "torch.distributed.launch", "accelerate.commands.launch", "deepspeed"}
# train*.py, finetune_x.py, run_train.py, llama_sft.py -- but not plot_training_curves.py
TRAIN_NAME = re.compile(
    r"^(train|finetune|fine_tune|fine-tune|pretrain)"
    r"|[_\-](train|training|trainer|finetune|fine_tune|pretrain|sft|dpo|grpo)$"
    r"|^(sft|dpo|grpo)$",
    re.I,
)
PYTHON = re.compile(r"^(python(\d+(\.\d+)?)?|py|pythonw)(\.exe)?$", re.I)
WRAPPERS = {"nohup", "time", "exec", "env", "nice", "stdbuf", "caffeinate", "&", "call"}
RUNNERS = {  # tool run <cmd>
    "uv": "run",
    "poetry": "run",
    "pipenv": "run",
    "pdm": "run",
    "hatch": "run",
    "rye": "run",
    "pixi": "run",
}
QUICK_FLAGS = {"--help", "-h", "--version", "-V", "--dry-run", "--dry_run"}
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
PS_ENV = re.compile(r"^\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=", re.I)
NO_QUEUE = re.compile(r"(#\s*no-queue\b|\bGPUSHIFT_NO_QUEUE=1\b|\$env:GPUSHIFT_NO_QUEUE\s*=)", re.I)


@dataclass
class Verdict:
    block: bool
    reason: str = ""
    segment: str = ""
    preset: str = "one"


# ---------------------------------------------------------------- parsing


def split_segments(cmd: str) -> list[str]:
    """Split on unquoted ; && || | & and newlines."""
    segs, cur, q, i = [], [], None, 0
    while i < len(cmd):
        c = cmd[i]
        if q:
            cur.append(c)
            if c == q:
                q = None
            elif c == "\\" and q == '"' and i + 1 < len(cmd):
                cur.append(cmd[i + 1])
                i += 1
        elif c in "\"'":
            q = c
            cur.append(c)
        elif c == "#" and (not cur or cur[-1].isspace()):
            # comment to end of line
            while i < len(cmd) and cmd[i] != "\n":
                i += 1
            continue
        elif c in ";|&\n":
            segs.append("".join(cur))
            cur = []
            while i + 1 < len(cmd) and cmd[i + 1] in "|&":
                i += 1
        else:
            cur.append(c)
        i += 1
    segs.append("".join(cur))
    return [s.strip() for s in segs if s.strip()]


def tokens(seg: str) -> list[str]:
    # Backslashes are path separators far more often than escapes here
    # (C:\envs\python.exe); only classification uses these tokens.
    seg = seg.replace("\\", "/")
    try:
        return shlex.split(seg, posix=True)
    except ValueError:
        return seg.split()


def basename(tok: str) -> str:
    return re.split(r"[\\/]", tok)[-1]


def strip_prefixes(t: list[str]) -> tuple[list[str], dict[str, str]]:
    """Drop env assignments, wrappers and `uv run`-style runners."""
    env = {}
    changed = True
    while t and changed:
        changed = False
        if ASSIGN.match(t[0]):
            k, _, v = t[0].partition("=")
            env[k] = v
            t = t[1:]
            changed = True
        elif basename(t[0]).lower() in WRAPPERS:
            t = t[1:]
            while t and t[0].startswith("-"):  # wrapper options, e.g. nice -n 10
                t = t[2:] if t[0] in ("-n", "-i", "-o", "-e") else t[1:]
            changed = True
        elif basename(t[0]).lower() in RUNNERS and len(t) > 1 and t[1] == RUNNERS[basename(t[0]).lower()]:
            t = t[2:]
            while t and t[0].startswith("-"):
                t = t[2:] if t[0] in ("--with", "--python", "-p", "--env-file", "-e") else t[1:]
            changed = True
        elif basename(t[0]).lower() == "conda" and len(t) > 1 and t[1] == "run":
            t = t[2:]
            while t and t[0].startswith("-"):
                t = t[2:] if t[0] in ("-n", "--name", "-p", "--prefix") else t[1:]
            changed = True
    return t, env


VISIBLE_VARS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


def classify(seg: str, env_vars: set[str]) -> Verdict:
    t, env = strip_prefixes(tokens(seg))
    if not t:
        return Verdict(False)
    prog = basename(t[0]).lower()
    prog_noext = re.sub(r"\.(exe|cmd|bat)$", "", prog)
    args = t[1:]
    if prog_noext in ("gpushift", "ts", "tsp"):
        return Verdict(False)
    if any(a in QUICK_FLAGS for a in args):
        return Verdict(False)
    cvd = next((v for v in VISIBLE_VARS if v in env or v in env_vars), None)

    def multi() -> str:
        joined = " ".join(args)
        m = re.search(r"--(nproc[_-]per[_-]node|num[_-]processes|num[_-]gpus|gpus)[= ](\d+)", joined)
        return "two" if (m and int(m.group(2)) > 1) or "--multi_gpu" in args or "--multi-gpu" in args else "one"

    if prog_noext in LAUNCHERS:
        return Verdict(True, f"`{prog_noext}` launches distributed GPU training", seg, multi())
    if prog_noext == "accelerate" and args[:1] == ["launch"]:
        return Verdict(True, "`accelerate launch` starts GPU training", seg, multi())
    if PYTHON.match(prog):
        i = 0
        while i < len(args) and args[i].startswith("-") and args[i] not in ("-m", "-c"):
            i += 1
        if i >= len(args):
            return Verdict(False)
        if args[i] == "-c":
            return Verdict(False)
        if args[i] == "-m":
            mod = args[i + 1] if i + 1 < len(args) else ""
            if mod in LAUNCH_MODULES:
                return Verdict(True, f"`python -m {mod}` launches distributed GPU training", seg, multi())
            if mod.split(".")[0] in (
                "pytest",
                "pip",
                "venv",
                "http",
                "json",
                "unittest",
                "ensurepip",
                "pydoc",
                "gpushift",
            ):
                return Verdict(False)
            if TRAIN_NAME.search(mod.split(".")[-1]):
                return Verdict(True, f"`python -m {mod}` looks like a training run", seg, "one")
            if cvd:
                return Verdict(True, f"it sets {cvd} and runs Python", seg, "one")
            return Verdict(False)
        script = basename(args[i])
        stem = re.sub(r"\.py$", "", script)
        if script.endswith(".py") and TRAIN_NAME.search(stem):
            return Verdict(True, f"`{script}` looks like a training script", seg, "one")
        if cvd:
            return Verdict(True, f"it sets {cvd} and runs Python", seg, "one")
    return Verdict(False)


def is_free_or_resume(seg: str, env_vars: set[str]) -> bool:
    """`gpushift free` / `gpushift resume` run as a command (not just mentioned in a
    quoted string such as a commit message), without --help, and not marked as
    run by the user (the /gpushift:free and /gpushift:resume slash commands set GPUSHIFT_BY_USER=1)."""
    t, env = strip_prefixes(tokens(seg))
    if len(t) < 2 or re.sub(r"\.exe$", "", basename(t[0]).lower()) != "gpushift" or t[1] not in ("free", "resume"):
        return False
    if any(a in ("--help", "-h") for a in t[2:]):
        return False
    return env.get("GPUSHIFT_BY_USER") != "1" and "GPUSHIFT_BY_USER" not in env_vars


def check(command: str, allow: list[str] = (), deny: list[str] = ()) -> Verdict:
    if NO_QUEUE.search(command):
        return Verdict(False)
    # PowerShell: $env:CUDA_VISIBLE_DEVICES = "0"; python x.py
    env_vars = {m.group(1) for s in split_segments(command) for m in [PS_ENV.match(s)] if m}
    for seg in split_segments(command):
        if is_free_or_resume(seg, env_vars):
            return Verdict(
                True, "free/resume affects every session on this machine and is reserved for the user", seg, "free"
            )
    for pat in allow:
        if re.search(pat, command):
            return Verdict(False)
    for pat in deny:
        if re.search(pat, command):
            return Verdict(True, f"it matches the configured deny pattern {pat!r}", command, "one")
    for seg in split_segments(command):
        v = classify(seg, env_vars)
        if v.block:
            return v
    return Verdict(False)


def message(v: Verdict) -> str:
    if v.preset == "free":
        return (
            "Blocked by gpushift: `gpushift free` / `gpushift resume` pause or resume GPU work for every session on this machine, "
            "so only the user may run them. Do not retry. If you need the machine freed, ask the user to run "
            "/gpushift:free (or `gpushift free` in their terminal)."
        )
    seg = v.segment.strip()
    return (
        f"Blocked by gpushift: {v.reason}. GPUs on this machine are shared through a queue; do not launch GPU work directly.\n"
        f"Submit it instead:\n\n    gpushift submit --{v.preset} --eta <estimate> -- {seg}\n\n"
        "Presets: --light (small job sharing a GPU), --one (1 GPU), --two (all GPUs), --exclusive (whole machine), "
        "--cpu (no GPU). Check `gpushift status` first, then keep working and check back with "
        "`gpushift logs <id>` / `gpushift wait <id>`.\n"
        "If this command does not use a GPU (or is a quick check), re-run it with `# no-queue` appended."
    )


def _hook_config() -> dict:
    """The [hook] table, read without importing gpushift.config (keeps the hook fast)."""
    from . import paths

    f = paths.config_file()
    if not f.exists():
        return {}
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover
        import tomli as tomllib
    with open(f, "rb") as fh:
        return tomllib.load(fh).get("hook", {}) or {}


def notify_session(data: dict) -> str | None:
    """Ended jobs this Claude session submitted and has not been told about,
    as a note for its context; marks them as told."""
    from . import db, notify, paths

    session = data.get("session_id")
    if not session or not paths.db_file().exists():
        return None
    con = db.connect()
    with db.tx(con):
        rows = con.execute(
            "SELECT * FROM jobs WHERE session=? AND notified=0 AND state IN ('done','failed','cancelled') ORDER BY id",
            (session,),
        ).fetchall()
        jobs = [db.decode(r) for r in rows]
        for j in jobs:
            db.update(con, j["id"], notified=1)
    return notify.for_agent(jobs) if jobs else None


def main(event: str = "pretooluse") -> int:
    if event == "notify":
        # PostToolUse / UserPromptSubmit: tell the agent about its ended jobs.
        try:
            data = json.load(sys.stdin)
            text = notify_session(data)
        except Exception:
            return 0
        if text:
            name = data.get("hook_event_name") or "PostToolUse"
            print(json.dumps({"hookSpecificOutput": {"hookEventName": name, "additionalContext": text}}))
        return 0
    try:
        data = json.load(sys.stdin)
        if data.get("tool_name") not in TOOLS:
            return 0
        command = (data.get("tool_input") or {}).get("command") or ""
        if not command:
            return 0
        allow, deny = [], []
        try:
            h = _hook_config()
            if h.get("enabled", True) is False:
                return 0
            allow, deny = list(h.get("allow", [])), list(h.get("deny", []))
        except Exception:
            pass
        v = check(command, allow, deny)
    except Exception:
        return 0
    if v.block:
        print(message(v), file=sys.stderr)
        return 2
    return 0
