"""Pure scheduling logic: presets -> resource requests, which queued jobs can
start now, and a simulation for estimated start times. No I/O here.

Resource model
--------------
* Slots: a machine-wide concurrency budget (``slots`` in the config).
* GPUs: a non-shared job claims its GPUs exclusively. ``--light`` jobs share
  the configured shared GPU with each other, but never with a non-shared job
  (so a light eval cannot OOM a training run).
* ``--exclusive`` needs every slot, i.e. nothing else may run.

Starvation guard (in-order head with reservations)
--------------------------------------------------
* A waiting multi-GPU or exclusive job holds back every job behind it.
* A waiting 1-GPU job holds back the GPU jobs behind it; CPU jobs may pass.
* A held-back job may still start on idle resources if it fits now and its
  expected duration says it ends before the waiting job could start anyway
  (when the running jobs it waits for are expected to end). Jobs without an
  estimate, or waiting on jobs without one, are simply held back.
Without the guard, any job that fits may start, like task-spooler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PRESETS = ("light", "one", "two", "exclusive", "cpu")


class PresetError(ValueError):
    pass


@dataclass(frozen=True)
class Req:
    id: int
    n_gpus: int
    slots: int
    exclusive: bool = False
    shared: bool = False
    vram: int | None = None  # MiB needed per GPU, if known (makes the job packable)

    @property
    def barrier(self) -> bool:
        return self.exclusive or self.n_gpus >= 2

    @property
    def uses_gpu(self) -> bool:
        return self.n_gpus > 0 or self.shared


@dataclass
class Machine:
    gpus: list[int]
    shared_gpu: int | None
    slots: int
    mem_ok: set[int] | None = None  # GPUs whose free memory passes the threshold; None = all
    # MiB per GPU that packed jobs may book (memory minus use outside gpushift);
    # None turns packing off.
    capacity: dict[int, int] | None = None


def preset_requirements(
    preset: str,
    n_gpus_total: int,
    total_slots: int,
    *,
    gpus: int | None = None,
    slots: int | None = None,
) -> dict:
    """Translate a preset (plus optional --gpus/--slots overrides) into a
    resource request. Raises PresetError for impossible requests."""
    if preset not in PRESETS:
        raise PresetError(f"unknown preset {preset!r}; choose one of {', '.join(PRESETS)}")
    if preset == "light":
        req = dict(n_gpus=1, slots=1, exclusive=False, shared=True)
    elif preset == "one":
        req = dict(n_gpus=1, slots=2, exclusive=False, shared=False)
    elif preset == "two":
        req = dict(n_gpus=n_gpus_total, slots=2, exclusive=False, shared=False)
    elif preset == "exclusive":
        req = dict(n_gpus=n_gpus_total, slots=total_slots, exclusive=True, shared=False)
    else:  # cpu
        req = dict(n_gpus=0, slots=1, exclusive=False, shared=False)

    if gpus is not None:
        if preset in ("light", "exclusive"):
            raise PresetError(f"--gpus cannot be combined with --{preset}")
        req["n_gpus"] = gpus
    if slots is not None:
        if preset == "exclusive":
            raise PresetError("--slots cannot be combined with --exclusive")
        req["slots"] = slots

    if n_gpus_total == 0 and (preset in ("light", "one", "two") or req["n_gpus"] > 0):
        raise PresetError("no GPUs available to gpushift (CPU-only mode); use --cpu")
    if req["n_gpus"] > n_gpus_total:
        raise PresetError(f"job needs {req['n_gpus']} GPUs but gpushift can only use {n_gpus_total}")
    if req["slots"] < 1:
        raise PresetError("--slots must be >= 1")
    # A job may never need more slots than exist, or it could never start.
    req["slots"] = min(req["slots"], total_slots)
    return req


class Usage:
    """Resources taken by running jobs."""

    def __init__(self, m: Machine):
        self.m = m
        self.slots = 0
        self.count = 0
        self.heavy: set[int] = set()  # GPUs held whole by one job
        self.light: dict[int, int] = {}  # GPU -> number of --light jobs on it
        self.packed: dict[int, int] = {}  # GPU -> MiB booked by packed jobs
        self.exclusive = False

    def packable(self, req: Req) -> bool:
        """Jobs with a known memory need share GPUs with each other; they use
        1 slot, since sharing a GPU is the point."""
        return req.vram is not None and self.m.capacity is not None and not req.exclusive and not req.shared

    def cost(self, req: Req) -> int:
        return 1 if self.packable(req) and req.n_gpus > 0 else req.slots

    def add(self, req: Req, gpus: list[int]) -> None:
        self.slots += self.cost(req)
        self.count += 1
        self.exclusive = self.exclusive or req.exclusive
        for g in gpus:
            if req.shared:
                self.light[g] = self.light.get(g, 0) + 1
            elif self.packable(req):
                self.packed[g] = self.packed.get(g, 0) + req.vram
            else:
                self.heavy.add(g)

    def fit(self, req: Req) -> list[int] | None:
        """GPUs to give this job if it can start now, else None."""
        m = self.m
        if self.exclusive or self.slots + self.cost(req) > m.slots:
            return None
        if req.exclusive:
            if self.count:
                return None
            if m.mem_ok is not None and any(g not in m.mem_ok for g in m.gpus):
                return None
            return list(m.gpus)
        if req.shared:
            g = m.shared_gpu
            if g is None or g in self.heavy or g in self.packed:
                return None
            return [g]
        if req.n_gpus == 0:
            return []
        if self.packable(req):
            cap = m.capacity or {}
            fits = [
                g
                for g in m.gpus
                if g not in self.heavy and not self.light.get(g) and self.packed.get(g, 0) + req.vram <= cap.get(g, 0)
            ]
            # Fill GPUs that are already shared first (tightest fit), so whole
            # GPUs stay free for jobs that need one; keep the --light GPU for last.
            fits.sort(key=lambda g: (g not in self.packed, g == m.shared_gpu, cap.get(g, 0) - self.packed.get(g, 0), g))
            return sorted(fits[: req.n_gpus]) if len(fits) >= req.n_gpus else None
        free = [
            g
            for g in m.gpus
            if g not in self.heavy
            and g not in self.packed
            and not self.light.get(g)
            and (m.mem_ok is None or g in m.mem_ok)
        ]
        # Keep the shared GPU for light jobs when another GPU will do.
        free.sort(key=lambda g: (g == m.shared_gpu, g))
        if len(free) < req.n_gpus:
            return None
        return sorted(free[: req.n_gpus])


def reservation(req: Req, ends: list[tuple[Req, list[int], float | None]], machine: Machine) -> float:
    """Seconds from now until ``req`` could start, if the jobs in ``ends``
    (req, gpus, expected seconds left; None = unknown) end as expected and
    nothing else starts. math.inf when it depends on a job with no estimate."""
    order = sorted(ends, key=lambda e: math.inf if e[2] is None else e[2])
    for k in range(len(order) + 1):
        u = Usage(machine)
        for r, g, _ in order[k:]:
            u.add(r, g)
        if u.fit(req) is not None:
            if k == 0:
                return 0.0
            left = order[k - 1][2]
            return math.inf if left is None else left
    return math.inf


def plan(
    queued: list[Req],
    running: list[tuple[Req, list[int]]],
    machine: Machine,
    guard: bool = True,
    durations: dict[int, float | None] | None = None,
    remaining: dict[int, float | None] | None = None,
) -> list[tuple[int, list[int]]]:
    """Which queued jobs (in queue order) start now, and on which GPUs.

    durations (queued job id -> expected seconds) and remaining (running job
    id -> expected seconds left) let held-back jobs use idle resources when
    they will be done before the job holding them back could start. Without
    them, held-back jobs wait."""
    durations = durations or {}
    remaining = remaining or {}
    u = Usage(machine)
    ends: list[tuple[Req, list[int], float | None]] = []
    for req, gpus in running:
        u.add(req, gpus)
        ends.append((req, gpus, remaining.get(req.id)))
    starts: list[tuple[int, list[int]]] = []
    barrier_waiting = gpu_waiting = False
    deadline = math.inf  # held-back jobs must be done by then

    def start(req: Req, g: list[int]) -> None:
        u.add(req, g)
        ends.append((req, g, durations.get(req.id)))
        starts.append((req.id, g))

    for req in queued:
        g = u.fit(req)
        if guard and (barrier_waiting or (gpu_waiting and req.uses_gpu)):
            dur = durations.get(req.id)
            if g is not None and dur is not None and dur <= deadline < math.inf:
                start(req, g)
            continue
        if g is None:
            if guard and (req.barrier or req.uses_gpu):
                deadline = min(deadline, reservation(req, ends, machine))
                barrier_waiting = barrier_waiting or req.barrier
                gpu_waiting = True
            continue
        start(req, g)
    return starts


def simulate(
    queued: list[tuple[Req, float | None]],
    running: list[tuple[Req, list[int], float | None]],
    machine: Machine,
    guard: bool = True,
) -> dict[int, float | None]:
    """Estimated start offset (seconds from now) for each queued job.

    queued:  (req, expected duration or None) in queue order
    running: (req, gpus, expected remaining seconds or None)
    None means unknown (a job ahead of it has no runtime estimate).
    machine.mem_ok (GPUs busy outside gpushift) only applies to "now"; later
    rounds optimistically assume that external usage is gone."""
    m = Machine(machine.gpus, machine.shared_gpu, machine.slots, None, machine.capacity)
    first = True
    t = 0.0
    act = [(r, g, (rem if rem is not None else math.inf)) for r, g, rem in running]
    pending = list(queued)
    est: dict[int, float | None] = {r.id: None for r, _ in queued}
    # Each round either finishes or starts at least one job, so this bound holds.
    for _ in range(2 * (len(pending) + len(act)) + 2):
        if not pending:
            break
        starts = dict(
            plan(
                [r for r, _ in pending],
                [(r, g) for r, g, _ in act],
                machine if first else m,
                guard,
                durations={r.id: d for r, d in pending},
                remaining={r.id: (None if e == math.inf else e - t) for r, _, e in act},
            )
        )
        first = False
        still = []
        for r, dur in pending:
            if r.id in starts:
                est[r.id] = t
                act.append((r, starts[r.id], t + dur if dur is not None else math.inf))
            else:
                still.append((r, dur))
        pending = still
        if not pending:
            break
        finite = [e for _, _, e in act if e != math.inf]
        if not finite:
            break
        t = min(finite)
        act = [a for a in act if a[2] > t]
    return est
