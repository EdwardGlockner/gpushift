import math

import pytest

from gpushift.scheduler import Machine, PresetError, Req, plan, preset_requirements, reorder, simulate

# ------------------------------------------------------------ presets


@pytest.mark.parametrize(
    "preset,n,slots,expect",
    [
        # 2 GPUs, 4 slots (the user's machine)
        ("light", 2, 4, dict(n_gpus=1, slots=1, exclusive=False, shared=True)),
        ("one", 2, 4, dict(n_gpus=1, slots=2, exclusive=False, shared=False)),
        ("two", 2, 4, dict(n_gpus=2, slots=2, exclusive=False, shared=False)),
        ("exclusive", 2, 4, dict(n_gpus=2, slots=4, exclusive=True, shared=False)),
        ("cpu", 2, 4, dict(n_gpus=0, slots=1, exclusive=False, shared=False)),
        # 1 GPU, 2 slots
        ("light", 1, 2, dict(n_gpus=1, slots=1, exclusive=False, shared=True)),
        ("one", 1, 2, dict(n_gpus=1, slots=2, exclusive=False, shared=False)),
        ("two", 1, 2, dict(n_gpus=1, slots=2, exclusive=False, shared=False)),
        ("exclusive", 1, 2, dict(n_gpus=1, slots=2, exclusive=True, shared=False)),
        # 4 GPUs, 8 slots
        ("two", 4, 8, dict(n_gpus=4, slots=2, exclusive=False, shared=False)),
        ("exclusive", 4, 8, dict(n_gpus=4, slots=8, exclusive=True, shared=False)),
        ("one", 4, 8, dict(n_gpus=1, slots=2, exclusive=False, shared=False)),
        # CPU-only, 2 slots
        ("cpu", 0, 2, dict(n_gpus=0, slots=1, exclusive=False, shared=False)),
        ("exclusive", 0, 2, dict(n_gpus=0, slots=2, exclusive=True, shared=False)),
    ],
)
def test_presets(preset, n, slots, expect):
    assert preset_requirements(preset, n, slots) == expect


@pytest.mark.parametrize("preset", ["light", "one", "two"])
def test_gpu_presets_rejected_on_cpu_only(preset):
    with pytest.raises(PresetError, match="CPU-only"):
        preset_requirements(preset, 0, 2)


def test_overrides_and_limits():
    assert preset_requirements("one", 4, 8, gpus=3)["n_gpus"] == 3
    assert preset_requirements("cpu", 2, 4, slots=3)["slots"] == 3
    assert preset_requirements("one", 2, 1)["slots"] == 1  # clamped to total
    with pytest.raises(PresetError):
        preset_requirements("one", 2, 4, gpus=3)
    with pytest.raises(PresetError):
        preset_requirements("light", 2, 4, gpus=2)
    with pytest.raises(PresetError):
        preset_requirements("nope", 2, 4)


# ------------------------------------------------------------ plan

M2 = Machine(gpus=[0, 1], shared_gpu=1, slots=4)


def one(i):
    return Req(i, 1, 2)


def two(i):
    return Req(i, 2, 2)


def light(i):
    return Req(i, 1, 1, shared=True)


def cpu(i):
    return Req(i, 0, 1)


def excl(i, m=M2):
    return Req(i, len(m.gpus), m.slots, exclusive=True)


def test_one_prefers_non_shared_gpu():
    assert plan([one(1)], [], M2) == [(1, [0])]
    assert plan([one(1), one(2)], [], M2) == [(1, [0]), (2, [1])]


def test_light_jobs_share_but_never_with_heavy():
    assert plan([light(1), light(2)], [], M2) == [(1, [1]), (2, [1])]
    # heavy job on the shared GPU blocks light jobs
    assert plan([light(3)], [(one(1), [0]), (one(2), [1])], M2) == []
    # light jobs on GPU 1 keep a --one job off it
    assert plan([one(3)], [(light(1), [1]), (one(2), [0])], M2) == []


def test_slots_limit():
    # two --one jobs use all 4 slots -> cpu job waits
    assert plan([cpu(3)], [(one(1), [0]), (one(2), [1])], M2) == []
    assert plan([cpu(3)], [(one(1), [0])], M2) == [(3, [])]


def test_exclusive_needs_empty_machine():
    assert plan([excl(1)], [], M2) == [(1, [0, 1])]
    assert plan([excl(2)], [(cpu(1), [])], M2) == []
    assert plan([cpu(2)], [(excl(1), [0, 1])], M2) == []


def test_mem_threshold():
    m = Machine([0, 1], 1, 4, mem_ok={1})
    assert plan([one(1)], [], m) == [(1, [1])]
    assert plan([excl(2)], [], m) == []
    # light jobs ignore the memory threshold
    assert plan([light(3)], [], Machine([0, 1], 1, 4, mem_ok=set())) == [(3, [1])]


def test_starvation_two_gpu_job_blocks_followers():
    running = [(one(1), [0])]
    queue = [two(2), one(3), one(4), cpu(5)]
    # guard on: nothing passes the waiting --two job
    assert plan(queue, running, M2, guard=True) == []
    # guard off (task-spooler behaviour): --one slips past and --two starves
    assert plan(queue, running, M2, guard=False) == [(3, [1])]


def test_starvation_stream_of_one_jobs():
    """A --two job behind a stream of --one jobs eventually runs (guard on)."""
    running = [(one(1), [0]), (one(2), [1])]
    queue = [two(3)] + [one(i) for i in range(4, 10)]
    # job 1 ends: --two is head and still blocked, GPU 0 must stay free
    running = [(one(2), [1])]
    assert plan(queue, running, M2) == []
    # job 2 ends: --two starts first
    assert plan(queue, [], M2)[0] == (3, [0, 1])


def test_blocked_one_job_lets_cpu_jobs_pass_but_not_gpu_jobs():
    running = [(one(1), [0]), (light(2), [1])]
    queue = [one(3), light(4), cpu(5)]
    assert plan(queue, running, M2) == [(5, [])]


# ------------------------------------------------------------ simulate


def test_simulate_estimates():
    running = [(one(1), [0], 600.0), (one(2), [1], 1200.0)]
    queue = [(one(3), 300.0), (two(4), 100.0), (cpu(5), None)]
    est = simulate(queue, running, M2)
    assert est[3] == 600.0  # GPU 0 frees at 600
    assert est[4] == 1200.0  # needs both GPUs: after job 2 (1200) and job 3 (900)
    assert est[5] == 1200.0  # held behind the --two job (guard), then fits beside it


def test_simulate_unknown():
    est = simulate([(one(2), 60.0)], [(one(1), [0], None), (one(9), [1], None)], M2)
    assert est[2] is None


def test_simulate_external_gpu_usage_only_blocks_now():
    # GPU 0 busy outside gpushift, GPU 1 runs job 1 for 300s: job 2 can't start now
    m = Machine([0, 1], 1, 4, mem_ok={1})
    est = simulate([(one(2), 60.0)], [(one(1), [1], 300.0)], m)
    assert est[2] == 300.0


# ------------------------------------------------------------ idle GPUs while a big job waits


def test_short_job_uses_idle_gpu_while_two_gpu_job_waits():
    # GPU 0 busy for another 600 s; the --two job can start in 600 s at best.
    running, remaining = [(one(1), [0])], {1: 600.0}
    queue = [two(2), one(3), one(4)]
    # job 3 (300 s) is done before then, so it may use idle GPU 1; job 4 (900 s) may not
    durations = {2: 100.0, 3: 300.0, 4: 900.0}
    assert plan(queue, running, M2, durations=durations, remaining=remaining) == [(3, [1])]


def test_no_idle_gpu_use_without_estimates():
    running = [(one(1), [0])]
    queue = [two(2), one(3)]
    # the running job has no estimate: nobody can tell when the --two job could start
    assert plan(queue, running, M2, durations={3: 60.0}, remaining={1: None}) == []
    # the waiting job has no estimate: it can't promise to be done in time
    assert plan(queue, running, M2, durations={}, remaining={1: 600.0}) == []


def test_waiting_one_job_reserves_its_gpu():
    # Both GPUs busy; job 3 (--one) waits for GPU 0 (free in 100 s).
    running, remaining = [(one(1), [0]), (light(2), [1])], {1: 100.0, 2: 5000.0}
    queue = [one(3), light(4), light(5)]
    # light jobs fit beside job 2 on GPU 1; only the short one may start
    durations = {4: 50.0, 5: 500.0}
    assert plan(queue, running, M2, durations=durations, remaining=remaining) == [(4, [1])]


def test_simulate_counts_idle_gpu_use():
    running = [(one(1), [0], 600.0)]
    queue = [(two(2), 100.0), (one(3), 300.0)]
    est = simulate(queue, running, M2)
    assert est[3] == 0.0  # uses GPU 1 now, done at 300
    assert est[2] == 600.0  # still starts as soon as GPU 0 frees


# ------------------------------------------------------------ packing

CAP = {0: 7000, 1: 6000}  # MiB packed jobs may book per GPU
MP = Machine(gpus=[0, 1], shared_gpu=1, slots=4, capacity=CAP)


def sized(i, mb, n=1):
    return Req(i, n, 2, vram=mb)


def test_jobs_with_known_memory_share_a_gpu():
    # Three 3 GB jobs: two fit together on GPU 0, the third goes to GPU 1.
    starts = plan([sized(1, 3000), sized(2, 3000), sized(3, 3000)], [], MP)
    assert starts == [(1, [0]), (2, [0]), (3, [1])]


def test_packing_fills_shared_gpus_first_and_keeps_whole_gpus_free():
    # Job 1 (2 GB) already runs on GPU 1; a 2 GB job joins it, so GPU 0 stays whole.
    assert plan([sized(2, 2000)], [(sized(1, 2000), [1])], MP) == [(2, [1])]
    # A job with unknown memory needs a whole GPU: GPU 0 is still free for it.
    assert plan([one(3)], [(sized(1, 2000), [1]), (sized(2, 2000), [1])], MP) == [(3, [0])]


def test_unknown_jobs_never_share_with_packed_ones():
    # GPU 0 has a packed job, GPU 1 a whole-GPU job: an unknown job can't start.
    assert plan([one(3)], [(sized(1, 1000), [0]), (one(2), [1])], MP) == []
    # ... and a packed job can't join the whole-GPU job either.
    assert plan([sized(4, 1000)], [(one(2), [1])], MP) == [(4, [0])]


def test_packing_respects_memory_and_can_be_off():
    assert plan([sized(1, 8000)], [], MP) == []  # bigger than any GPU's capacity
    off = Machine(gpus=[0, 1], shared_gpu=1, slots=4)  # capacity None: packing off
    assert plan([sized(1, 1000), sized(2, 1000), sized(3, 1000)], [], off) == [(1, [0]), (2, [1])]


def test_a_job_that_cannot_fit_now_holds_nothing_back():
    # The desktop holds 1.5 GB of GPU 0, so a 7.7 GB job fits nowhere, even
    # with no jobs running: the jobs behind it run meanwhile.
    m = Machine(gpus=[0, 1], shared_gpu=1, slots=4, capacity={0: 6700, 1: 6144})
    running = [(one(1), [0])]
    assert plan([sized(2, 7700), one(3)], running, m) == [(3, [1])]
    est = simulate([(sized(2, 7700), 60.0), (one(3), 60.0)], [(one(1), [0], 600.0)], m)
    assert est[2] is None and est[3] == 0.0
    # Once it would fit on an idle machine, it holds the others back again.
    m = Machine(gpus=[0, 1], shared_gpu=1, slots=4, capacity={0: 8192, 1: 6144})
    assert plan([sized(2, 7700), one(3)], running, m) == []
    # Same for a whole-machine job while a program outside gpushift uses GPU 0.
    m = Machine([0, 1], 1, 4, mem_ok={1})
    assert plan([two(2), one(3)], [], m) == [(3, [1])]


# ------------------------------------------------------------ running jobs ahead of their turn

H = 3600.0
Q = 900.0  # a 15-minute whole-machine job


def test_short_whole_machine_jobs_go_before_a_long_one_gpu_job():
    # Empty machine. In turn order the 4.5 h one-GPU job starts first and the
    # eight 15-minute exclusive jobs wait for it, leaving GPU 0 idle for hours.
    queue = [(one(1), 4.5 * H)] + [(excl(10 + i), Q) for i in range(8)]
    base = simulate(queue, [], M2)
    assert base[1] == 0 and base[10] == 4.5 * H
    res = reorder(queue, [], M2)
    # Half its own run time (2.25 h) is what job 1 may lose: all eight fit in 2 h.
    assert res.moved == list(range(10, 18)) and res.passed == [1]
    assert res.est[10] == 0 and res.est[1] == 2 * H
    assert plan([r for r, _ in queue if r.id in res.moved] + [one(1)], [], M2)[0] == (10, [0, 1])


def test_reorder_never_delays_a_job_more_than_allowed():
    # Job 1 runs 1 h: it may be pushed back max(1 h, 30 min) = 1 h, so only
    # four of the 15-minute jobs may go first.
    queue = [(one(1), H)] + [(excl(10 + i), Q) for i in range(8)]
    res = reorder(queue, [], M2)
    assert res.moved == [10, 11, 12, 13] and res.est[1] == H
    assert reorder(queue, [], M2, slack=0).moved == []  # off


def test_a_long_job_does_not_take_an_idle_gpu_if_it_delays_the_next_job_for_hours():
    # Tonight: GPU 1 busy 4.5 h, a 15-minute exclusive job waits for it, and a
    # 16 h job could use idle GPU 0, but would push the exclusive job back 11.5 h.
    running = [(one(1), [1], 4.5 * H)]
    queue = [(excl(2), Q), (one(3), 16 * H)]
    assert reorder(queue, running, M2).moved == []


def test_prioritized_jobs_are_never_pushed_back():
    queue = [(one(1), 4.5 * H)] + [(excl(10 + i), Q) for i in range(8)]
    assert reorder(queue, [], M2, pinned={1}).moved == []


def test_promises_stop_a_job_being_passed_again_and_again():
    # Job 1 (1 h) may lose 1 h in total, however many times the daemon reorders.
    dur = {1: H, **{10 + i: Q for i in range(8)}}
    queue = [one(1)] + [excl(10 + i) for i in range(8)]
    promised: dict[int, float] = {}
    t, ran = 0.0, []
    while True:
        res = reorder([(r, dur[r.id]) for r in queue], [], M2, limits={k: v - t for k, v in promised.items()})
        if not res.moved:
            break
        for i, off in res.limits.items():
            promised[i] = min(promised.get(i, math.inf), t + off)
        ran.append(res.moved[0])  # it runs alone (exclusive), then the daemon decides again
        queue = [r for r in queue if r.id != res.moved[0]]
        t += Q
    assert ran == [10, 11, 12, 13] and t == H


def test_equal_jobs_keep_their_order():
    # Job 11 is a bit shorter than job 10, but the same size: it never passes it.
    queue = [(excl(10), Q), (excl(11), Q - 120), (one(12), 16 * H)]
    assert reorder(queue, [], M2).moved == []
    est = simulate(queue, [], M2, slack=H)
    assert est[10] < est[11] < est[12]


def test_forecast_applies_reorders_later_on():
    # GPU 1 busy 4 h. When it frees up, the short exclusive jobs run before the
    # 16 h job, so they are forecast at 4 h, 4 h 15, ... not after 20 h.
    running = [(one(1), [1], 4 * H)]
    queue = [(excl(10), Q), (one(12), 16 * H), (excl(11), Q)]
    plain = simulate(queue, running, M2)
    assert plain[11] == 4 * H + Q + 16 * H
    est = simulate(queue, running, M2, slack=H)
    assert (est[10], est[11], est[12]) == (4 * H, 4 * H + Q, 4 * H + 2 * Q)


def test_a_job_waiting_for_another_neither_starts_nor_holds_anything_back():
    # Eval 11 waits for train 10 (4 h on GPU 0). GPU 1 is free: job 12 behind
    # the eval starts now; the eval is forecast for when the training ends.
    running = [(one(10), [0], 4 * H)]
    queue = [(one(11), Q), (one(12), H)]
    est = simulate(queue, running, M2, after={11: {10}})
    assert est == {11: 4 * H, 12: 0.0}
    # Even an exclusive job that waits does not block the queue behind it.
    est = simulate([(excl(11), Q), (one(12), H)], running, M2, after={11: {10}})
    assert est[12] == 0.0 and est[11] == 4 * H
    # Waiting for a running job without an estimate: start unknown.
    assert simulate(queue, [(one(10), [0], None)], M2, after={11: {10}})[11] is None
    # Waiting for a queued job: after that one ends.
    out = {}
    est = simulate([(one(20), H), (one(21), Q)], [], M2, after={20: {21}}, out=out)
    assert est == {21: 0.0, 20: Q}
    assert out[20] == (Q, [0], Q + H) and out[21][0] == 0.0


def test_reorder_never_moves_a_job_that_waits():
    running = [(one(1), [1], 4 * H)]
    queue = [(excl(10), Q), (one(12), 16 * H), (excl(11), Q)]
    res = reorder(queue, running, M2, slack=H, after={10: {99}, 11: {99}})
    assert 10 not in res.moved and 11 not in res.moved
