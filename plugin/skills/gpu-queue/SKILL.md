---
name: gpu-queue
description: Use whenever you are about to run anything that uses a GPU (training, fine-tuning, evaluation, inference, benchmarks, torchrun/accelerate/deepspeed) on a machine with the gpushift queue. GPUs are shared with other sessions, so GPU work must be submitted with `gpushift submit` instead of launched directly. Also covers checking the queue, reading job logs and results, making training scripts resumable, and what to do when asked to pause, free or stop the GPUs.
---

# GPU queue (gpushift)

The GPUs on this machine are shared by several Claude sessions and the user.
Every GPU job goes through one queue: you **submit** it, and a scheduler starts it
when its GPUs are free. Launching GPU work directly collides with other sessions'
jobs (OOM, slowdowns, corrupt results), so don't do it.

## Rules

1. **Never launch GPU work directly.** No `python train.py`, `torchrun`,
   `accelerate launch`, `deepspeed`, or `CUDA_VISIBLE_DEVICES=... python ...`.
   Use `gpushift submit <preset> -- <command>`. A hook enforces this for common
   patterns; if it blocks you, resubmit through gpushift. Don't look for ways around it.
2. **Look before you submit:** `gpushift status` (compact, includes estimated starts;
   `--json` when you need to parse it, don't truncate it with `head`). Never guess
   whether a GPU is free.
3. **gpushift options go before `--`**, everything after `--` is the job's command:
   `gpushift submit --light --json -- python eval.py`, not `... -- python eval.py --json`.
4. **Tag and estimate.** The tag defaults to the repo name; add `--tag` to tell
   your jobs apart. Give `--eta` when you can estimate the runtime (e.g. from the
   epoch count and a previous run). It makes everyone's start-time estimates better.
5. **`--priority` only when the user asks for it.** It jumps ahead of other
   sessions' work.
6. **Never run `gpushift free` or `gpushift resume`.** They pause or resume every session's
   GPU work and are reserved for the user (`/gpushift:free`, `/gpushift:resume`). If the user asks you
   to pause, free or stop the GPUs, show them `gpushift status` and tell them to run
   `/gpushift:free` (`/gpushift:free --soft` only freezes) and later `/gpushift:resume`. Don't go hunting
   through `nvidia-smi` or other tools.
7. **Don't block on the job, and don't poll.** After submitting, carry on with other
   work (code, docs, analysis). When your job ends, a note saying so (with the
   failure reason, if it failed) appears in your context after your next step; then
   read `gpushift logs <id>`. Use `gpushift wait <id>` only when you have nothing
   else to do until it finishes.
8. **Make new training scripts resumable** with `gpushift.checkpoint` (below), so the
   user can free the GPUs without losing progress. If a job is safe to stop and run
   again from the start (an eval that skips tasks it already finished, a benchmark,
   a short job), submit it with `--restartable`: `/gpushift:free` then stops it and puts
   it back at the front instead of freezing it on the GPU.
9. **Give an `--eta` when you can.** Besides better start estimates, a job with an
   estimate may run on a GPU that would otherwise sit idle while a bigger job waits.
10. **Chain dependent jobs with `--after`** instead of waiting and polling: submit
    the training run, then `gpushift submit --light --after <train id> -- python eval.py`.
    The eval starts once training has finished successfully, and is cancelled if
    training fails (so check why before resubmitting). Submitting jobs one after
    another does not make them run in that order: they may run side by side.
11. **Give `--vram` when you know the memory need** (e.g. `--vram 5G` for a small
    fine-tune measured before). Jobs with a known need can share a GPU instead of
    waiting for a whole one. gpushift also learns it from earlier runs of the same
    command, so don't guess wildly: an underestimate can make a job run out of memory.

## Choosing a preset

| Preset | Use for | Resources |
|---|---|---|
| `--light` | Small evals, inference, notebooks-as-scripts, smoke tests that fit in a fraction of VRAM | Shares one GPU with other light jobs (never with a training job) |
| `--one` | A normal single-GPU training or eval run | 1 GPU (shared with other jobs only if its `--vram` need is known and fits) |
| `--two` | Multi-GPU runs (DDP, `torchrun --nproc_per_node=N`, `accelerate --multi_gpu`) | All GPUs |
| `--exclusive` | Benchmarks/profiling, or jobs that need all CPU, RAM and GPUs | The whole machine; nothing else runs |
| `--cpu` | Data preprocessing, tokenization, CPU-only scripts | No GPU (`CUDA_VISIBLE_DEVICES` is empty) |

When unsure, use `--one`. Use `--light` only when you're confident the job stays
small (< ~2 GB VRAM); a light job that grows can OOM its neighbours.

```bash
gpushift submit --one --eta 40m -- python train.py --config configs/resnet.yaml
gpushift submit --two --eta 3h --tag llama-sft -- torchrun --nproc_per_node 2 sft.py
gpushift submit --light -- python eval.py --ckpt runs/a/last.pt
gpushift submit --cpu -- python scripts/tokenize.py
gpushift submit --one -- "python train.py && python eval.py"   # one shell string = one job
gpushift submit --light --after 12 -- python eval.py --ckpt runs/a/last.pt   # once job 12 succeeded
```

Give the command either as separate words (`-- python run.py a 1`) or as one
quoted string (`-- "python run.py a 1 && python eval.py"`), never half-quoted
(`-- "python run.py a 1" --flag`); gpushift rejects that. A single string runs in a
shell: on Windows, the Git Bash it was submitted from (so `/c/...` paths, `&&`,
`2>&1` and `cp` work as in your Bash tool), or cmd.exe if it uses backslash
paths (`.venv\Scripts\python.exe`) or comes from cmd/PowerShell; `/bin/sh` on
Linux/macOS.
Keep job output in the gpushift log (`gpushift logs <id>`) rather than redirecting it
to a file, so failures show up where you and the user will look.

Output: job id, queue position and estimated start time. `--json` gives
`{"id", "position", "est_start", "mode", "log"}`.

The job runs with the cwd and environment (venv/conda PATH, env vars) of the
`gpushift submit` call. Inside the job, `CUDA_VISIBLE_DEVICES` is set to its GPUs, so
the code should use `cuda:0`..`cuda:N-1` and not hardcode physical GPU ids.

## Checking on jobs

```bash
gpushift status            # human view: each GPU, CPU, queue, estimated starts, mode
gpushift status --json     # stable schema (see below)
gpushift logs <id>         # full output; -n 50 for the tail; -f to follow
gpushift wait <id> --timeout 30m   # blocks; exit code mirrors the job's
gpushift show <id> --json  # one job's details (state, exit_code, log path, note)
gpushift cancel <id>       # your own jobs only, unless the user asks
gpushift retry <id>        # requeue a copy of a failed job
```

`status --json` (schema 1): `mode` (`normal` | `free` | `soft`), `daemon.running`,
`slots {total, used}`, `gpus[] {index, name, mem_used_mb, mem_total_mb, shared, jobs[]}`,
`running[]`, `queue[]` (with `position`, `est_start`) and `recent[]`. Each job has
`id, state, preset, gpus, tag, project, session, command, cwd, runtime_s,
exit_code, eta_s, expected_duration_s, resumable, evictions, log, note`.
Your own jobs have `session` equal to your `CLAUDE_CODE_SESSION_ID`.

Job states: `queued`, `running`, `frozen` (paused by the user's free mode; it
continues later), `evicting` (checkpointing so the user can use the GPUs),
`done`, `failed`, `cancelled`.

If `mode` isn't `normal`, the user has freed the machine, by hand or automatically
because they're using the GPU themselves (`auto_free` then names the program).
Your job will wait and continue afterwards; don't try to work around it. If
`daemon.running` is false, tell the user (they can run `gpushift start`).

Jobs run first come, first served, but a short job may run before a long one
when that cuts everyone's waiting (within limits). The queue order in `status` is
the order jobs are expected to start in, and it can change as jobs come and go. A
good `--eta` helps: jobs without an estimate are never moved ahead.

## Making a training script resumable

When the user runs `gpushift free`, resumable jobs save a checkpoint and exit, then
restart from it later. Other jobs are only frozen and keep holding their VRAM. Any
job can `import gpushift.checkpoint`; gpushift puts it on the job's PYTHONPATH.

```python
from gpushift import checkpoint as ckpt

last = ckpt.latest_checkpoint("checkpoints", "step*.pt")  # None on a fresh start
step = load(last) if last else 0
while step < total:
    train_step()
    step += 1
    if step % 1000 == 0 or ckpt.should_stop():
        with ckpt.atomic_path(f"checkpoints/step{step:08d}.pt") as tmp:
            save(tmp)  # never leaves a half-written file
    if ckpt.should_stop():
        break  # exit code 0 -> gpushift requeues and resumes
```

- `should_stop()` is cheap; call it every step. The first call also marks the job
  as resumable.
- `ckpt.resuming()` is True when gpushift restarted the job after an eviction.
- HF Trainer: add a callback that sets `control.should_save = True` and
  `control.should_training_stop = True` when `should_stop()` is True, and pass
  `resume_from_checkpoint=get_last_checkpoint(output_dir)`.
- Lightning: in a callback's `on_train_batch_end`, call `trainer.save_checkpoint(...)`
  and set `trainer.should_stop = True`; pass `ckpt_path=` to `fit`.
- Multi-process (DDP/accelerate): reduce the flag across ranks (max) before acting,
  so all ranks stop at the same step.

Full examples: `examples/` in the gpushift repo (plain PyTorch, HF Trainer/accelerate,
Lightning).

## Not GPU work?

The hook only targets GPU launches. If it blocks something that doesn't use a GPU
(e.g. a script named `train_test_split.py`), append `# no-queue` to the command and
run it again. Quick checks such as `python -c "import torch; ..."`, `--help`, and
pytest are never blocked.
