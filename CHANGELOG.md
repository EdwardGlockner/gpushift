# Changelog

## 0.1.16 (2026-11-23)

- Jobs survive `/clear`. The new conversation takes over the jobs that the one
  before it submitted, is told what they are and how far they got, and hears
  when they end. A `gpushift wait` started before the clear keeps waiting for
  them. Only the same Claude Code window takes them over; other windows and
  resumed conversations do not.

## 0.1.15 (2026-11-22)

- A running job past its estimate (with no longer earlier runs to go by) is now
  expected to need a fifth more than it has run so far, at least 5 minutes,
  instead of as long again. Jobs queued behind it get more realistic start times.

## 0.1.14 (2026-11-21)

- Dashboard: a wide queue table scrolls inside its own box instead of making
  the whole page scroll sideways.
- Jobs that can't start for lack of the same free RAM share one line on the
  dashboard and in `gpushift status` instead of one repeated line per job.

## 0.1.13 (2026-11-21)

- `gpushift submit --gpu 0` (or `--gpu 1070`, part of the name) runs a job only
  on that GPU, for timings that must be comparable. While it waits for its GPU,
  jobs that never get in its way may use the other GPUs.
- A running job far past its estimate whose log has been silent for half an
  hour is flagged as looking stuck, with whether it still uses the CPU: in
  `gpushift status`, on the dashboard, as a desktop notification, and once to
  the agent that submitted it (`gpushift wait` returns to tell it). Nothing is
  cancelled automatically.
- Dashboard: the part of a job's timeline bar past its estimate is drawn as a
  guess, and "ahead of" badges no longer list jobs whose start is unknown.
- Starting the daemon waits up to 30 s on a slow machine instead of giving up
  early, and reports at once when it exits instead.

## 0.1.12 (2026-11-19)

- A job that has run for half an hour is booked near the RAM it has used so
  far (its peak plus a margin) instead of its `--ram` or the 2 GB default, so
  generous estimates stop holding RAM that queued jobs could use.
- `gpushift status` and the dashboard say why a job can't start, e.g. "needs
  12.0 GB RAM, but only 2.7 GB is free for jobs", instead of blaming a missing
  time estimate.

## 0.1.11 (2026-11-19)

- When a job succeeds in under 10% of its `--eta`, the agent that submitted it
  is told to check its log before trusting the results: a script that skipped
  its work or stopped early also exits with 0.

## 0.1.10 (2026-11-19)

- Better runtime estimates. `--eta` is scaled by how long earlier jobs really
  ran compared with theirs (per project once it has 5 finished jobs, else over
  all projects), so start times no longer drift when estimates are padded.
  Runs that ended almost at once (under 10% of their `--eta`) don't count.
- Estimates come with a span: the middle half of what earlier runs took.
  `gpushift status` shows start times like `~44m (11:12-11:27)`, the dashboard
  shows spans for runtimes, starts and when running jobs finish, and the JSON
  has `expected_duration_range_s` and `est_start_range`. A span wider than 3x
  is left out.
- On machines with different GPUs, gpushift learns how much slower each GPU
  runs the same command and estimates a job's runtime for the GPU it runs or
  would run on. Start times and short jobs filling idle GPUs use it.

## 0.1.9 (2026-11-18)

- Agents no longer change the queue order. `--priority`, `gpushift priority`
  and `gpushift swap` are reserved for the user: the hook blocks them for
  agents, the skill tells agents not to ask for a job to be moved up, and
  `/gpushift:priority` can only be run by you.

## 0.1.8 (2026-11-18)

- Jobs only start when the RAM they need fits: gpushift learns each command's
  peak RAM from earlier runs (or takes `--ram 12G`), books it while the job
  runs, and keeps `ram_reserve_perc` (10%) of RAM free. Jobs it has not seen
  before count as 2 GB. `gpushift status` shows RAM booked. `ram_aware = false`
  turns this off.
- The default is now 3 slots per GPU (was 2), so CPU jobs no longer leave a GPU
  idle on a 2-GPU machine; RAM, not slots, now keeps the machine from
  overfilling. A `slots` value in your config still applies.

## 0.1.7 (2026-11-18)

- A `--light` job no longer waits while a `--one` job holds the shared GPU; it
  runs on an idle GPU instead.
- On Windows, freezing a job (`gpushift free`) also moves its memory out of RAM,
  so the frozen job no longer holds RAM you need. It comes back when the job
  resumes.

## 0.1.6 (2026-11-16)

- The dashboard queue shows when each job is expected to start, and marks jobs
  that start ahead of earlier ones, so it agrees with the timeline.

## 0.1.5 (2026-11-15)

- A job that runs longer than estimated no longer stops other jobs from using
  idle GPUs. Its time left is estimated from earlier runs that lasted at least
  as long, or else taken to be as long again as it has run so far.

## 0.1.4 (2026-11-15)

- AMD GPUs on Linux and Windows (not macOS). Integrated graphics with less than
  2 GB of dedicated memory are skipped.
- The hook also catches `HIP_VISIBLE_DEVICES=...` and `ROCR_VISIBLE_DEVICES=...`.

## 0.1.3 (2026-11-13)

- A job that needs more GPU memory than is free with no jobs running (because
  other programs use it) no longer blocks the jobs behind it.

## 0.1.2 (2026-11-12)

- Removed automatic free mode (freeing the GPUs while you run a game or other
  heavy GPU program). Free and resume by hand with `gpushift free` / `resume`.
- The daemon uses about a quarter of the CPU it did on Windows.

## 0.1.1 (2026-11-12)

- Agents are now woken when their jobs end, even if they were idle: `gpushift wait`
  without a job id waits for the next of the session's own jobs, and the skill tells
  agents to run it in the background after submitting.
- The dashboard is shown in the README, which is now much shorter.

## 0.1.0 (2026-11-12)

First release.

- `gpushift` CLI and scheduler daemon: presets (`--light`, `--one`, `--two`,
  `--exclusive`, `--cpu`), slots, start-time estimates learned from earlier runs,
  and `status --json` with a stable schema.
- Scheduling: first come, first served (or sessions take turns, `fair_share = true`); a waiting
  multi-GPU job can't starve, while jobs that finish in time may use idle GPUs;
  jobs with a known memory need (`--vram`, or learned from earlier runs) share GPUs;
  short jobs may run before long ones when that cuts the total waiting, without
  pushing any job back more than an hour or half its run time.
- Free mode: `gpushift free` checkpoints resumable jobs, restarts `--restartable`
  ones and freezes the rest, and says which GPU frozen jobs still hold;
  `gpushift resume` continues everything. Survives reboots and daemon restarts.
- Automatic free mode while the user runs heavy GPU programs (games, Blender, ...),
  detected from per-process GPU memory, with automatic resume afterwards.
- Notifications: the submitting Claude session is told when its jobs end, and
  desktop notifications report failures.
- `--after <job>`: a job starts only once the jobs it depends on finished
  successfully (train, then eval); it is cancelled if one of them fails.
- `gpushift ui`: a local web dashboard with the GPUs, CPU and drives, a timeline
  of what runs where and when, running jobs with their logs, the queue (drag to
  reorder) and controls. Time a job spends frozen doesn't count as run time.
- Jobs run with the submitter's cwd and environment. A command given as one
  string runs in a shell: on Windows, the Git Bash it was submitted from (what
  Claude Code's Bash tool uses) unless it uses backslash paths, otherwise
  cmd.exe; `/bin/sh` elsewhere. Missing programs or scripts and half-quoted
  commands are rejected at submit; a failed job's note includes its last output line.
- `gpushift.checkpoint`: a dependency-free helper that makes training scripts
  resumable, with examples for plain PyTorch, HF Trainer and Lightning.
- Autostart at login on Windows (HKCU Run key), Linux (systemd user service)
  and macOS (LaunchAgent).
- Claude Code plugin: the `gpu-queue` skill; `/gpushift:queue`, `/gpushift:priority`,
  `/gpushift:cancel`, `/gpushift:free` and `/gpushift:resume`; a hook that redirects
  direct GPU launches to the queue, and one that reports ended jobs.
