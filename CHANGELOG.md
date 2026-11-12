# Changelog

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
