# Changelog

## 0.1.0 (unreleased)

First release.

- `gpushift` CLI and scheduler daemon: presets (`--light`, `--one`, `--two`,
  `--exclusive`, `--cpu`), slots, start-time estimates learned from earlier runs,
  and `status --json` with a stable schema.
- Scheduling: sessions take turns (fair share by recent GPU time); a waiting
  multi-GPU job can't starve, while jobs that finish in time may use idle GPUs;
  jobs with a known memory need (`--vram`, or learned from earlier runs) share GPUs.
- Free mode: `gpushift free` checkpoints resumable jobs, restarts `--restartable`
  ones and freezes the rest, and says which GPU frozen jobs still hold;
  `gpushift resume` continues everything. Survives reboots and daemon restarts.
- Automatic free mode while the user runs heavy GPU programs (games, Blender, ...),
  detected from per-process GPU memory, with automatic resume afterwards.
- Notifications: the submitting Claude session is told when its jobs end, and
  desktop notifications report failures.
- `gpushift ui`: a local web dashboard with GPUs, jobs, the queue and controls.
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
