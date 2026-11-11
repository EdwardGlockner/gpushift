---
description: Show the shared GPU queue (what runs on each GPU, what is waiting, estimated starts)
allowed-tools: Bash(gpushift status:*)
---

!`gpushift status`

Show the user the status above as-is, in a code block. Then add at most two short
lines: anything that needs attention (daemon not running, free mode on, a failed
job from this session, a job that has waited unusually long). Don't run other
commands.
