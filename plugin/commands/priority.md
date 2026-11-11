---
description: Move a queued GPU job to the front of the queue
argument-hint: <job id>
allowed-tools: Bash(gpushift priority:*), Bash(gpushift status:*)
---

!`gpushift priority $ARGUMENTS && gpushift status --short`

Report the result above in one or two lines: which job is now first, and its
estimated start if shown. If it failed (e.g. the job is already running), say why.
