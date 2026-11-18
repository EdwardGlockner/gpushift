---
description: Move a queued GPU job to the front of the queue
argument-hint: <job id>
disable-model-invocation: true
allowed-tools: Bash(GPUSHIFT_BY_USER=1 gpushift priority:*), Bash(gpushift status:*)
---

!`GPUSHIFT_BY_USER=1 gpushift priority $ARGUMENTS && gpushift status --short`

Report the result above in one or two lines: which job is now first, and its
estimated start if shown. If it failed (e.g. the job is already running), say why.
