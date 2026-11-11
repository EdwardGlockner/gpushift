---
description: Leave free mode. Unfreeze paused GPU jobs and restart the queue
disable-model-invocation: true
allowed-tools: Bash(GPUSHIFT_BY_USER=1 gpushift resume:*)
---

The user ran /gpushift:resume, which explicitly authorizes resuming the queue.

!`GPUSHIFT_BY_USER=1 gpushift resume`

Report the result above in a few short lines (which jobs resumed, which will
restart from a checkpoint).
