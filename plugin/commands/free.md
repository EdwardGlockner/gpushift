---
description: Free the machine. Pause the GPU queue, checkpoint resumable jobs, stop restartable ones, freeze the rest (add --soft to only freeze)
argument-hint: "[--soft]"
disable-model-invocation: true
allowed-tools: Bash(GPUSHIFT_BY_USER=1 gpushift free:*), Bash(gpushift status:*)
---

The user ran /gpushift:free, which explicitly authorizes free mode.

!`GPUSHIFT_BY_USER=1 gpushift free --no-wait $ARGUMENTS`

Show the user the output above. If any job is listed as `evicting` or
`restarting`, run `gpushift status --json` every ~15 seconds (at most 5 minutes) until
no job is in the `evicting` state, then report per job: evicted (back in the queue,
will resume from its checkpoint), restarting (back in the queue, will run again) or
frozen (VRAM still held). If frozen jobs hold memory on a GPU, say which GPU and
how much, and that `/gpushift:cancel <id>` releases it (`gpushift retry <id> --priority`
runs it again later; it starts over unless the job skips finished work). End with
the VRAM still in use per GPU (from `gpus[].mem_used_mb`) and a reminder that
`/gpushift:resume` continues everything. Keep it short.
