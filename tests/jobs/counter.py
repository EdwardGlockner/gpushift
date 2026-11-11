"""Test job: counts to N, one step per 0.1 s, checkpointing to a JSON file.
Resumable via gpushift.checkpoint (imported from the PYTHONPATH gpushift provides)."""

import json
import sys
import time
from pathlib import Path

from gpushift import checkpoint as ckpt

out = Path(sys.argv[1])
total = int(sys.argv[2])
state_file = out / "state.json"
out.mkdir(parents=True, exist_ok=True)

step = 0
if state_file.exists():
    step = json.loads(state_file.read_text())["step"]
    print(f"resumed at step {step} (GPUSHIFT_RESUME={ckpt.resuming()})", flush=True)

while step < total:
    time.sleep(0.1)
    step += 1
    if ckpt.should_stop(poll_interval=0.1):
        with ckpt.atomic_path(state_file) as tmp:
            tmp.write_text(json.dumps({"step": step}))
        print(f"stop requested; checkpointed at step {step}", flush=True)
        sys.exit(0)

(out / "done.txt").write_text(str(step))
print(f"finished at step {step}", flush=True)
