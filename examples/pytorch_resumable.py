"""Plain PyTorch training loop that gpushift can evict and resume.

    gpushift submit --one --eta 30m -- python examples/pytorch_resumable.py

The pattern:
  1. On start, load the newest checkpoint if there is one.
  2. Call ckpt.should_stop() every step (cheap). The first call also tells gpushift
     the job is resumable, so `gpushift free` evicts it instead of freezing it.
  3. When it returns True: save, then exit with code 0. gpushift requeues the job at
     the front of the queue and restarts it later with GPUSHIFT_RESUME=1.
"""

from pathlib import Path

import torch
import torch.nn as nn

from gpushift import checkpoint as ckpt

CKPT_DIR = Path("checkpoints/pytorch_example")
TOTAL_STEPS = 20_000
SAVE_EVERY = 1_000
KEEP = 3


def save(step, model, opt):
    with ckpt.atomic_path(CKPT_DIR / f"step{step:08d}.pt") as tmp:
        torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict()}, tmp)
    for old in sorted(CKPT_DIR.glob("step*.pt"))[:-KEEP]:
        old.unlink()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 10)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    step = 0
    last = ckpt.latest_checkpoint(CKPT_DIR, "step*.pt")
    if last is not None:
        state = torch.load(last, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        print(f"resumed from {last} at step {step}", flush=True)

    while step < TOTAL_STEPS:
        x = torch.randn(256, 512, device=device)
        y = torch.randint(0, 10, (256,), device=device)
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1

        if step % SAVE_EVERY == 0:
            save(step, model, opt)
        if ckpt.should_stop():
            save(step, model, opt)
            print(f"gpushift asked us to stop; checkpointed at step {step}", flush=True)
            return  # exit 0 -> gpushift requeues and resumes later

    print(f"done: {step} steps, final loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
