"""Hugging Face Trainer that gpushift can evict and resume.

    gpushift submit --one --eta 2h -- python examples/hf_trainer_resumable.py

Two pieces:
  * GpushiftCallback: when gpushift asks the job to stop, save a checkpoint and end
    training cleanly.
  * resume_from_checkpoint=<latest checkpoint in output_dir> on (re)start.
"""

import os

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from gpushift import checkpoint as ckpt


class GpushiftCallback(TrainerCallback):
    """Save and stop when gpushift wants the GPUs back."""

    def on_step_end(self, args, state, control, **kwargs):
        if ckpt.should_stop():
            control.should_save = True
            control.should_training_stop = True
        return control


def main():
    out = "checkpoints/hf_example"
    name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)

    from datasets import load_dataset

    ds = load_dataset("imdb", split="train[:2000]").map(
        lambda b: tok(b["text"], truncation=True, padding="max_length", max_length=128), batched=True
    )

    args = TrainingArguments(
        output_dir=out,
        per_device_train_batch_size=16,
        num_train_epochs=1,
        save_steps=200,
        save_total_limit=3,
        logging_steps=50,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, callbacks=[GpushiftCallback()])

    last = get_last_checkpoint(out) if os.path.isdir(out) else None
    if last:
        print(f"resuming from {last}", flush=True)
    trainer.train(resume_from_checkpoint=last)
    if ckpt.should_stop():
        print("stopped for gpushift; will resume from the checkpoint", flush=True)


if __name__ == "__main__":
    main()


# --- accelerate / multi-GPU note -------------------------------------------
# With several processes (accelerate launch, torchrun), every rank must agree
# to stop at the same step or a collective op will hang. Reduce the flag:
#
#     stop = torch.tensor(int(ckpt.should_stop()), device=accelerator.device)
#     if accelerator.reduce(stop, reduction="max").item():
#         accelerator.save_state("checkpoints/accel/latest")
#         break
#
# and on start: if ckpt.latest_checkpoint("checkpoints/accel"): accelerator.load_state(...).
# The stop file reaches every rank, so this works on Windows and Linux alike.
