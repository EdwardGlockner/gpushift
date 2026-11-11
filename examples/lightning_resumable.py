"""PyTorch Lightning model that gpushift can evict and resume.

gpushift submit --one --eta 1h -- python examples/lightning_resumable.py
"""

import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gpushift import checkpoint as ckpt

CKPT_DIR = "checkpoints/lightning_example"


class GpushiftStop(L.Callback):
    """Save and stop when gpushift wants the GPUs back."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if ckpt.should_stop():
            trainer.save_checkpoint(f"{CKPT_DIR}/gpushift-stop.ckpt")
            trainer.should_stop = True


class Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 2))

    def training_step(self, batch, _):
        x, y = batch
        return nn.functional.cross_entropy(self.net(x), y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main():
    data = TensorDataset(torch.randn(50_000, 64), torch.randint(0, 2, (50_000,)))
    trainer = L.Trainer(
        max_epochs=20,
        callbacks=[GpushiftStop(), L.pytorch.callbacks.ModelCheckpoint(dirpath=CKPT_DIR, save_last=True)],
    )
    last = ckpt.latest_checkpoint(CKPT_DIR, "*.ckpt")
    if last:
        print(f"resuming from {last}", flush=True)
    trainer.fit(Model(), DataLoader(data, batch_size=256), ckpt_path=str(last) if last else None)


if __name__ == "__main__":
    main()
