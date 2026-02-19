import sys

import torch
from torch.utils.data import DataLoader, random_split

from ..model.anomalybert import AnomalyBertModel
from ..model.config import ModelConfig
from ..data.dataset import TimeseriesDataset
from .loss import AnomalyLoss
from .checkpoint import save_checkpoint


class Trainer:
    def __init__(
        self,
        model: AnomalyBertModel,
        train_dataset: TimeseriesDataset,
        normalizer=None,
        lr: float = 1e-3,
        batch_size: int = 32,
        val_split: float = 0.2,
    ):
        self.model = model
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.loss_fn = AnomalyLoss()

        # Split into train/val
        total = len(train_dataset)
        val_size = int(total * val_split)
        train_size = total - val_size
        if val_size > 0 and train_size > 0:
            self.train_ds, self.val_ds = random_split(train_dataset, [train_size, val_size])
        else:
            self.train_ds = train_dataset
            self.val_ds = None

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )

    def train(self, epochs: int = 50, checkpoint_path: str = "models/model.pt") -> list[float]:
        losses = []
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
            for batch in loader:
                values = batch["values"]
                labels = batch["labels"]

                self.optimizer.zero_grad()
                scores = self.model(values)
                loss = self.loss_fn(scores, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            # Validation
            val_loss_str = ""
            if self.val_ds is not None and len(self.val_ds) > 0:
                val_loss = self._validate()
                val_loss_str = f" | val_loss: {val_loss:.4f}"

            print(f"Epoch {epoch + 1}/{epochs} | train_loss: {avg_loss:.4f}{val_loss_str}")
            sys.stdout.flush()

        save_checkpoint(self.model, self.normalizer, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        return losses

    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                values = batch["values"]
                labels = batch["labels"]
                scores = self.model(values)
                loss = self.loss_fn(scores, labels)
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)
