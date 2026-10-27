# dl_models/trainer.py

from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Standard label-smoothed cross-entropy for multi-class (including binary as 2 classes).
    """

    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, logits, target):
        """
        logits: (N, C)
        target: (N,) with class indices in [0, C-1]
        """
        n_classes = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class Trainer:
    """
    Handles training and validation with early stopping and AMP.

    Saves best model by validation loss. Returns history dict for plotting.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        save_path,
        patience: int = 10,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
        max_grad_norm: float = 5.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path
        self.patience = patience
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.max_grad_norm = max_grad_norm

        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

    def _step_batch(self, batch, train: bool):
        x, y, _subjects = batch  # we don't use subjects in loss/metrics
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        if train:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.model.eval()

        with torch.set_grad_enabled(train):
            if self.use_amp and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
            else:
                logits = self.model(x)
                loss = self.criterion(logits, y)

            preds = torch.argmax(logits, dim=-1)
            correct = (preds == y).sum().item()
            n_samples = y.size(0)

            if train:
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return loss.item(), correct, n_samples

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> Dict[str, List[float]]:
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        for epoch in range(1, epochs + 1):
            # ----------------- Train -----------------
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                loss, correct, n_samples = self._step_batch(batch, train=True)
                train_loss_sum += loss * n_samples
                train_correct += correct
                train_total += n_samples

            train_loss = train_loss_sum / max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            # ----------------- Validation -----------------
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    loss, correct, n_samples = self._step_batch(batch, train=False)
                    val_loss_sum += loss * n_samples
                    val_correct += correct
                    val_total += n_samples

            val_loss = val_loss_sum / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Epoch {epoch:03d} | "
                f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | "
                f"Val loss: {val_loss:.4f}, acc: {val_acc:.3f}"
            )

            # Early stopping on val loss
            if val_loss < self.best_val_loss - 1e-4:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"  [Checkpoint] Saved best model -> {self.save_path}")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"  [Early stopping] No improvement for {self.patience} epochs.")
                    break

        return history