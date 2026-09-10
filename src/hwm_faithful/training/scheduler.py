"""Learning-rate schedule matching original ``pldm/optimizers/schedulers.py``.

YAML omits ``optimizer_schedule`` → ``TrainConfig`` default ``Cosine``.

Each step (including 0):

    base_lr_eff = base_lr * batch_size / 256
    warmup_steps = int(0.10 * epochs * batch_steps)

    if step < warmup_steps:
        lr = base_lr_eff * step / warmup_steps   # step 0 → lr 0
    else:
        cosine from base_lr_eff down to base_lr_eff * 0.001
"""

from __future__ import annotations

import math
from typing import Any

import torch

from hwm_faithful.training.config import (
    COSINE_END_LR_RATIO,
    LR_BATCH_REF,
    WARMUP_FRACTION,
    Level1TrainConfig,
)


class CosineWarmupScheduler:
    """Original ``Scheduler.adjust_learning_rate`` for Cosine / Constant."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        schedule: str,
        base_lr: float,
        batch_size: int,
        epochs: int,
        batch_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.schedule = schedule.lower()
        self.base_lr = float(base_lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.batch_steps = int(batch_steps)
        self.last_lr: float = 0.0

    @classmethod
    def from_config(
        cls,
        optimizer: torch.optim.Optimizer,
        cfg: Level1TrainConfig,
        batch_steps: int,
    ) -> "CosineWarmupScheduler":
        # Scheduler ``epochs`` uses the YAML field, not n_epoch_passes.
        return cls(
            optimizer,
            schedule=cfg.optimizer_schedule,
            base_lr=cfg.base_lr,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            batch_steps=batch_steps,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schedule": self.schedule,
            "base_lr": self.base_lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "batch_steps": self.batch_steps,
            "last_lr": self.last_lr,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.schedule = str(state["schedule"])
        self.base_lr = float(state["base_lr"])
        self.batch_size = int(state["batch_size"])
        self.epochs = int(state["epochs"])
        self.batch_steps = int(state["batch_steps"])
        self.last_lr = float(state.get("last_lr", 0.0))

    def lr_at(self, step: int) -> float:
        if self.schedule == "constant":
            return self.base_lr * self.batch_size / LR_BATCH_REF
        max_steps = self.epochs * self.batch_steps
        warmup_steps = int(WARMUP_FRACTION * max_steps)
        base_lr = self.base_lr * self.batch_size / LR_BATCH_REF
        if warmup_steps <= 0:
            warmup_steps = 1
        if step < warmup_steps:
            return base_lr * step / warmup_steps
        step_c = step - warmup_steps
        max_c = max_steps - warmup_steps
        if max_c <= 0:
            return base_lr * COSINE_END_LR_RATIO
        q = 0.5 * (1.0 + math.cos(math.pi * step_c / max_c))
        end_lr = base_lr * COSINE_END_LR_RATIO
        return base_lr * q + end_lr * (1.0 - q)

    def adjust_learning_rate(self, step: int) -> float:
        lr = self.lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_lr = lr
        return lr
