"""Freeze Level-1 weights for Level-2 training.

Original ``Trainer`` (``pldm/train.py``) when ``freeze_l1=true``:

    for m in self.model.level1.modules():
        for p in m.parameters():
            p.requires_grad = False

It does **not** call ``eval()``. Gradients do not reach L1 during L2 training.
The posterior and L2 action encoder remain trainable.
"""

from __future__ import annotations

from torch import nn


def freeze_module(module: nn.Module) -> None:
    """Match original freeze_l1: ``requires_grad=False`` on all parameters."""
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True
