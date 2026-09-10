"""Faithful L1 checkpoint format. Independent of original ``.ckpt`` layout."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

from hwm_faithful.training.config import Level1TrainConfig

CHECKPOINT_FORMAT = "hwm_faithful_l1"
CKPT_PATTERN = re.compile(r"epoch=(\d+)_sample_step=(\d+)\.pt$")


def save_checkpoint(
    path: str | Path,
    *,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    idm: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_state: dict[str, Any],
    epoch: int,
    global_step: int,
    sample_step: int,
    config: Level1TrainConfig | dict[str, Any],
    seed: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = config.to_dict() if isinstance(config, Level1TrainConfig) else dict(config)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "idm": idm.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_state,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "sample_step": int(sample_step),
        "config": cfg,
        "seed": int(seed),
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        blob = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=map_location)
    if not isinstance(blob, dict) or blob.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    return blob


def apply_checkpoint(
    blob: dict[str, Any],
    *,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    idm: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    encoder.load_state_dict(blob["encoder"], strict=strict)
    predictor.load_state_dict(blob["predictor"], strict=strict)
    idm.load_state_dict(blob["idm"], strict=strict)
    if optimizer is not None and blob.get("optimizer") is not None:
        optimizer.load_state_dict(blob["optimizer"])
    if scheduler is not None and blob.get("scheduler") is not None:
        scheduler.load_state_dict(blob["scheduler"])
    return blob


def checkpoint_filename(epoch: int, sample_step: int) -> str:
    return f"epoch={epoch}_sample_step={sample_step}.pt"


def pick_latest_checkpoint(output_dir: str | Path) -> Path | None:
    d = Path(output_dir)
    latest = d / "latest.pt"
    if latest.is_file():
        return latest
    found: list[tuple[int, int, Path]] = []
    if not d.is_dir():
        return None
    for p in d.glob("epoch=*_sample_step=*.pt"):
        m = CKPT_PATTERN.search(p.name)
        if m:
            found.append((int(m.group(1)), int(m.group(2)), p))
    if not found:
        return None
    found.sort()
    return found[-1][2]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)
