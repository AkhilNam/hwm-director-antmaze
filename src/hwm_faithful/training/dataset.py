"""Torch Dataset / DataLoader for Level-1 15-step windows.

This is not a copy of original ``D4RLDataset``. It wraps
``DiverseMazeOfflineDataset`` and collates to ``[B, T, ...]`` so the
trainer can ``transpose(0, 1)`` like ``pldm/train.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hwm_faithful.data.diverse_maze import (
    DEFAULT_TRAIN_DIR,
    DiverseMazeOfflineDataset,
    DiverseMazePaths,
    open_images_memmap,
)
from hwm_faithful.training.config import EXPECTED_25MAP_FRAMES, Level1TrainConfig


class Level1WindowDataset(Dataset):
    """Map-style dataset of L1 windows. Images are mmap'd, not materialized."""

    def __init__(self, offline: DiverseMazeOfflineDataset) -> None:
        self.offline = offline

    def __len__(self) -> int:
        return len(self.offline)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        window = self.offline.get_level1_window(int(idx))
        return {
            "images": window.images,
            "proprio": window.proprio,
            "actions": window.actions,
            "locations": window.locations,
        }


def collate_level1(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack to ``[B, T, ...]`` (original DataLoader layout before transpose)."""
    return {
        key: torch.stack([item[key] for item in batch], dim=0) for key in batch[0]
    }


def _worker_init_fn(_worker_id: int) -> None:
    torch.set_num_threads(1)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def describe_images(images_path: str | Path) -> dict[str, Any]:
    path = Path(images_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    arr = open_images_memmap(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "shape": tuple(int(x) for x in arr.shape),
        "dtype": str(arr.dtype),
        "mmap": True,
        "n_frames": int(arr.shape[0]),
        "matches_25map_expectation": int(arr.shape[0]) == EXPECTED_25MAP_FRAMES,
        "expected_frames": EXPECTED_25MAP_FRAMES,
    }


def locate_25maps_images(
    explicit: str | Path | None = None,
) -> Path | None:
    """Search known locations. Does not render or download a new corpus."""
    import os

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("HWM_25MAPS_IMAGES")
    if env:
        candidates.append(Path(env))
    train = DiverseMazePaths.train_25maps()
    if train.images_path is not None:
        candidates.append(train.images_path)
    candidates.extend(
        [
            DEFAULT_TRAIN_DIR / "images.npy",
            Path("/scratch/wz1232/data/maze2d_large_diverse_25maps/images.npy"),
            Path(
                "/storage/scratch1/2/anampally3/hwm-original-repro/"
                "pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps/images.npy"
            ),
        ]
    )
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return p
    return None


def build_offline_dataset(cfg: Level1TrainConfig) -> DiverseMazeOfflineDataset:
    if not cfg.data_path or not cfg.images_path:
        raise ValueError("data_path and images_path are required")
    data_path = Path(cfg.data_path)
    images_path = Path(cfg.images_path)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not images_path.is_file():
        raise FileNotFoundError(images_path)
    if cfg.require_25maps:
        info = describe_images(images_path)
        if not info["matches_25map_expectation"]:
            raise RuntimeError(
                "require_25maps: images.npy N="
                f"{info['n_frames']} != expected {EXPECTED_25MAP_FRAMES}. "
                "Refusing to train. Do not regenerate the corpus."
            )
    return DiverseMazeOfflineDataset(
        data_path,
        images_path,
        n_steps=cfg.n_steps,
        normalize=cfg.normalize,
        l1_only=cfg.l1_only,
        max_windows=cfg.max_windows,
    )


def make_level1_loader(
    cfg: Level1TrainConfig,
    offline: DiverseMazeOfflineDataset | None = None,
) -> DataLoader:
    ds = Level1WindowDataset(offline or build_offline_dataset(cfg))
    n_workers = 0 if cfg.num_workers < 1 else int(cfg.num_workers)
    kwargs: dict[str, Any] = {
        "dataset": ds,
        "batch_size": cfg.batch_size,
        "shuffle": cfg.shuffle,
        "num_workers": n_workers,
        "drop_last": cfg.drop_last and len(ds) >= cfg.batch_size,
        "pin_memory": bool(cfg.pin_memory),
        "collate_fn": collate_level1,
    }
    if n_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = cfg.prefetch_factor
        kwargs["worker_init_fn"] = _worker_init_fn
    return DataLoader(**kwargs)
