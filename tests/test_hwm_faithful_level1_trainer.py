"""Unit tests for faithful Level-1 trainer (M9).

Uses a tiny synthetic pickle+npy. Does not open the multi-GB corpus.
Does not train Level 2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hwm_faithful.data.diverse_maze import DiverseMazeOfflineDataset
from hwm_faithful.shapes import IMAGE_SIZE
from hwm_faithful.training.checkpoint import load_checkpoint, pick_latest_checkpoint
from hwm_faithful.training.config import Level1TrainConfig, YAML_BASE_LR
from hwm_faithful.training.dataset import Level1WindowDataset, make_level1_loader
from hwm_faithful.training.level1_trainer import Level1Trainer
from hwm_faithful.training.scheduler import CosineWarmupScheduler


def _write_tiny(tmp: Path, n_states: int = 40, n_episodes: int = 2) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    splits = []
    frames = []
    for ep in range(n_episodes):
        obs = rng.normal(size=(n_states, 4)).astype(np.float64)
        act = rng.uniform(-1.0, 1.0, size=(n_states - 1, 2)).astype(np.float32)
        splits.append({"actions": act, "observations": obs, "map_idx": ep})
        frames.append(
            rng.integers(20, 220, size=(n_states, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        )
    pickle_path = tmp / "data.p"
    images_path = tmp / "images.npy"
    torch.save(splits, pickle_path)
    np.save(images_path, np.concatenate(frames, axis=0))
    return pickle_path, images_path


def test_l1_only_increases_window_count(tmp_path: Path) -> None:
    pickle_path, images_path = _write_tiny(tmp_path, n_states=70, n_episodes=2)
    both = DiverseMazeOfflineDataset(pickle_path, images_path, l1_only=False)
    l1 = DiverseMazeOfflineDataset(pickle_path, images_path, l1_only=True)
    # 70-61=9 vs 70-15=55 windows per episode.
    assert len(both) == 9 * 2
    assert len(l1) == 55 * 2


def test_cosine_step0_is_zero_then_warmup() -> None:
    m = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(m.parameters(), lr=1.0)
    sched = CosineWarmupScheduler(
        opt,
        schedule="cosine",
        base_lr=YAML_BASE_LR,
        batch_size=128,
        epochs=3,
        batch_steps=100,
    )
    assert sched.lr_at(0) == 0.0
    warmup = int(0.10 * 3 * 100)
    assert warmup == 30
    scaled = YAML_BASE_LR * 128 / 256
    assert sched.lr_at(warmup) == pytest.approx(scaled)
    assert 0.0 < sched.lr_at(1) < scaled


def test_constant_schedule_uses_linear_scaling() -> None:
    m = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(m.parameters(), lr=1.0)
    sched = CosineWarmupScheduler(
        opt,
        schedule="constant",
        base_lr=YAML_BASE_LR,
        batch_size=4,
        epochs=1,
        batch_steps=10,
    )
    assert sched.lr_at(0) == pytest.approx(YAML_BASE_LR * 4 / 256)


def test_trainer_two_steps_and_resume(tmp_path: Path) -> None:
    pickle_path, images_path = _write_tiny(tmp_path)
    out = tmp_path / "ckpts"
    cfg = Level1TrainConfig(
        data_path=str(pickle_path),
        images_path=str(images_path),
        output_dir=str(out),
        batch_size=2,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        l1_only=True,
        max_windows=4,
        max_steps=2,
        epochs=1,
        epoch_end_inclusive=False,
        optimizer_schedule="constant",
        log_every=1,
        device="cpu",
        pin_memory=False,
        resume_if_possible=False,
        seed=0,
        deterministic=False,
    )
    trainer = Level1Trainer(cfg)
    hist = trainer.train()
    assert len(hist) == 2
    assert all(np.isfinite(r["loss"]) for r in hist)
    assert all(np.isfinite(r["grad_norm"]) for r in hist)
    latest = pick_latest_checkpoint(out)
    assert latest is not None
    blob = load_checkpoint(latest)
    assert blob["format"] == "hwm_faithful_l1"
    assert "encoder" in blob and "predictor" in blob and "idm" in blob
    assert blob["sample_step"] == 4  # 2 steps * batch 2

    cfg_resume = Level1TrainConfig(
        data_path=str(pickle_path),
        images_path=str(images_path),
        output_dir=str(out),
        batch_size=2,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        l1_only=True,
        max_windows=4,
        max_steps=1,
        epochs=1,
        epoch_end_inclusive=False,
        optimizer_schedule="constant",
        log_every=1,
        device="cpu",
        pin_memory=False,
        resume_if_possible=True,
        seed=0,
        deterministic=False,
    )
    resumed = Level1Trainer(cfg_resume)
    assert resumed.sample_step == 4
    hist2 = resumed.train()
    assert len(hist2) >= 1
    assert np.isfinite(hist2[-1]["loss"])


def test_loader_shapes(tmp_path: Path) -> None:
    pickle_path, images_path = _write_tiny(tmp_path)
    cfg = Level1TrainConfig(
        data_path=str(pickle_path),
        images_path=str(images_path),
        batch_size=2,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        l1_only=True,
        max_windows=4,
        pin_memory=False,
        device="cpu",
    )
    loader = make_level1_loader(cfg)
    batch = next(iter(loader))
    assert batch["images"].shape == (2, 15, 3, 98, 98)
    assert batch["proprio"].shape == (2, 15, 2)
    assert batch["actions"].shape == (2, 14, 2)
    ds = Level1WindowDataset(
        DiverseMazeOfflineDataset(
            pickle_path, images_path, l1_only=True, max_windows=4
        )
    )
    assert len(ds) == 4
