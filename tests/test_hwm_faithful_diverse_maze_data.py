"""Unit tests for faithful Diverse Maze data (M8).

Pure unit tests never open the multi-GB corpus. Probe/PACE integration
tests skip if those files are missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hwm_faithful.data.diverse_maze import (
    L1_N_ACTIONS,
    L1_N_STEPS,
    DiverseMazeOfflineDataset,
    add_time_batch,
    images_to_nchw_float,
    load_probe_dataset,
    preprocess_image,
    probe_data_available,
)
from hwm_faithful.data.env import (
    SUCCESS_DISTANCE_THRESHOLD,
    encode_goal,
    load_starts_targets,
    start_target_path,
    success_from_distance,
    success_from_reward,
)
from hwm_faithful.data.normalizer import (
    ACTION_MEAN,
    ACTION_STD,
    EPS,
    STATE_MEAN,
    STATE_STD,
    MazeNormalizer,
    default_normalizer,
)
from hwm_faithful.level2.temporal_abstraction import build_level2_inputs
from hwm_faithful.level2.world_model import Level2WorldModel
from hwm_faithful.losses.level1_objective import Level1Objective
from hwm_faithful.losses.level2_objective import level2_prediction_losses
from hwm_faithful.models.level1_encoder import Level1Encoder
from hwm_faithful.models.level1_world_model import Level1WorldModel
from hwm_faithful.shapes import (
    FEATURE_SIZE,
    FUSED_CHANNELS,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    L2_L1_WINDOW,
    L2_N_STATES,
    L2_PRIMITIVE_HORIZON,
    VISUAL_CHANNELS,
)


def _batch2_l1(window) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VICRegObs requires B>=2; duplicate the single window."""
    images = window.images.unsqueeze(1).repeat(1, 2, 1, 1, 1)
    proprio = window.proprio.unsqueeze(1).repeat(1, 2, 1)
    actions = window.actions.unsqueeze(1).repeat(1, 2, 1)
    return images, proprio, actions


def _write_tiny_dataset(tmp: Path, n_states: int = 70, n_episodes: int = 2) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    splits = []
    frames = []
    for ep in range(n_episodes):
        obs = rng.normal(size=(n_states, 4)).astype(np.float64)
        obs[:, :2] = np.clip(obs[:, :2] * 2 + 4.0, 0.5, 9.5)
        act = rng.uniform(-1.0, 1.0, size=(n_states - 1, 2)).astype(np.float32)
        splits.append({"actions": act, "observations": obs, "map_idx": ep})
        img = rng.integers(20, 220, size=(n_states, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        frames.append(img)
    pickle_path = tmp / "data.p"
    images_path = tmp / "images.npy"
    torch.save(splits, pickle_path)
    np.save(images_path, np.concatenate(frames, axis=0))
    return pickle_path, images_path


def test_uint8_nhwc_to_nchw_float_no_div255() -> None:
    raw = np.array([[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8)
    # pad conceptually: use full 98 later; this helper only permutes
    raw98 = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    raw98[0, :2, :] = raw[0]
    out = images_to_nchw_float(raw98)
    assert out.shape == (IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    assert out.dtype == torch.float32
    assert out[0, 0, 0] == 10.0
    assert out[1, 0, 0] == 20.0
    assert out[2, 0, 0] == 30.0
    assert out.max() <= 255.0


def test_image_normalize_matches_hardset_formula() -> None:
    raw = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 100, dtype=np.uint8)
    got = preprocess_image(raw)
    expected = (torch.tensor([100.0, 100.0, 100.0]).view(3, 1, 1) - STATE_MEAN.view(3, 1, 1)) / (
        STATE_STD.view(3, 1, 1) + EPS
    )
    torch.testing.assert_close(got, expected.expand_as(got))


def test_proprio_and_action_roundtrip() -> None:
    n = default_normalizer()
    vel = torch.tensor([[0.5, -1.25], [0.0, 0.0]], dtype=torch.float32)
    act = torch.tensor([[0.8, -0.3], [0.0, 1.0]], dtype=torch.float32)
    nvel = n.normalize_proprio_vel(vel)
    nact = n.normalize_action(act)
    torch.testing.assert_close(n.unnormalize_proprio_vel(nvel), vel, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(n.unnormalize_action(nact), act, rtol=1e-5, atol=1e-5)
    expected_act = (act - ACTION_MEAN) / (ACTION_STD + EPS)
    torch.testing.assert_close(nact, expected_act)


def test_unnormalize_action_has_no_eps() -> None:
    n = MazeNormalizer()
    z = torch.ones(1, 2)
    got = n.unnormalize_action(z)
    expected = z * ACTION_STD + ACTION_MEAN
    torch.testing.assert_close(got, expected)


def test_success_threshold() -> None:
    xy = np.array([0.0, 0.0])
    assert success_from_distance(xy, np.array([0.49, 0.0])) is True
    assert success_from_distance(xy, np.array([0.5, 0.0])) is False
    assert success_from_distance(xy, np.array([0.51, 0.0])) is False
    assert SUCCESS_DISTANCE_THRESHOLD == 0.5
    assert success_from_reward(1.0) is True
    assert success_from_reward(0.0) is False


def test_level1_and_level2_window_indexing(tmp_path: Path) -> None:
    pickle_path, images_path = _write_tiny_dataset(tmp_path)
    ds = DiverseMazeOfflineDataset(pickle_path, images_path, normalize=True)
    w1 = ds.get_level1_window(0)
    assert w1.images.shape == (L1_N_STEPS, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert w1.proprio.shape == (L1_N_STEPS, 2)
    assert w1.actions.shape == (L1_N_ACTIONS, 2)
    assert w1.locations.shape == (L1_N_STEPS, 2)
    assert w1.images_raw.dtype == torch.float32
    assert w1.images_raw.min() >= 0.0
    assert w1.images_raw.max() <= 255.0

    mmap = np.load(images_path, mmap_mode="r")
    raw0 = torch.from_numpy(np.asarray(mmap[0])).permute(2, 0, 1).float()
    torch.testing.assert_close(w1.images_raw[0], raw0)

    w2 = ds.get_level2_primitive_window(0)
    assert w2.images.shape == (L2_L1_WINDOW, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert w2.proprio.shape == (L2_L1_WINDOW, 2)
    assert w2.actions.shape == (L2_PRIMITIVE_HORIZON, 2)
    imgs_b, prop_b = add_time_batch(w2.images, w2.proprio)
    fused = torch.zeros(L2_L1_WINDOW, 1, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    l2_in = build_level2_inputs(fused, w2.actions.unsqueeze(1))
    assert l2_in.states.shape[0] == L2_N_STATES
    assert l2_in.action_chunks.shape == (6, 1, 10, 2)
    assert l2_in.state_indices == (0, 10, 20, 30, 40, 50, 60)


def test_mmap_does_not_own_full_array(tmp_path: Path) -> None:
    pickle_path, images_path = _write_tiny_dataset(tmp_path, n_states=62, n_episodes=1)
    ds = DiverseMazeOfflineDataset(pickle_path, images_path)
    assert isinstance(ds.images, np.memmap)


def test_level1_forward_loss_finite_on_tiny(tmp_path: Path) -> None:
    pickle_path, images_path = _write_tiny_dataset(tmp_path, n_states=62, n_episodes=1)
    ds = DiverseMazeOfflineDataset(pickle_path, images_path)
    w = ds.get_level1_window(0)
    model = Level1WorldModel()
    model.eval()
    images, proprio, actions = _batch2_l1(w)
    with torch.no_grad():
        enc, rolled = model.encode_and_rollout(images, proprio, actions)
        obj = Level1Objective()
        losses = obj(
            obs_gt=enc.visual,
            obs_pred=rolled.obs_component,
            proprio_gt=enc.proprio_map,
            proprio_pred=rolled.proprio_component,
            fused_gt=enc.fused,
            actions=actions,
        )
    assert enc.fused.shape == (L1_N_STEPS, 2, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    assert rolled.fused.shape == (L1_N_STEPS, 2, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.vicreg_obs)
    assert torch.isfinite(losses.idm)
    assert torch.isfinite(losses.prediction_proprio)


def test_level2_forward_loss_finite_on_tiny(tmp_path: Path) -> None:
    pickle_path, images_path = _write_tiny_dataset(tmp_path, n_states=62, n_episodes=1)
    ds = DiverseMazeOfflineDataset(pickle_path, images_path)
    w = ds.get_level2_primitive_window(0)
    l1 = Level1WorldModel()
    l1.eval()
    images, proprio = add_time_batch(w.images, w.proprio)
    with torch.no_grad():
        enc = l1.encode_sequence(images, proprio)
        l2_in = build_level2_inputs(enc.fused, w.actions.unsqueeze(1))
        l2 = Level2WorldModel()
        l2.eval()
        fwd = l2.forward_train(l2_in.states, l2_in.action_chunks)
        losses = level2_prediction_losses(fwd.h_gt, fwd.h_pred)
    assert fwd.h_gt.shape == (7, 1, 18, 43, 43)
    assert l2_in.action_chunks.shape == (6, 1, 10, 2)
    assert fwd.z.shape == (6, 1, 8)
    assert fwd.h_pred.shape == (7, 1, 18, 43, 43)
    assert torch.isfinite(losses.total)


def test_goal_encoding_shape() -> None:
    enc = Level1Encoder()
    enc.eval()
    img = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    visual = encode_goal(enc, img)
    assert visual.shape == (2, VISUAL_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    visual1 = encode_goal(enc, img[0])
    assert visual1.shape == (1, VISUAL_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)


@pytest.mark.skipif(not probe_data_available(), reason="probe images.npy / data.p not on this machine")
def test_real_probe_l1_l2_windows_and_finite_losses() -> None:
    ds = load_probe_dataset()
    info = ds.describe()
    assert info["n_episodes"] == 1000
    assert info["images"]["dtype"] == "uint8"
    assert info["images"]["shape"][1:] == (98, 98, 3)
    w1 = ds.get_level1_window(0)
    assert w1.images.shape == (15, 3, 98, 98)
    assert w1.actions.shape == (14, 2)
    w2 = ds.get_level2_primitive_window(0)
    assert w2.images.shape == (61, 3, 98, 98)
    assert w2.actions.shape == (60, 2)
    model = Level1WorldModel()
    model.eval()
    images, proprio, actions = _batch2_l1(w1)
    with torch.no_grad():
        enc, rolled = model.encode_and_rollout(images, proprio, actions)
        losses = Level1Objective()(
            obs_gt=enc.visual,
            obs_pred=rolled.obs_component,
            proprio_gt=enc.proprio_map,
            proprio_pred=rolled.proprio_component,
            fused_gt=enc.fused,
            actions=actions,
        )
        enc2 = model.encode_sequence(*add_time_batch(w2.images, w2.proprio))
        l2_in = build_level2_inputs(enc2.fused, w2.actions.unsqueeze(1))
        l2 = Level2WorldModel()
        l2.eval()
        fwd = l2.forward_train(l2_in.states, l2_in.action_chunks)
        l2_losses = level2_prediction_losses(fwd.h_gt, fwd.h_pred)
    assert torch.isfinite(losses.total)
    assert fwd.h_gt.shape == (7, 1, 18, 43, 43)
    assert torch.isfinite(l2_losses.total)


@pytest.mark.skipif(not start_target_path().is_file(), reason="starts_targets_9_12.pt missing")
def test_start_target_loader() -> None:
    trials = load_starts_targets(difficulty="medium")
    assert len(trials) == 40
    t0 = trials.trial(0)
    assert t0["start"].shape == (2,)
    assert t0["target"].shape == (2,)
    assert isinstance(t0["map_layout"], str)
    assert "block_dist" in t0 and "turns" in t0
    hard = load_starts_targets(difficulty="hard")
    assert len(hard) == 40
