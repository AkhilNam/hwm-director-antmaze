"""Original-vs-ours preprocessing and copied-weight parity (M8).

Skipped if the original HWM clone or probe corpus is unavailable.
These tests copy original *random* weights into ours; they do not deploy
original checkpoints as the faithful model.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from hwm_faithful.data.diverse_maze import (
    DEFAULT_ORIGINAL_ROOT,
    DEFAULT_PROBE_DIR,
    images_to_nchw_float,
    load_probe_dataset,
    preprocess_image,
    probe_data_available,
    proprio_vel_from_obs,
)
from hwm_faithful.data.normalizer import default_normalizer
from hwm_faithful.data.weight_copy import (
    build_original_l1_encoder,
    build_original_l1_predictor,
    build_original_l2_predictor,
    build_original_posterior,
    copy_encoder_weights,
    copy_l1_predictor_weights,
    copy_l2_action_encoder_weights,
    copy_l2_predictor_conv_weights,
    copy_posterior_weights,
    original_models_importable,
)
from hwm_faithful.level2.action_encoder import Level2ActionEncoder
from hwm_faithful.level2.posterior import Level2ActionPosterior
from hwm_faithful.level2.predictor import Level2Predictor
from hwm_faithful.models.conv_predictor import Level1Predictor
from hwm_faithful.models.level1_encoder import Level1Encoder
from hwm_faithful.shapes import FEATURE_SIZE, FUSED_CHANNELS, IMAGE_SIZE, Z_DIM

pytestmark = pytest.mark.skipif(
    not original_models_importable(),
    reason="original HWM_PLDM clone not importable",
)


def _add_original_path() -> None:
    root = str(DEFAULT_ORIGINAL_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def test_image_preprocess_parity_with_original_normalizer() -> None:
    _add_original_path()
    from pldm_envs.utils.normalizer import STATS, Normalizer

    rng = np.random.default_rng(1)
    raw = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    ours = preprocess_image(raw)
    stats = STATS["maze2d_large_diverse"]
    orig_n = Normalizer(
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
        action_mean=stats["action_mean"],
        action_std=stats["action_std"],
        location_mean=stats["location_mean"],
        location_std=stats["location_std"],
        proprio_pos_mean=stats["proprio_pos_mean"],
        proprio_pos_std=stats["proprio_pos_std"],
        proprio_vel_mean=stats["proprio_vel_mean"],
        proprio_vel_std=stats["proprio_vel_std"],
        min_max_state=False,
        image_based=True,
    )
    nchw = torch.from_numpy(raw).permute(2, 0, 1).float()
    theirs = orig_n.normalize_state(nchw)
    torch.testing.assert_close(ours, theirs, rtol=0.0, atol=0.0)


def test_proprio_and_action_parity_with_original() -> None:
    _add_original_path()
    from pldm_envs.utils.normalizer import STATS, Normalizer

    stats = STATS["maze2d_large_diverse"]
    orig_n = Normalizer(
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
        action_mean=stats["action_mean"],
        action_std=stats["action_std"],
        location_mean=stats["location_mean"],
        location_std=stats["location_std"],
        proprio_pos_mean=stats["proprio_pos_mean"],
        proprio_pos_std=stats["proprio_pos_std"],
        proprio_vel_mean=stats["proprio_vel_mean"],
        proprio_vel_std=stats["proprio_vel_std"],
    )
    ours = default_normalizer()
    vel = torch.tensor([[1.2, -0.4], [0.0, 3.1]], dtype=torch.float32)
    act = torch.tensor([[-0.9, 0.5], [0.25, -0.25]], dtype=torch.float32)
    torch.testing.assert_close(
        ours.normalize_proprio_vel(vel), orig_n.normalize_proprio_vel(vel)
    )
    torch.testing.assert_close(ours.normalize_action(act), orig_n.normalize_action(act))
    nact = ours.normalize_action(act)
    torch.testing.assert_close(
        ours.unnormalize_action(nact), orig_n.unnormalize_action(nact)
    )


@pytest.mark.skipif(not probe_data_available(), reason="probe dataset missing")
def test_loader_matches_original_d4rl_getitem() -> None:
    _add_original_path()
    from pldm_envs.diverse_maze.d4rl import D4RLDataset
    from pldm_envs.diverse_maze.enums import D4RLDatasetConfig

    ours_ds = load_probe_dataset(normalize=False)
    cfg = D4RLDatasetConfig(
        env_name="maze2d_large_diverse",
        path=str(DEFAULT_PROBE_DIR / "data.p"),
        images_path=str(DEFAULT_PROBE_DIR / "images.npy"),
        n_steps=15,
        l2_n_steps=6,
        l2_step_skip=10,
        chunked_actions=True,
        stack_states=1,
        image_based=True,
        train=True,
    )
    orig = D4RLDataset(cfg, load_l1=True)
    sample = orig[0]
    w1 = ours_ds.get_level1_window(0)
    torch.testing.assert_close(w1.images_raw, sample.states.float())
    torch.testing.assert_close(w1.proprio, sample.proprio_vel.float())
    torch.testing.assert_close(w1.actions, sample.actions.float())
    w2 = ours_ds.get_level2_primitive_window(0)
    # Original L2 images are already skip-10 (7 frames).
    skip = w2.images_raw[::10]
    torch.testing.assert_close(skip, sample.l2_states.float())
    torch.testing.assert_close(w2.actions, sample.l2_actions.reshape(60, 2).float())


def test_copied_weight_l1_encoder_parity() -> None:
    torch.manual_seed(0)
    orig = build_original_l1_encoder()
    ours = Level1Encoder()
    copy_encoder_weights(orig, ours)
    orig.eval()
    ours.eval()
    image = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    proprio = torch.randn(2, 2)
    with torch.no_grad():
        o = orig(image, proprio=proprio)
        y = ours(image, proprio)
    torch.testing.assert_close(y.visual, o.obs_component, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y.fused, o.encodings, rtol=1e-5, atol=1e-5)


def test_copied_weight_l1_predictor_parity() -> None:
    torch.manual_seed(1)
    orig = build_original_l1_predictor()
    ours = Level1Predictor()
    copy_l1_predictor_weights(orig, ours)
    orig.eval()
    ours.eval()
    h = torch.randn(2, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    a = torch.randn(2, 2)
    with torch.no_grad():
        o = orig(h, curr_action=a, curr_obs=None, curr_proprio=None)
        y = ours(h, a)
    torch.testing.assert_close(y.fused, o, rtol=1e-5, atol=1e-5)


def test_copied_weight_posterior_parity() -> None:
    torch.manual_seed(2)
    orig = build_original_posterior()
    ours = Level2ActionPosterior()
    copy_posterior_weights(orig, ours)
    orig.eval()
    ours.eval()
    chunk = torch.randn(4, 10, 2)
    flat = chunk.reshape(4, 20)
    with torch.no_grad():
        o_mu, o_std = orig(flat)
        y = ours(chunk)
    torch.testing.assert_close(y.mu, o_mu, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y.std, o_std, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y.z, o_mu, rtol=1e-5, atol=1e-5)


def test_copied_weight_l2_action_encoder_and_predictor_parity() -> None:
    torch.manual_seed(3)
    orig = build_original_l2_predictor()
    ours = Level2Predictor()
    copy_l2_action_encoder_weights(orig, ours.action_encoder)
    copy_l2_predictor_conv_weights(orig, ours)
    orig.eval()
    ours.eval()
    z = torch.randn(2, Z_DIM)
    h = torch.randn(2, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    with torch.no_grad():
        o_map = orig.action_encoder(z)
        y_enc = ours.action_encoder(z)
        torch.testing.assert_close(y_enc.spatial, o_map, rtol=1e-5, atol=1e-5)
        o_h = orig(h, curr_action=z, curr_obs=None, curr_proprio=None)
        y = ours(h, z)
    torch.testing.assert_close(y.fused, o_h, rtol=1e-5, atol=1e-5)
