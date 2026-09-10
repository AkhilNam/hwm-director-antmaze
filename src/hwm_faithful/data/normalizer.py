"""Hard-set Diverse Maze normalizer matching original ``STATS``.

Original (SHA ``e197375``):

- ``pldm_envs/utils/normalizer.py`` ``STATS["maze2d_large_diverse"]``
- YAML: ``data.normalize = true``, ``data.normalizer_hardset = true``
- ``Normalizer.build_normalizer``: if ``normalizer_hardset`` and the loader
  has ``.config``, the hardcoded table is used. Sample mean/std are still
  computed in that function but **discarded**.
- ``min_max_normalize_state`` is **off** in the released YAML.

Formulas (``+ 1e-6`` on the denominator, matching original):

    x_hat = (x - mean) / (std + 1e-6)
    x     = x_hat * std + mean     # unnormalize has **no** 1e-6

Images stay float in roughly ``[0, 255]`` (no ``/255``). Channel stats are
broadcast as ``mean.view(-1, 1, 1)`` over ``[..., C, H, W]``.

Maze proprio **position is empty** in the dataset (not used). Position
stats in the table are zeros and would divide by ``1e-6`` if applied; we
do not apply them on the active maze path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hwm_faithful.shapes import ACTION_DIM, IMAGE_CHANNELS, PROPRIO_DIM

ENV_NAME = "maze2d_large_diverse"
EPS = 1e-6

# Original ``STATS["maze2d_large_diverse"]`` — do not re-estimate.
STATE_MEAN = torch.tensor([146.5709, 120.0509, 93.3956], dtype=torch.float32)
STATE_STD = torch.tensor([84.9847, 45.3689, 10.3962], dtype=torch.float32)
ACTION_MEAN = torch.tensor([0.0004, -0.0022], dtype=torch.float32)
ACTION_STD = torch.tensor([0.4095, 0.4082], dtype=torch.float32)
LOCATION_MEAN = torch.tensor([4.3646, 4.2948], dtype=torch.float32)
LOCATION_STD = torch.tensor([2.3662, 2.3378], dtype=torch.float32)
PROPRIO_POS_MEAN = torch.tensor([0.0, 0.0], dtype=torch.float32)
PROPRIO_POS_STD = torch.tensor([0.0, 0.0], dtype=torch.float32)
PROPRIO_VEL_MEAN = torch.tensor([-0.0291, -0.0461], dtype=torch.float32)
PROPRIO_VEL_STD = torch.tensor([1.4084, 1.4102], dtype=torch.float32)


def _as_tensor(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return value.to(device=ref.device, dtype=ref.dtype)


def _channel_affine(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, *, inverse: bool
) -> torch.Tensor:
    """Per-channel image affine. ``x`` is ``[..., C, H, W]``."""
    if x.ndim < 3:
        raise ValueError(f"image tensor must be [..., C, H, W], got {tuple(x.shape)}")
    mean = _as_tensor(mean, x).view(-1, 1, 1)
    std = _as_tensor(std, x).view(-1, 1, 1)
    n_ch = x.shape[-3]
    if mean.shape[0] != n_ch:
        if mean.shape[0] % n_ch == 0 and mean.shape[0] > n_ch:
            mean = mean[:n_ch]
            std = std[:n_ch]
        else:
            raise ValueError(
                f"image channels {n_ch} vs stats {tuple(mean.shape)}"
            )
    if inverse:
        return x * std + mean
    return (x - mean) / (std + EPS)


def _vec_affine(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, *, inverse: bool
) -> torch.Tensor:
    mean = _as_tensor(mean, x)
    std = _as_tensor(std, x)
    if inverse:
        return x * std + mean
    return (x - mean) / (std + EPS)


@dataclass(frozen=True)
class MazeHardsetStats:
    """Active Diverse Maze hard-set statistics."""

    state_mean: torch.Tensor = STATE_MEAN
    state_std: torch.Tensor = STATE_STD
    action_mean: torch.Tensor = ACTION_MEAN
    action_std: torch.Tensor = ACTION_STD
    location_mean: torch.Tensor = LOCATION_MEAN
    location_std: torch.Tensor = LOCATION_STD
    proprio_vel_mean: torch.Tensor = PROPRIO_VEL_MEAN
    proprio_vel_std: torch.Tensor = PROPRIO_VEL_STD

    def to(self, device: torch.device | str | None = None) -> "MazeHardsetStats":
        if device is None:
            return self
        return MazeHardsetStats(
            state_mean=self.state_mean.to(device),
            state_std=self.state_std.to(device),
            action_mean=self.action_mean.to(device),
            action_std=self.action_std.to(device),
            location_mean=self.location_mean.to(device),
            location_std=self.location_std.to(device),
            proprio_vel_mean=self.proprio_vel_mean.to(device),
            proprio_vel_std=self.proprio_vel_std.to(device),
        )


class MazeNormalizer:
    """Original hard-set normalizer for the active maze fields."""

    def __init__(self, stats: MazeHardsetStats | None = None) -> None:
        self.stats = stats or MazeHardsetStats()

    def normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-3] != IMAGE_CHANNELS:
            raise ValueError(
                f"image channels must be {IMAGE_CHANNELS}, got {tuple(image.shape)}"
            )
        return _channel_affine(
            image.float(), self.stats.state_mean, self.stats.state_std, inverse=False
        )

    def unnormalize_image(self, image: torch.Tensor) -> torch.Tensor:
        return _channel_affine(
            image.float(), self.stats.state_mean, self.stats.state_std, inverse=True
        )

    def normalize_proprio_vel(self, vel: torch.Tensor) -> torch.Tensor:
        if vel.shape[-1] != PROPRIO_DIM:
            raise ValueError(
                f"proprio_vel last dim must be {PROPRIO_DIM}, got {tuple(vel.shape)}"
            )
        return _vec_affine(
            vel.float(),
            self.stats.proprio_vel_mean,
            self.stats.proprio_vel_std,
            inverse=False,
        )

    def unnormalize_proprio_vel(self, vel: torch.Tensor) -> torch.Tensor:
        return _vec_affine(
            vel.float(),
            self.stats.proprio_vel_mean,
            self.stats.proprio_vel_std,
            inverse=True,
        )

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] != ACTION_DIM:
            raise ValueError(
                f"action last dim must be {ACTION_DIM}, got {tuple(action.shape)}"
            )
        return _vec_affine(
            action.float(), self.stats.action_mean, self.stats.action_std, inverse=False
        )

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Original: ``action * std + mean`` (no epsilon)."""
        return _vec_affine(
            action.float(), self.stats.action_mean, self.stats.action_std, inverse=True
        )

    def normalize_location(self, location: torch.Tensor) -> torch.Tensor:
        if location.shape[-1] != 2:
            raise ValueError(
                f"location last dim must be 2, got {tuple(location.shape)}"
            )
        return _vec_affine(
            location.float(),
            self.stats.location_mean,
            self.stats.location_std,
            inverse=False,
        )

    def unnormalize_location(self, location: torch.Tensor) -> torch.Tensor:
        return _vec_affine(
            location.float(),
            self.stats.location_mean,
            self.stats.location_std,
            inverse=True,
        )


def default_normalizer() -> MazeNormalizer:
    return MazeNormalizer()
