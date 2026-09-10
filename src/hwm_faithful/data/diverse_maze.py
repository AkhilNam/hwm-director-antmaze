"""Memory-safe Diverse Maze offline dataset for the faithful HWM.

This is **not** a copy of original ``D4RLDataset``. It implements only the
active maze path used by our Level-1 / Level-2 models.

Disk (probe ``images.npy``, confirmed with ``mmap_mode='r'``):

    uint8 NHWC  [N, 98, 98, 3]

Loader (matching ``D4RLDataset._load_images_tensor`` then collate normalize):

    slice → float32 NCHW  (no /255, no crop, no resize, no stacking)
    then (x - state_mean) / (state_std + 1e-6) if ``normalize=True``

Pickle ``data.p`` (torch.load list of episode dicts):

    observations  [T+1, 4] float64   (x, y, vx, vy)
    actions       [T, 2]   float32
    map_idx       int

Proprio that reaches M1 is **velocity** ``observations[:, 2:]``. Position
is used only as ``locations`` for metrics / probing.

L1 window (``n_steps=15``):

    images   [15, 3, 98, 98]
    proprio  [15, 2]
    actions  [14, 2]
    xy       [15, 2]

L2 primitive window (``l2_n_steps_total = 6*10+1 = 61``):

    images   [61, 3, 98, 98]   (no skip here)
    proprio  [61, 2]
    actions  [60, 2]
    xy       [61, 2]

Original ``__getitem__`` already skip-samples L2 **images/proprio** with
``skip_frame=10`` and chunks **unskipped** actions. We expose the dense
61-step window and reuse ``build_level2_inputs`` for skip/chunk.

Train/val: released D4RL factory sets ``val_ds=None``. ``val_fraction=0.2``
is unused. The 25-map pickle is the training corpus; probe is separate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from hwm_faithful.data.normalizer import MazeNormalizer, default_normalizer
from hwm_faithful.level2.temporal_abstraction import Level2TemporalConfig
from hwm_faithful.shapes import IMAGE_CHANNELS, IMAGE_SIZE

# YAML ``n_steps: 15``. Actions are ``n_steps - 1``.
L1_N_STEPS = 15
L1_N_ACTIONS = L1_N_STEPS - 1


DEFAULT_ORIGINAL_ROOT = Path(
    os.environ.get(
        "HWM_ORIGINAL_ROOT",
        "/storage/scratch1/2/anampally3/hwm-original-repro",
    )
)
DEFAULT_PROBE_DIR = DEFAULT_ORIGINAL_ROOT / (
    "pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe"
)
DEFAULT_TRAIN_DIR = DEFAULT_ORIGINAL_ROOT / (
    "pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps"
)


def images_to_nchw_float(images_nhwc: np.ndarray | torch.Tensor) -> torch.Tensor:
    """uint8/float NHWC → float32 NCHW in the stored ``[0, 255]`` range.

    Matches ``torch.from_numpy(...).permute(0, 3, 1, 2).float()``.
    Accepts ``[H, W, C]`` or ``[T, H, W, C]``.
    """
    if isinstance(images_nhwc, torch.Tensor):
        arr = images_nhwc
    else:
        arr = torch.from_numpy(np.array(images_nhwc, copy=True))
    if arr.ndim == 3:
        arr = arr.unsqueeze(0)
        squeeze = True
    elif arr.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"expected HWC or THWC, got {tuple(arr.shape)}")
    if arr.shape[-1] != IMAGE_CHANNELS:
        raise ValueError(f"last dim must be {IMAGE_CHANNELS}, got {tuple(arr.shape)}")
    out = arr.permute(0, 3, 1, 2).contiguous().float()
    if squeeze:
        out = out[0]
    return out


def preprocess_image(
    images_nhwc: np.ndarray | torch.Tensor,
    normalizer: MazeNormalizer | None = None,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Stored uint8 NHWC → model image tensor.

    After this call the tensor is the value original HWM's encoder sees
    (normalized float NCHW when ``normalize=True``).
    """
    nchw = images_to_nchw_float(images_nhwc)
    if nchw.ndim == 3:
        if nchw.shape != (IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"expected [{IMAGE_CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE}], "
                f"got {tuple(nchw.shape)}"
            )
    elif nchw.shape[-3:] != (IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"expected [..., {IMAGE_CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE}], "
            f"got {tuple(nchw.shape)}"
        )
    if not normalize:
        return nchw
    norm = normalizer or default_normalizer()
    return norm.normalize_image(nchw)


def proprio_vel_from_obs(observations: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Maze proprio for M1: ``observations[..., 2:]`` → ``[..., 2]`` velocity."""
    obs = torch.as_tensor(observations).float()
    if obs.shape[-1] < 4:
        raise ValueError(f"observations last dim must be >= 4, got {tuple(obs.shape)}")
    return obs[..., 2:4]


def locations_from_obs(observations: np.ndarray | torch.Tensor) -> torch.Tensor:
    obs = torch.as_tensor(observations).float()
    return obs[..., :2]


class Level1Window(NamedTuple):
    """One L1 training sample, time-leading, batch-free until stacked."""

    images: torch.Tensor  # [15, 3, 98, 98] normalized if requested
    proprio: torch.Tensor  # [15, 2]
    actions: torch.Tensor  # [14, 2]
    locations: torch.Tensor  # [15, 2] unnormalized xy (metrics)
    images_raw: torch.Tensor  # [15, 3, 98, 98] float [0,255] before affine
    episode_idx: int
    start_idx: int
    global_index: int


class Level2PrimitiveWindow(NamedTuple):
    """61 primitive steps used to build skip-10 L2 inputs."""

    images: torch.Tensor  # [61, 3, 98, 98]
    proprio: torch.Tensor  # [61, 2]
    actions: torch.Tensor  # [60, 2]
    locations: torch.Tensor  # [61, 2]
    images_raw: torch.Tensor
    episode_idx: int
    start_idx: int
    global_index: int


@dataclass(frozen=True)
class DiverseMazePaths:
    pickle_path: Path
    images_path: Path | None

    @classmethod
    def probe(cls, root: Path | None = None) -> "DiverseMazePaths":
        d = (root or DEFAULT_PROBE_DIR)
        img = d / "images.npy"
        return cls(pickle_path=d / "data.p", images_path=img if img.is_file() else None)

    @classmethod
    def train_25maps(cls, root: Path | None = None) -> "DiverseMazePaths":
        d = (root or DEFAULT_TRAIN_DIR)
        img = d / "images.npy"
        return cls(pickle_path=d / "data.p", images_path=img if img.is_file() else None)


def probe_data_available(root: Path | None = None) -> bool:
    paths = DiverseMazePaths.probe(root)
    return paths.pickle_path.is_file() and paths.images_path is not None


def open_images_memmap(images_path: str | Path) -> np.ndarray:
    """Open ``images.npy`` with ``mmap_mode='r'``. Does not load the file."""
    path = Path(images_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r")


class DiverseMazeOfflineDataset:
    """Episode pickle + optional mmap images. No DataLoader, no full RAM load."""

    def __init__(
        self,
        pickle_path: str | Path,
        images_path: str | Path | None = None,
        *,
        n_steps: int = L1_N_STEPS,
        l2_config: Level2TemporalConfig | None = None,
        normalizer: MazeNormalizer | None = None,
        normalize: bool = True,
        images_memmap: np.ndarray | None = None,
        l1_only: bool = False,
        max_windows: int | None = None,
    ) -> None:
        self.pickle_path = Path(pickle_path)
        self.images_path = Path(images_path) if images_path is not None else None
        self.n_steps = int(n_steps)
        self.l2_config = l2_config or Level2TemporalConfig()
        self.l1_only = bool(l1_only)
        self.max_windows = max_windows
        self.normalizer = normalizer or default_normalizer()
        self.normalize = bool(normalize)
        if not self.pickle_path.is_file():
            raise FileNotFoundError(self.pickle_path)

        try:
            self.splits = torch.load(
                self.pickle_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            self.splits = torch.load(self.pickle_path, map_location="cpu")
        if not isinstance(self.splits, list) or not self.splits:
            raise ValueError(f"expected non-empty episode list in {self.pickle_path}")

        l2_window = 0 if self.l1_only else self.l2_config.l1_window
        max_n_steps = max(self.n_steps, l2_window)
        self.episode_lengths = np.array(
            [len(ep["observations"]) for ep in self.splits], dtype=np.int64
        )
        valid = self.episode_lengths - max_n_steps
        if np.any(valid <= 0):
            raise ValueError(
                f"episodes shorter than max_n_steps={max_n_steps}: "
                f"min_len={int(self.episode_lengths.min())}"
            )
        self.cum_lengths = np.cumsum(valid)
        self.cum_lengths_total = np.cumsum(self.episode_lengths)

        if images_memmap is not None:
            self.images = images_memmap
        elif self.images_path is not None:
            self.images = open_images_memmap(self.images_path)
        else:
            self.images = None

        if self.images is not None:
            if self.images.ndim != 4:
                raise ValueError(
                    f"images.npy must be NHWC, got shape {self.images.shape}"
                )
            n_frames = int(self.cum_lengths_total[-1])
            if self.images.shape[0] != n_frames:
                raise ValueError(
                    f"images N={self.images.shape[0]} != pickle frames {n_frames}"
                )

    def __len__(self) -> int:
        n = int(self.cum_lengths[-1])
        if self.max_windows is not None:
            return min(n, int(self.max_windows))
        return n

    def _locate(self, idx: int) -> tuple[int, int, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        episode_idx = int(np.searchsorted(self.cum_lengths, idx, side="right"))
        start_idx = (
            idx - int(self.cum_lengths[episode_idx - 1]) if episode_idx > 0 else idx
        )
        if episode_idx == 0:
            global_index = start_idx
        else:
            global_index = int(self.cum_lengths_total[episode_idx - 1]) + start_idx
        return episode_idx, int(start_idx), int(global_index)

    def _load_image_slice(self, global_index: int, length: int) -> torch.Tensor:
        if self.images is None:
            raise RuntimeError("images.npy is not available for this dataset")
        # Copy only the requested window. Do not materialize the mmap.
        sl = np.asarray(self.images[global_index : global_index + length])
        return images_to_nchw_float(sl)

    def _episode_actions(self, episode_idx: int, start_idx: int, n_states: int) -> torch.Tensor:
        ep = self.splits[episode_idx]
        # Original: actions[start : start + length - 1]  (no skip).
        raw = ep["actions"][start_idx : start_idx + n_states - 1]
        return torch.from_numpy(np.asarray(raw)).float()

    def _episode_obs(self, episode_idx: int, start_idx: int, n_states: int) -> torch.Tensor:
        raw = self.splits[episode_idx]["observations"][start_idx : start_idx + n_states]
        return torch.from_numpy(np.asarray(raw)).float()

    def _maybe_normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return images
        return self.normalizer.normalize_image(images)

    def _maybe_normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return proprio
        return self.normalizer.normalize_proprio_vel(proprio)

    def _maybe_normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return actions
        return self.normalizer.normalize_action(actions)

    def get_level1_window(self, idx: int = 0) -> Level1Window:
        episode_idx, start_idx, global_index = self._locate(idx)
        images_raw = self._load_image_slice(global_index, self.n_steps)
        obs = self._episode_obs(episode_idx, start_idx, self.n_steps)
        actions = self._episode_actions(episode_idx, start_idx, self.n_steps)
        proprio = self._maybe_normalize_proprio(proprio_vel_from_obs(obs))
        return Level1Window(
            images=self._maybe_normalize_images(images_raw),
            proprio=proprio,
            actions=self._maybe_normalize_actions(actions),
            locations=locations_from_obs(obs),
            images_raw=images_raw,
            episode_idx=episode_idx,
            start_idx=start_idx,
            global_index=global_index,
        )

    def get_level2_primitive_window(self, idx: int = 0) -> Level2PrimitiveWindow:
        n_states = self.l2_config.l1_window
        episode_idx, start_idx, global_index = self._locate(idx)
        images_raw = self._load_image_slice(global_index, n_states)
        obs = self._episode_obs(episode_idx, start_idx, n_states)
        actions = self._episode_actions(episode_idx, start_idx, n_states)
        proprio = self._maybe_normalize_proprio(proprio_vel_from_obs(obs))
        return Level2PrimitiveWindow(
            images=self._maybe_normalize_images(images_raw),
            proprio=proprio,
            actions=self._maybe_normalize_actions(actions),
            locations=locations_from_obs(obs),
            images_raw=images_raw,
            episode_idx=episode_idx,
            start_idx=start_idx,
            global_index=global_index,
        )

    def describe(self) -> dict[str, object]:
        img = None
        if self.images is not None:
            sl = np.asarray(self.images[0])
            img = {
                "shape": tuple(self.images.shape),
                "dtype": str(self.images.dtype),
                "sample0_shape": tuple(sl.shape),
                "sample0_min": float(sl.min()),
                "sample0_max": float(sl.max()),
                "mmap": True,
            }
        ep0 = self.splits[0]
        return {
            "pickle": str(self.pickle_path),
            "images_path": None if self.images_path is None else str(self.images_path),
            "n_episodes": len(self.splits),
            "n_windows": len(self),
            "episode0_keys": list(ep0.keys()),
            "episode0_obs": tuple(ep0["observations"].shape),
            "episode0_act": tuple(ep0["actions"].shape),
            "obs_dtype": str(ep0["observations"].dtype),
            "act_dtype": str(ep0["actions"].dtype),
            "total_frames": int(self.cum_lengths_total[-1]),
            "images": img,
            "l1_n_steps": self.n_steps,
            "l2_l1_window": self.l2_config.l1_window,
        }


def load_probe_dataset(
    *,
    normalize: bool = True,
    require_images: bool = True,
) -> DiverseMazeOfflineDataset:
    paths = DiverseMazePaths.probe()
    if require_images and paths.images_path is None:
        raise FileNotFoundError("probe images.npy is not available")
    return DiverseMazeOfflineDataset(
        paths.pickle_path,
        paths.images_path,
        normalize=normalize,
    )


def add_time_batch(window_images: torch.Tensor, window_proprio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[T, C, H, W]`` + ``[T, 2]`` → time-leading batch ``[T, 1, ...]``."""
    return window_images.unsqueeze(1), window_proprio.unsqueeze(1)
