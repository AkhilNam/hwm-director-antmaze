"""Eval interface: starts/targets, goal encoding, success, env adapter.

Allowed to depend on the original environment package. Model code stays in
``hwm_faithful``. Original sources (SHA ``e197375``):

- ``starts_targets_{5_8,9_12,13_16}.pt`` via ``EnvsGenerator``
- ``NormEvalWrapper.get_target_obs`` / ``get_target_proprio``
- ``ant_draw.CustomMazeEnv._is_goal_reached`` (distance < 0.5)
- ``HierarchicalD4RLMPCEvaluator._construct_report`` (reward became True)
- ``mpc.py._encode_targets``: L1 backbone(goal_image, proprio=zeros)
  → ``obs_component`` ``[B, 16, 43, 43]``
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

from hwm_faithful.data.diverse_maze import DEFAULT_ORIGINAL_ROOT, DEFAULT_PROBE_DIR
from hwm_faithful.data.normalizer import MazeNormalizer, default_normalizer
from hwm_faithful.models.level1_encoder import Level1Encoder, Level1EncoderOutput
from hwm_faithful.shapes import IMAGE_CHANNELS, IMAGE_SIZE, PROPRIO_DIM, VISUAL_CHANNELS

SUCCESS_DISTANCE_THRESHOLD = 0.5
ENV_NAME = "maze2d_large_diverse"
ACTION_REPEAT = 4
ACTION_REPEAT_MODE = "id"

START_TARGET_FILES = {
    "easy": "starts_targets_5_8.pt",
    "medium": "starts_targets_9_12.pt",
    "hard": "starts_targets_13_16.pt",
}


class StartTargetSet(NamedTuple):
    starts: list[np.ndarray]
    targets: list[np.ndarray]
    map_layouts: list[str]
    block_dists: list[Any]
    turns: list[Any]
    path: Path

    def __len__(self) -> int:
        return len(self.starts)

    def trial(self, index: int = 0) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return {
            "start": np.asarray(self.starts[index], dtype=np.float64),
            "target": np.asarray(self.targets[index], dtype=np.float64),
            "map_layout": self.map_layouts[index],
            "block_dist": self.block_dists[index],
            "turns": self.turns[index],
        }


def start_target_path(
    difficulty: str = "medium", probe_dir: Path | None = None
) -> Path:
    if difficulty not in START_TARGET_FILES:
        raise ValueError(
            f"difficulty must be one of {tuple(START_TARGET_FILES)}, got {difficulty!r}"
        )
    d = probe_dir or DEFAULT_PROBE_DIR
    return d / START_TARGET_FILES[difficulty]


def load_starts_targets(
    path: str | Path | None = None, *, difficulty: str = "medium"
) -> StartTargetSet:
    pt = Path(path) if path is not None else start_target_path(difficulty)
    if not pt.is_file():
        raise FileNotFoundError(pt)
    try:
        blob = torch.load(pt, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(pt, map_location="cpu")
    if not isinstance(blob, dict):
        raise ValueError(f"starts/targets file must be a dict, got {type(blob)}")
    required = ("starts", "targets", "map_layouts", "block_dists", "turns")
    missing = [k for k in required if k not in blob]
    if missing:
        raise ValueError(f"{pt} missing keys {missing}")
    return StartTargetSet(
        starts=list(blob["starts"]),
        targets=list(blob["targets"]),
        map_layouts=list(blob["map_layouts"]),
        block_dists=list(blob["block_dists"]),
        turns=list(blob["turns"]),
        path=pt,
    )


def success_from_distance(
    xy: np.ndarray | torch.Tensor,
    target_xy: np.ndarray | torch.Tensor,
    threshold: float = SUCCESS_DISTANCE_THRESHOLD,
) -> bool:
    """Conceptual metric: ``||xy - target_xy||_2 < 0.5``."""
    a = np.asarray(xy, dtype=np.float64).reshape(-1)[:2]
    b = np.asarray(target_xy, dtype=np.float64).reshape(-1)[:2]
    return bool(np.linalg.norm(a - b) < threshold)


def success_from_reward(reward: float | int | bool) -> bool:
    """Executed original path: env reward is 1 iff distance < 0.5."""
    return bool(reward)


def encode_goal(
    encoder: Level1Encoder,
    goal_image: torch.Tensor,
    goal_proprio: torch.Tensor | None = None,
) -> torch.Tensor:
    """MPPI target: L1 visual ``[B, 16, 43, 43]``.

    Original ``_encode_targets`` for hierarchical planning:

    1. ``get_target_obs()`` → already-normalized goal image
    2. ``get_target_proprio()`` → zeros (goal velocity unused)
    3. L1 backbone → ``obs_component``
    4. L2 identity encoder is a no-op on that visual map

    ``goal_image`` must already be the model tensor (normalized NCHW).
    """
    if goal_image.ndim == 3:
        goal_image = goal_image.unsqueeze(0)
    if goal_image.ndim != 4 or goal_image.shape[1:] != (
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):
        raise ValueError(
            f"goal_image must be [B, {IMAGE_CHANNELS}, {IMAGE_SIZE}, "
            f"{IMAGE_SIZE}], got {tuple(goal_image.shape)}"
        )
    b = goal_image.shape[0]
    if goal_proprio is None:
        goal_proprio = torch.zeros(
            b, PROPRIO_DIM, dtype=goal_image.dtype, device=goal_image.device
        )
    elif goal_proprio.ndim == 1:
        goal_proprio = goal_proprio.unsqueeze(0)
    if goal_proprio.shape != (b, PROPRIO_DIM):
        raise ValueError(
            f"goal_proprio must be [{b}, {PROPRIO_DIM}], got {tuple(goal_proprio.shape)}"
        )
    out: Level1EncoderOutput = encoder(goal_image, goal_proprio)
    if out.visual.shape[1] != VISUAL_CHANNELS:
        raise ValueError(f"unexpected visual shape {tuple(out.visual.shape)}")
    return out.visual


class EvalObservation(NamedTuple):
    image: torch.Tensor  # [3, 98, 98] normalized
    proprio: torch.Tensor  # [2] normalized velocity
    xy: np.ndarray  # unnormalized (x, y) for metrics
    goal_image: torch.Tensor  # [3, 98, 98] normalized
    goal_xy: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


def _ensure_original_on_path(original_root: Path | None = None) -> Path:
    root = Path(original_root or DEFAULT_ORIGINAL_ROOT)
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def ensure_mujoco_runtime() -> None:
    """Set ``LD_LIBRARY_PATH`` so ``mujoco_py`` can import (headless-safe).

    Original HWM uses mujoco210. We do not modify the original package.
    """
    candidates = [
        Path(os.environ.get("MUJOCO_PY_MUJOCO_PATH", "")),
        Path.home() / ".mujoco/mujoco210",
        Path("/storage/home/hcoda1/2/anampally3/.mujoco/mujoco210"),
    ]
    root = next((p for p in candidates if p and (p / "bin").is_dir()), None)
    if root is None:
        return
    bin_dir = str(root / "bin")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if bin_dir not in current.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{bin_dir}:{current}" if current else bin_dir
    os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", str(root))
    os.environ.setdefault("MUJOCO_GL", "egl")


_ORIGINAL_ENV_IMPORT_ERROR: str | None = None


def original_env_importable(original_root: Path | None = None) -> bool:
    global _ORIGINAL_ENV_IMPORT_ERROR
    try:
        ensure_mujoco_runtime()
        _ensure_original_on_path(original_root)
        from pldm_envs.diverse_maze.evaluation.maze2d_envs_generator import (  # noqa: F401
            Maze2DEnvsGenerator,
        )
        _ORIGINAL_ENV_IMPORT_ERROR = None
        return True
    except Exception as exc:
        _ORIGINAL_ENV_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return False


def original_env_import_error() -> str | None:
    return _ORIGINAL_ENV_IMPORT_ERROR


def make_original_normalizer():
    """Original ``Normalizer`` instance populated with the same hard-set stats."""
    _ensure_original_on_path()
    from pldm_envs.utils.normalizer import STATS, Normalizer

    stats = STATS[ENV_NAME]
    return Normalizer(
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


def create_eval_env(
    *,
    difficulty: str = "medium",
    trial_index: int = 0,
    probe_dir: Path | None = None,
    original_root: Path | None = None,
    n_envs: int = 1,
):
    """One original Diverse Maze eval env at a stored start/target.

    Reuses ``Maze2DEnvsGenerator``. Does not regenerate layouts.
    ``n_envs`` defaults to 1 (M8 smoke). Do not pass 40.
    """
    ensure_mujoco_runtime()
    _ensure_original_on_path(original_root)
    from pldm_envs.diverse_maze.evaluation.maze2d_envs_generator import (
        Maze2DEnvsGenerator,
    )

    data_path = str(probe_dir or DEFAULT_PROBE_DIR)
    trials_path = str(start_target_path(difficulty, Path(data_path)))
    trials = load_starts_targets(trials_path)
    if trial_index >= len(trials):
        raise IndexError(trial_index)
    if n_envs < 1:
        raise ValueError(n_envs)
    if trial_index + n_envs > len(trials):
        raise ValueError("requested more envs than stored trials")

    # Generator always takes the first n_envs trials. For a non-zero index,
    # build a one-off trials dict.
    if trial_index != 0 or n_envs != 1:
        subset = {
            "starts": trials.starts[trial_index : trial_index + n_envs],
            "targets": trials.targets[trial_index : trial_index + n_envs],
            "map_layouts": trials.map_layouts[trial_index : trial_index + n_envs],
            "block_dists": trials.block_dists[trial_index : trial_index + n_envs],
            "turns": trials.turns[trial_index : trial_index + n_envs],
        }
        import tempfile
        import torch as _torch

        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        _torch.save(subset, tmp.name)
        trials_path = tmp.name

    gen = Maze2DEnvsGenerator(
        env_name=ENV_NAME,
        n_envs=n_envs,
        min_block_radius=1,
        max_block_radius=9999,
        action_repeat=ACTION_REPEAT,
        action_repeat_mode=ACTION_REPEAT_MODE,
        seed=42,
        stack_states=1,
        image_obs=True,
        data_path=data_path,
        trials_path=trials_path,
        unique_shortest_path=False,
        normalizer=make_original_normalizer(),
    )
    envs, _ = gen()
    return envs[0] if n_envs == 1 else envs


def _as_image_tensor(obs: Any) -> torch.Tensor:
    if isinstance(obs, dict):
        obs = obs["image"]
    if isinstance(obs, np.ndarray):
        t = torch.from_numpy(obs).float()
    else:
        t = torch.as_tensor(obs).float()
    if t.ndim == 3 and t.shape[-1] == IMAGE_CHANNELS:
        t = t.permute(2, 0, 1)
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    if t.shape != (IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"env image shape {tuple(t.shape)}")
    return t


@dataclass
class DiverseMazeEnvAdapter:
    """Thin adapter: original env in, planner-facing tensors out.

    Observation images from ``NormEvalWrapper`` are **already**
    ``normalize_state``'d. Planner actions:

    - ``step_normalized``: unnormalize with hard-set action stats, then
      ``env.step`` (raw physics action).
    - ``step_raw``: pass through (original MPPI already unnormalized).
    """

    env: Any
    normalizer: MazeNormalizer

    @classmethod
    def from_stored_trial(
        cls,
        *,
        difficulty: str = "medium",
        trial_index: int = 0,
        probe_dir: Path | None = None,
    ) -> "DiverseMazeEnvAdapter":
        env = create_eval_env(
            difficulty=difficulty, trial_index=trial_index, probe_dir=probe_dir
        )
        return cls(env=env, normalizer=default_normalizer())

    def _read(self, image: Any, reward: float = 0.0, done: bool = False, info: dict | None = None) -> EvalObservation:
        image_t = _as_image_tensor(image)
        proprio = torch.as_tensor(self.env.get_proprio_vel(normalized=True)).float().reshape(-1)[:2]
        xy = np.asarray(self.env.get_pos(), dtype=np.float64).reshape(-1)[:2]
        goal_image = _as_image_tensor(self.env.get_target_obs())
        goal_xy = np.asarray(self.env.get_target()[:2], dtype=np.float64)
        info = dict(info or {})
        info["success_distance"] = success_from_distance(xy, goal_xy)
        return EvalObservation(
            image=image_t,
            proprio=proprio,
            xy=xy,
            goal_image=goal_image,
            goal_xy=goal_xy,
            reward=float(reward),
            done=bool(done),
            info=info,
        )

    def reset(self) -> EvalObservation:
        obs = self.env.reset()
        return self._read(obs, reward=0.0, done=False, info=self.env.get_info())

    def step_raw(self, action: np.ndarray | torch.Tensor) -> EvalObservation:
        a = np.asarray(
            action.detach().cpu() if isinstance(action, torch.Tensor) else action,
            dtype=np.float32,
        ).reshape(-1)
        result = self.env.step(a)
        if len(result) == 5:
            obs, rew, done, _trunc, info = result
        else:
            obs, rew, done, info = result
        return self._read(obs, reward=float(rew), done=bool(done), info=info)

    def step_normalized(self, action: np.ndarray | torch.Tensor) -> EvalObservation:
        t = torch.as_tensor(action, dtype=torch.float32).reshape(-1, 2)
        raw = self.normalizer.unnormalize_action(t).reshape(-1)
        return self.step_raw(raw)

    def encode_current(self, encoder: Level1Encoder, obs: EvalObservation | None = None) -> Level1EncoderOutput:
        if obs is None:
            obs = self._read(self.env.get_obs(), info=self.env.get_info())
        return encoder(obs.image.unsqueeze(0), obs.proprio.unsqueeze(0))

    def encode_goal(self, encoder: Level1Encoder, obs: EvalObservation | None = None) -> torch.Tensor:
        if obs is None:
            obs = self._read(self.env.get_obs(), info=self.env.get_info())
        zeros = torch.zeros(1, PROPRIO_DIM, dtype=obs.goal_image.dtype)
        return encode_goal(encoder, obs.goal_image, zeros)

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()
