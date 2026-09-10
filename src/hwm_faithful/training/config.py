"""Active Level-1 training settings from ``large_diverse_25maps.yaml``.

Traced from the released YAML + executed Python at SHA ``e197375``
(``pldm/train.py``, ``optimizer_factory.py``, ``schedulers.py``,
``pldm/data/utils.py``, ``D4RLDataset``). Dead YAML fields are listed
below the dataclass and are **not** used by this trainer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hwm_faithful.losses.coefficients import (
    IDM_ARCH,
    IDM_ARCH_SUBCLASS,
    IDM_COEFF,
    IDM_USE_PRED,
    PRED_PROPRIO_COEFF,
    VICREG_ADJUST_COV,
    VICREG_COV_COEFF,
    VICREG_SIM_COEFF,
    VICREG_STD_COEFF,
    VICREG_STD_COEFF_T,
)
from hwm_faithful.shapes import ACTION_DIM
from hwm_faithful.data.diverse_maze import L1_N_STEPS


# YAML ``base_lr``. Scheduler then scales by ``batch_size / 256``.
YAML_BASE_LR = 0.017632900482959527
ADAM_WEIGHT_DECAY = 1e-6
LR_BATCH_REF = 256
WARMUP_FRACTION = 0.10
COSINE_END_LR_RATIO = 0.001
EXPECTED_25MAP_FRAMES = 4_540_859


@dataclass
class Level1TrainConfig:
    """Active L1 training config. Paths come from CLI / sbatch, not here."""

    # DATA (active)
    n_steps: int = L1_N_STEPS
    batch_size: int = 128
    normalize: bool = True
    normalizer_hardset: bool = True
    shuffle: bool = True
    drop_last: bool = True
    num_workers: int = 10
    prefetch_factor: int = 4
    pin_memory: bool = True
    prioritized: bool = False
    l1_only: bool = True  # original L1 YAML: D4RL ``l2_n_steps`` default 0

    # MODEL (active)
    encoder_arch: str = "menet6"
    encoder_subclass: str = "d4rl_a"
    late_proprio_arch: str = "id_expand"
    late_proprio_fuse: bool = True
    predictor_arch: str = "conv2"
    predictor_subclass: str = "d4rl_b_p"
    predictor_residual: bool = True
    action_encoder_arch: str = "id"
    z_dim: int = 0
    action_dim: int = ACTION_DIM
    compile_model: bool = False

    # OBJECTIVES (active; coefficients from YAML, not retuned)
    objectives: tuple[str, ...] = ("VICRegObs", "IDM", "PredictionProprio")
    vicreg_sim_coeff: float = VICREG_SIM_COEFF
    vicreg_std_coeff: float = VICREG_STD_COEFF
    vicreg_cov_coeff: float = VICREG_COV_COEFF
    vicreg_std_coeff_t: float = VICREG_STD_COEFF_T
    vicreg_adjust_cov: bool = VICREG_ADJUST_COV
    idm_coeff: float = IDM_COEFF
    idm_arch: str = IDM_ARCH
    idm_arch_subclass: str = IDM_ARCH_SUBCLASS
    idm_use_pred: bool = IDM_USE_PRED
    prediction_proprio_coeff: float = PRED_PROPRIO_COEFF

    # OPTIMIZER / SCHEDULER (active)
    optimizer_type: str = "Adam"
    base_lr: float = YAML_BASE_LR
    weight_decay: float = ADAM_WEIGHT_DECAY
    # YAML omits ``optimizer_schedule`` → TrainConfig default Cosine.
    optimizer_schedule: str = "cosine"
    include_idm_in_optimizer: bool = True
    grad_clip: float | None = None  # original train.py does not clip
    mixed_precision: bool = False  # original has no AMP

    # TRAINING (active)
    epochs: int = 3
    # Original loop is ``range(epoch, epochs + 1)`` → 4 passes when epochs=3.
    epoch_end_inclusive: bool = True
    seed: int = 246
    save_every_n_epochs: int = 1
    log_every: int = 100
    resume_if_possible: bool = True
    deterministic: bool = True

    # Runtime / CLI (not YAML)
    data_path: str | None = None
    images_path: str | None = None
    output_dir: str | None = None
    resume_path: str | None = None
    device: str = "cuda"
    max_steps: int | None = None
    max_windows: int | None = None
    max_epochs: int | None = None
    require_25maps: bool = False
    expected_frames: int = EXPECTED_25MAP_FRAMES

    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def scaled_base_lr(self) -> float:
        return self.base_lr * self.batch_size / LR_BATCH_REF

    @property
    def n_epoch_passes(self) -> int:
        n = self.epochs if self.max_epochs is None else int(self.max_epochs)
        return n + 1 if self.epoch_end_inclusive else n

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["scaled_base_lr"] = self.scaled_base_lr
        d["n_epoch_passes"] = self.n_epoch_passes
        return d


DEAD_YAML_FIELDS: dict[str, str] = {
    "val_fraction: 0.2": "D4RL factory sets val_ds=None; unused for L1.",
    "backbone_width_factor: 2": "No Python code reads this; d4rl_a channels win.",
    "level1.input_dim: 4": "Trainer infers input_dim from image tensor [3,98,98].",
    "objectives_l1.probe": "Not listed in objectives_l1.objectives.",
    "vicreg sim_coeff_t / cov_coeff_t = 0": "Inactive temporal VICReg terms.",
    "wandb: true": "Disabled for our jobs; not part of the model.",
    "eval_* / eval_cfg": "Evaluation YAML; not L1 training.",
    "hjepa.step_skip: 4": "Unused by L1 maze training.",
    "compile_model default True": "YAML overrides to False; we keep False.",
}


def overfit_config(**overrides: Any) -> Level1TrainConfig:
    """Tiny real-subset overfit: Constant LR so step 0 is not a no-op."""
    cfg = Level1TrainConfig(
        batch_size=4,
        num_workers=0,
        prefetch_factor=4,
        shuffle=False,
        drop_last=False,
        optimizer_schedule="constant",
        epochs=40,
        epoch_end_inclusive=False,
        max_windows=8,
        max_steps=40,
        log_every=1,
        save_every_n_epochs=1,
        pin_memory=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def smoke_config(**overrides: Any) -> Level1TrainConfig:
    cfg = Level1TrainConfig(
        num_workers=4,
        max_steps=300,
        log_every=10,
        epochs=1,
        epoch_end_inclusive=False,
        save_every_n_epochs=1,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def resolve_device(name: str) -> str:
    import torch

    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name


def output_path(cfg: Level1TrainConfig) -> Path:
    if not cfg.output_dir:
        raise ValueError("output_dir is required")
    return Path(cfg.output_dir)
