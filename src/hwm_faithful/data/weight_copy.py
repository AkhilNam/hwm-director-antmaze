"""One-off original→ours parameter copy for architecture parity tests.

Not used for training. Do not bundle original checkpoints as the faithful
model. Mapping is by module *type order* (Conv2d, GroupNorm, Linear,
LayerNorm), not by parameter names.
"""

from __future__ import annotations

import sys
from pathlib import Path
import torch
from torch import nn

from hwm_faithful.data.diverse_maze import DEFAULT_ORIGINAL_ROOT
from hwm_faithful.level2.action_encoder import Level2ActionEncoder
from hwm_faithful.level2.posterior import Level2ActionPosterior
from hwm_faithful.level2.predictor import Level2Predictor
from hwm_faithful.models.conv_predictor import Level1Predictor
from hwm_faithful.models.level1_encoder import Level1Encoder
from hwm_faithful.shapes import FEATURE_SIZE, FUSED_CHANNELS, IMAGE_SIZE, VISUAL_CHANNELS, Z_DIM


AFFINE_TYPES = (nn.Conv2d, nn.GroupNorm, nn.Linear, nn.LayerNorm)


def ensure_original_path(root: Path | None = None) -> Path:
    path = Path(root or DEFAULT_ORIGINAL_ROOT)
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)
    return path


def original_models_importable() -> bool:
    try:
        ensure_original_path()
        from pldm.models.encoders.encoders import MeNet6  # noqa: F401
        from pldm.models.predictors.conv_predictors import ConvPredictor  # noqa: F401
        from pldm.models.misc import PosteriorContinuous  # noqa: F401
        return True
    except Exception:
        return False


def iter_affine_modules(module: nn.Module) -> list[nn.Module]:
    return [m for m in module.modules() if isinstance(m, AFFINE_TYPES)]


def copy_affine_modules(src: nn.Module, dst: nn.Module) -> int:
    """Copy matching Conv/GN/Linear/LN parameters in traversal order."""
    src_ms = iter_affine_modules(src)
    dst_ms = iter_affine_modules(dst)
    if len(src_ms) != len(dst_ms):
        raise ValueError(
            f"affine module count {len(src_ms)} != {len(dst_ms)} "
            f"src={[type(m).__name__ for m in src_ms]} "
            f"dst={[type(m).__name__ for m in dst_ms]}"
        )
    n = 0
    with torch.no_grad():
        for a, b in zip(src_ms, dst_ms):
            if type(a) is not type(b):
                raise TypeError(f"{type(a)} vs {type(b)}")
            sa, sb = a.state_dict(), b.state_dict()
            if sa.keys() != sb.keys():
                raise KeyError(f"{sa.keys()} vs {sb.keys()}")
            for k in sa:
                if sa[k].shape != sb[k].shape:
                    raise ValueError(f"{k} shape {tuple(sa[k].shape)} vs {tuple(sb[k].shape)}")
            b.load_state_dict(sa)
            n += sum(p.numel() for p in b.parameters())
    return n


def build_original_l1_encoder():
    ensure_original_path()
    from pldm.models.encoders.encoders import MeNet6
    from pldm.models.encoders.enums import BackboneConfig, ProprioConfig

    cfg = BackboneConfig(
        arch="menet6",
        backbone_subclass="d4rl_a",
        channels=3,
        late_proprio_cfg=ProprioConfig(
            ignore=False, encoder_arch="id_expand", fuse=True
        ),
    )
    return MeNet6(cfg, input_dim=[3, IMAGE_SIZE, IMAGE_SIZE], input_proprio_dim=2)


def build_original_l1_predictor():
    ensure_original_path()
    from pldm.models.predictors.conv_predictors import ConvPredictor
    from pldm.models.predictors.enums import PredictorConfig

    cfg = PredictorConfig(
        predictor_arch="conv2",
        predictor_subclass="d4rl_b_p",
        residual=True,
        action_encoder_arch="id",
        z_dim=0,
        z_stochastic=False,
    )
    return ConvPredictor(
        config=cfg,
        repr_dim=(FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
        action_dim=2,
        pred_proprio_dim=2,
        pred_obs_dim=(VISUAL_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
    )


def build_original_posterior():
    ensure_original_path()
    from pldm.models.misc import PosteriorContinuous

    return PosteriorContinuous(
        input_dim=20,
        arch="32-32",
        z_dim=Z_DIM,
        min_std=0.05,
        posterior_input_type="actions",
    )


def build_original_l2_predictor():
    ensure_original_path()
    from pldm.models.predictors.conv_predictors import ConvPredictor
    from pldm.models.predictors.enums import PredictorConfig

    cfg = PredictorConfig(
        predictor_arch="conv2",
        predictor_subclass="l2_d4rl_e_p",
        residual=True,
        action_encoder_arch="8-64-8",
        # z_dim=0 avoids SequencePredictor constructing PriorContinuous
        # with a spatial repr_dim (that path is unused for this conv copy).
        z_dim=0,
        z_stochastic=False,
    )
    return ConvPredictor(
        config=cfg,
        repr_dim=(FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
        action_dim=Z_DIM,
        pred_proprio_dim=2,
        pred_obs_dim=(VISUAL_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
    )


def copy_encoder_weights(src, dst: Level1Encoder) -> int:
    return copy_affine_modules(src.layers, dst.visual_encoder)


def copy_l1_predictor_weights(src, dst: Level1Predictor) -> int:
    return copy_affine_modules(src.layers, dst)


def copy_posterior_weights(src, dst: Level2ActionPosterior) -> int:
    # Original MLP wraps Sequential; LayerNorm is sibling mu_ln.
    n = copy_affine_modules(src.posterior_net, dst.posterior_net)
    n += copy_affine_modules(src.mu_ln, dst.mu_ln)
    return n


def copy_l2_action_encoder_weights(src, dst: Level2ActionEncoder) -> int:
    # Original: Sequential(Linear, ReLU, Linear, Expander2D)
    return copy_affine_modules(src.action_encoder, dst.mlp)


def copy_l2_predictor_conv_weights(src, dst: Level2Predictor) -> int:
    return copy_affine_modules(src.layers, dst.blocks)
