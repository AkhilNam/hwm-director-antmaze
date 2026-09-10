#!/usr/bin/env python3
"""Count original ConvPredictor f_H params (read-only original repo). CPU only."""

from __future__ import annotations

import sys

ORIGINAL = "/storage/scratch1/2/anampally3/hwm-original-repro"
sys.path.insert(0, ORIGINAL)

from pldm.models.predictors.conv_predictors import ConvPredictor
from pldm.models.predictors.enums import PredictorConfig


def nparams(module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    config = PredictorConfig(
        predictor_arch="conv2",
        predictor_subclass="l2_d4rl_e_p",
        residual=True,
        action_encoder_arch="8-64-8",
        z_dim=8,
        z_stochastic=False,
        z_discrete=False,
        predictor_ln=True,
        tie_backbone_ln=False,
        posterior_arch="32-32",
        posterior_input_type="actions",
        posterior_input_dim=20,
        prior_arch="",
    )
    pred = ConvPredictor(
        config=config,
        repr_dim=(18, 43, 43),
        action_dim=8,
        pred_proprio_dim=(2, 43, 43),
        pred_obs_dim=(16, 43, 43),
    )
    print("original conv layers params     ", nparams(pred.layers))
    print("original action encoder params  ", nparams(pred.action_encoder))
    print("original f_H (layers+enc)       ", nparams(pred.layers) + nparams(pred.action_encoder))
    print("original unused final_ln params ", nparams(pred.final_ln))
    print("original posterior params       ", nparams(pred.posterior_model) if pred.posterior_model is not None else 0)
    first = pred.layers[0]
    print("original first conv in_channels ", first.in_channels)
    print("original first conv out_channels", first.out_channels)
    last = [m for m in pred.layers.modules() if m.__class__.__name__ == "Conv2d"][-1]
    print("original last conv out_channels ", last.out_channels)


if __name__ == "__main__":
    main()
