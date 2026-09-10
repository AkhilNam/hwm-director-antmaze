"""Released Diverse Maze Level-1 objective coefficients.

Source: ``pldm/configs/diverse_maze/icml/large_diverse_25maps.yaml``
at original SHA ``e197375``. Fields present in YAML but **not** in
``objectives_l1.objectives`` (inactive): ``probe``.

Temporal VICReg terms with coefficient 0 are not applied
(``sim_coeff_t``, ``cov_coeff_t``). ``std_coeff_t`` is active.
"""

from __future__ import annotations

# VICRegObs (VICRegObjectiveConfig vicreg_obs, pred_attr="obs")
VICREG_PROJECTOR = "id"
VICREG_RANDOM_PROJECTOR = False
VICREG_SIM_COEFF = 1.0
VICREG_STD_COEFF = 29.409481669124336
VICREG_COV_COEFF = 17.8664279184067
VICREG_STD_COEFF_T = 2.9199
VICREG_COV_COEFF_T = 0.0
VICREG_SIM_COEFF_T = 0.0
VICREG_COV_PER_FEATURE = False
VICREG_ADJUST_COV = True
VICREG_STD_MARGIN = 1.0
VICREG_STD_MARGIN_T = 1.0
VICREG_VAR_EPS = 0.0001

# IDMObjectiveConfig
IDM_COEFF = 4.810550706458433
IDM_ARCH = "conv"
IDM_ARCH_SUBCLASS = "a"
IDM_USE_PRED = False
IDM_ACTION_DIM = 2

# PredictionObjectiveConfig prediction_proprio
PRED_PROPRIO_COEFF = 2.416154262252218
