"""Level-1 observation and predictor tensor shapes for faithful HWM.

These are the Diverse Maze L1 sizes traced from original HWM
(``MeNet6`` / ``d4rl_a`` on 98x98 RGB plus 2D proprio velocity, and
``ConvPredictor`` / ``d4rl_b_p`` on fused 18-channel maps plus 2D action).
"""

IMAGE_CHANNELS = 3
IMAGE_SIZE = 98
VISUAL_CHANNELS = 16
FEATURE_SIZE = 43
PROPRIO_DIM = 2
FUSED_CHANNELS = VISUAL_CHANNELS + PROPRIO_DIM  # 18
ACTION_DIM = 2
PREDICTOR_IN_CHANNELS = FUSED_CHANNELS + ACTION_DIM  # 20: fused h + expanded action
