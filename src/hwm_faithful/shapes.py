"""Level-1 observation tensor shapes for the faithful HWM encoder.

These are the Diverse Maze L1 sizes traced from original HWM
(``MeNet6`` / ``d4rl_a`` on 98x98 RGB plus 2D proprio velocity).
"""

IMAGE_CHANNELS = 3
IMAGE_SIZE = 98
VISUAL_CHANNELS = 16
FEATURE_SIZE = 43
PROPRIO_DIM = 2
FUSED_CHANNELS = VISUAL_CHANNELS + PROPRIO_DIM  # 18
