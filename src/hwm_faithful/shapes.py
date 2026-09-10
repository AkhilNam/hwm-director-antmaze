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

# Level-2 temporal abstraction (original ``l2_step_skip``, ``l2_n_steps``).
L2_STEP_SKIP = 10
L2_N_STEPS = 6  # number of L2 *transitions*
L2_N_STATES = L2_N_STEPS + 1  # 7 states: t = 0,10,...,60
L2_PRIMITIVE_HORIZON = L2_N_STEPS * L2_STEP_SKIP  # 60
L2_L1_WINDOW = L2_PRIMITIVE_HORIZON + 1  # 61
Z_DIM = 8
POSTERIOR_INPUT_DIM = ACTION_DIM * L2_STEP_SKIP  # 20
L2_ACTION_ENCODER_CHANNELS = Z_DIM  # after 8-64-8, before spatial expand
L2_PREDICTOR_IN_CHANNELS = FUSED_CHANNELS + L2_ACTION_ENCODER_CHANNELS  # 26: H + z_map


