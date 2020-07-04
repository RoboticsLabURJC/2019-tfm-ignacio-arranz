# Global variables

import numpy as np

from environs import Env


env = Env()
env.read_env()

debug_level = env.int("DEBUG_LEVEL")
telemetry = env.bool("TELEMETRY", True)
my_board = env.bool("MY_BOARD", False)
save_model = env.bool("SAVE_MODEL", False)
load_model = env.bool("LOAD_MODEL", False)

# === ACTIONS SET ===
# Deprecated?
space_reward = np.flip(np.linspace(0, 1, 300))

# action: (lineal, angular)
actions_simple = {
    0: (3, 0),
    1: (3, 1),
    2: (3, -1)
}

actions_medium = {
    0: (3, 0),
    1: (6, 0),
    2: (3, 1),
    3: (3, -1),
    4: (4, 4),
    5: (4, -4),
}

actions_hard = {
    0: (3, 0),
    1: (6, 0),
    2: (3, 1),
    3: (3, -1),
    4: (4, 4),
    5: (4, -4),
    6: (2, 5),
    7: (2, -5),
}

# === POSES ===
gazebo_positions = [(0, 53.462, -41.988, 0.004, 0, 0,      1.57,   -1.57),
                    (1, 53.462, -8.734,  0.004, 0, 0,      1.57,   -1.57),
                    (2, 39.712, -30.741, 0.004, 0, 0,      1.56,    1.56),
                    (3, -6.861, -36.481, 0.004, 0, 0.01,   -0.858, 0.613),
                    (4, 20.043,  37.130, 0.003, 0, 0.103, -1.4383, -1.4383)]

# === CAMERA ===
# Images size
witdh = 640
center_image = witdh/2

# Coord X ROW
x_row = [260, 360, 450]
# Maximum distance from the line
ranges = [300, 280, 250]  # Line 1, 2 and 3
reset_range = [-40, 40]
last_center_line = 0
