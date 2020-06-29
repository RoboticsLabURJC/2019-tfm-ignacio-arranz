# Global variables

from environs import Env

env = Env()
env.read_env()

debug_level = env.int("DEBUG_LEVEL")
telemetry = env.bool("TELEMETRY", True)
my_board = env.bool("MY_BOARD", False)
save_model = env.bool("SAVE_MODEL", False)
load_model = env.bool("LOAD_MODEL", False)

# === ACTIONS SET ===
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

# POSES
gazebo_positions = [(0, 53.462, -41.988, 0.004, 0, 0, 1.57, -1.57),
             (1, 53.462, -8.734, 0.004, 0, 0, 1.57, -1.57),
             (2, 39.712, -30.741, 0.004, 0, 0, 1.56, 1.56),
             (3, -7.894, -39.051, 0.004, 0, 0.01, -2.021, 2.021),
             (4, 20.043, 37.130, 0.003, 0, 0.103, -1.4383, -1.4383)]