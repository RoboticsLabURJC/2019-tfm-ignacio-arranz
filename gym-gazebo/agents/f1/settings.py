# Global variables

from environs import Env

env = Env()
env.read_env()

debug_level = env("DEBUG_LEVEL", 'DEBUG')
telemetry = env("TELEMETRY", False)
my_board = env("MY_BOARD", False)