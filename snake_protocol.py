# Shared Configuration
MAP_WIDTH = 50
MAP_HEIGHT = 50
CELL_SIZE = 20

# Game Settings
ROOM_COUNT = 20
ROOM_CAPACITY = 10
SERVER_MAX_PLAYERS = ROOM_COUNT * ROOM_CAPACITY
SIM_TICK_HZ = 15
TICK_DT_MS = 1000 / SIM_TICK_HZ

# Protocol Opcodes / Types
MSG_JOIN = "join"
MSG_JOIN_OK = "join_ok"
MSG_INPUT = "in"
MSG_GAME_START = "game_start"
MSG_DELTA = "d"
MSG_GAME_OVER = "game_over"
MSG_ERROR = "err"
MSG_EXIT = "exit"

# Room Stats API
MSG_ROOM_STATS_REQ = "room_stats_req"
MSG_ROOM_STATS = "room_stats"
