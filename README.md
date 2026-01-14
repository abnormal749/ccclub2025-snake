# Snake Connect (Quick README)

Realtime multiplayer Snake with a Python server and pygame client. One AI bot is auto-added to every room.

## Quick Start

1) Install deps (Python 3.9+):
   `pip install pygame pygame-menu websockets torch numpy`
2) Start server:
   `python snake_server.py`
3) Start client:
   `python snake_gui.py`

## How To Play

- Pick name, room (1-20), and server IP in the menu.
- Move with `WASD` or arrow keys.
- Host can start with `Space` or the START button.

## Notes

- Map size: 50x50. Rooms: 20. Capacity: 5.
- Server sim tick: 20Hz. Broadcast: 40Hz.
- Food: up to 3 items, server-spawned.
- Mid-game join is allowed as spectator (dead).

## Stress Test (Optional)

`python stress_test.py --count 50 --uri ws://127.0.0.1:8765`
