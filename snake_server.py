import asyncio
import json
import random
import time
import uuid
import numpy as np
import torch
from collections import deque
import websockets
from snake_protocol import *
from snake_agent import Linear_QNet

# --- Game Engine & Data Models ---

class PlayerState:
    def __init__(self, player_id, username, websocket):
        self.player_id = player_id
        self.username = username
        self.websocket = websocket # None for Bots
        self.connected = True
        self.alive = True
        self.score = 0
        self.direction = (1, 0) # default right
        self.body = deque()
        self.body_set = set()
        self.last_input_ts = time.time()
        self.last_seen_ts = time.time()
        self.pending_direction = None
        self.is_bot = False

class BotPlayer(PlayerState):
    def __init__(self, player_id, model):
        super().__init__(player_id, "AI", None)
        self.is_bot = True
        self.model = model
        self.current_step = 0

    def get_move(self, room):
        # Calculate state vector for AI
        head = self.body[0] if self.body else (0,0)
        w, h = MAP_WIDTH, MAP_HEIGHT
        
        # Helper: Check if point is dangerous (wall or body)
        def is_danger(pt):
            x, y = pt
            if not (0 <= x < w and 0 <= y < h): return True
            return (x, y) in room.occupied_set

        # Current direction
        dx, dy = self.direction
        
        # Clockwise directions: [Right, Down, Left, Up]
        clock_wise = [(1,0), (0,1), (-1,0), (0,-1)] # R, D, L, U
        try:
            idx = clock_wise.index((dx, dy))
        except:
            idx = 0

        # Points
        hx, hy = head
        
        # Rebuild the 20-dim state used by the GUI/agent:
        # 1) danger body (R,L,U,D), 2) danger wall (R,L,U,D),
        # 3) ray body (L,R,U,D), 4) direction one-hot (L,R,U,D),
        # 5) food relative (L,R,U,D)
        
        pt_l = (hx - 1, hy)
        pt_r = (hx + 1, hy)
        pt_u = (hx, hy - 1)
        pt_d = (hx, hy + 1)
        
        # Ray casting to detect body across the map
        def check_ray(points):
            for p in points:
                if p in room.occupied_set and p != head:
                    return True
            return False

        pts_l = [(i, hy) for i in range(0, hx)]
        pts_r = [(i, hy) for i in range(hx + 1, MAP_WIDTH)]
        pts_u = [(hx, i) for i in range(0, hy)]
        pts_d = [(hx, i) for i in range(hy + 1, MAP_HEIGHT)]

        # Find closest food
        fx, fy = 0, 0
        if room.food:
             # Manhattan distance
             closest = min(room.food, key=lambda f: abs(f[0]-hx) + abs(f[1]-hy))
             fx, fy = closest

        # Construct 20-dim state vector
        state = [
             # 1. Danger Body [R, L, U, D]
             int((pt_r) in room.occupied_set), int((pt_l) in room.occupied_set), int((pt_u) in room.occupied_set), int((pt_d) in room.occupied_set),
             
             # 2. Danger Wall [R, L, U, D]
             int(not (0 <= pt_r[0] < w and 0 <= pt_r[1] < h)), 
             int(not (0 <= pt_l[0] < w and 0 <= pt_l[1] < h)),
             int(not (0 <= pt_u[0] < w and 0 <= pt_u[1] < h)),
             int(not (0 <= pt_d[0] < w and 0 <= pt_d[1] < h)),
             
             # 3. Ray Body [L, R, U, D]
             int(check_ray(pts_l)), int(check_ray(pts_r)), int(check_ray(pts_u)), int(check_ray(pts_d)),
             
             # 4. Direction [L, R, U, D]
             int(self.direction == (-1, 0)), int(self.direction == (1, 0)), int(self.direction == (0, -1)), int(self.direction == (0, 1)),
             
             # 5. Food Relative Pos [L, R, U, D]
             int(fx < hx), int(fx > hx), int(fy < hy), int(fy > hy)
        ]
        
        state_t = torch.tensor(np.array(state, dtype=int), dtype=torch.float)
        
        with torch.no_grad():
             prediction = self.model(state_t)
        
        move_idx = torch.argmax(prediction).item()
        
        # 0: Straight, 1: Right turn, 2: Left turn
        if move_idx == 0:
            new_dir = clock_wise[idx]
        elif move_idx == 1:
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]
            
        self.direction = new_dir

class Room:
    def __init__(self, room_id):
        self.room_id = room_id
        self.capacity = ROOM_CAPACITY
        self.status = "IDLE" # IDLE, WAITING, RUNNING, FINISHED
        self.players = {} # player_id -> PlayerState
        self.host_id = None
        
        # Game State
        self.food = []
        self.occupied_set = set() # All snake bodies
        self.tick_id = 0
        self.start_time = 0
        self.death_order = []
        
        # Auto-start logic
        self.countdown_deadline = None
        self.pending_deaths = set()

    async def _safe_send(self, ws, msg):
        try:
            await ws.send(msg)
        except Exception:
            pass

    def broadcast(self, message):
        for p in self.players.values():
            if p.connected and p.websocket:
                asyncio.create_task(self._safe_send(p.websocket, json.dumps(message)))

    def add_player(self, player):
        if len(self.players) >= self.capacity:
            return False, "ROOM_FULL"
        
        # Spectator logic: If running, they join as dead
        if self.status == "RUNNING":
            player.alive = False
        
        self.players[player.player_id] = player
        
        # First player becomes host
        if self.host_id is None:
            self.host_id = player.player_id
            
        if self.status == "IDLE":
            self.status = "WAITING"
            
        return True, None

    def remove_player(self, player_id):
        if player_id in self.players:
            p = self.players[player_id]
            p.connected = False
            
            if self.status == "WAITING":
                del self.players[player_id]
                if self.host_id == player_id:
                    self.host_id = next(iter(self.players)) if self.players else None
                if not self.players:
                    self.status = "IDLE"
            elif self.status == "RUNNING":
                p.alive = False
                self.pending_deaths.add(player_id)
                # Cleanup handled in game loop

        # Check if any humans left connected
        humans_active = [p for p in self.players.values() if not getattr(p, 'is_bot', False) and p.connected]
        if not humans_active:
             print(f"Room {self.room_id}: Last human left. Resetting AI scores.")
             for p in self.players.values():
                 if getattr(p, 'is_bot', False):
                     p.score = 0
                
    def spawn_food(self):
        # Maintain 3 foods
        attempts = 0
        while len(self.food) < 3 and attempts < 100:
            attempts += 1
            x = random.randint(0, MAP_WIDTH - 1)
            y = random.randint(0, MAP_HEIGHT - 1)
            if (x, y) not in self.occupied_set and (x, y) not in self.food:
                self.food.append((x, y))

    def start_game(self, reason):
        print(f"Room {self.room_id} starting: {reason}")
        self.status = "RUNNING"
        self.tick_id = 0
        self.start_time = time.time()
        self.death_order = []
        self.occupied_set = set()
        self.pending_deaths.clear()
        
        # Prune disconnected players
        to_remove = [pid for pid, p in self.players.items() if not p.connected and not getattr(p, 'is_bot', False)]
        for pid in to_remove:
            print(f"Pruning disconnected player {pid} from Room {self.room_id}")
            del self.players[pid]
        
        # Initialize Snakes
        spawn_info = []
        for p in self.players.values():
            p.alive = True
            p.body.clear()
            p.body_set.clear()
            
            # Find spawn spot
            found = False
            for _ in range(100):
                sx = random.randint(2, MAP_WIDTH - 3)
                sy = random.randint(2, MAP_HEIGHT - 3)
                
                # Check collision with others
                collides = False
                for other in self.players.values():
                    if (sx, sy) in other.body_set:
                        collides = True
                if not collides:
                    start_body = [(sx, sy), (sx-1, sy), (sx-2, sy)]
                    p.body = deque(start_body)
                    p.body_set = set(start_body)
                    p.direction = (1, 0) # Right
                    self.occupied_set.update(start_body)
                    found = True
                    spawn_info.append({
                        "id": p.player_id, 
                        "name": p.username,
                        "body": start_body
                    })
                    break
            
            if not found:
                p.alive = False # Could not spawn
        
        self.food = []
        self.spawn_food()
        
        self.broadcast({
            "t": MSG_GAME_START,
            "tick_id": 0,
            "food": self.food,
            "players": spawn_info
        })

    def step(self):
        if self.status != "RUNNING":
            return
            
        self.tick_id += 1
        moves = []
        alive_players = [p for p in self.players.values() if p.alive]
        
        if not alive_players:
            self.end_game()
            return
        
        # Bot Logic
        for p in alive_players:
            if hasattr(p, 'is_bot') and p.is_bot:
                p.get_move(self)
                
        # Check counts for Game Over logic
        human_alive_count = sum(1 for p in alive_players if not getattr(p, 'is_bot', False))
        human_total_count = sum(1 for p in self.players.values() if not getattr(p, 'is_bot', False))
        
        # Game Over Condition: 
        # 1. If humans connected: Wait until ALL humans dead.
        # 2. If bots only: Wait until <= 1 survivor.
        if human_total_count > 0:
            if human_alive_count == 0:
                self.end_game()
                return
        else:
            if len(alive_players) <= 1 and len(self.players) >= 2:
                self.end_game()
                return

        # Phase 1: Calculate Intent
        snake_intents = {} # pid -> {next_head, will_grow, tail_to_free}
        
        for p in alive_players:
            dx, dy = p.direction
            hx, hy = p.body[0]
            nx, ny = hx + dx, hy + dy
            
            will_grow = (nx, ny) in self.food
            tail_to_free = p.body[-1] if not will_grow else None
            
            snake_intents[p.player_id] = {
                "next_head": (nx, ny),
                "will_grow": will_grow,
                "tail_to_free": tail_to_free
            }
            
        # Phase 2: Decide Deaths
        tails_to_free = set()
        for info in snake_intents.values():
            if info["tail_to_free"]:
                tails_to_free.add(info["tail_to_free"])
                
        dying_ids = set()
        dying_ids.update(self.pending_deaths)
        self.pending_deaths.clear()
        
        for p in alive_players:
            intent = snake_intents[p.player_id]
            nx, ny = intent["next_head"]
            
            # Wall
            if not (0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT):
                dying_ids.add(p.player_id)
                continue
                
            # Body collision
            if (nx, ny) in self.occupied_set:
                if (nx, ny) not in tails_to_free:
                    dying_ids.add(p.player_id)
                    continue
            
            # Head-to-Head
            for other in alive_players:
                if other.player_id == p.player_id: continue
                other_intent = snake_intents[other.player_id]
                if other_intent["next_head"] == (nx, ny):
                    dying_ids.add(p.player_id)
                    dying_ids.add(other.player_id)
        
        food_eaten = False
        
        # Phase 3: Apply Moves
        for p in alive_players:
            if p.player_id in dying_ids:
                continue
            
            intent = snake_intents[p.player_id]
            nx, ny = intent["next_head"]
            
            p.body.appendleft((nx, ny))
            p.body_set.add((nx, ny))
            self.occupied_set.add((nx, ny))
            
            head_add = (nx, ny)
            tail_remove = None
            
            if not intent["will_grow"]:
                tx, ty = p.body.pop()
                # Handled Chasing Tail case:
                # If head matches old tail pos, do NOT remove from set.
                if (tx, ty) != (nx, ny):
                    if (tx, ty) in p.body_set:
                        p.body_set.remove((tx, ty))
                    tail_remove = (tx, ty)
                else:
                    tail_remove = None
            
            if tail_remove:
                 self.occupied_set.discard(tail_remove)
            else:
                p.score += 1
                food_eaten = True
                if (nx, ny) in self.food:
                    self.food.remove((nx, ny))
                
            moves.append({
                "id": p.player_id,
                "head_add": head_add,
                "tail_remove": tail_remove,
                "score": p.score,
                "alive": True
            })

        # Phase 4: Cleanup Deaths
        for pid in dying_ids:
            if pid not in self.players:
                continue
            p = self.players[pid]
            p.alive = False
            p.score = max(0, p.score // 2)
            self.death_order.append(pid)
            
            for cell in p.body:
                 self.occupied_set.discard(cell)
            p.body.clear()
            p.body_set.clear()
            
            moves.append({
                "id": pid,
                "dead": True,
                "reason": "collision"
            })
            
        if food_eaten:
            self.spawn_food()
            
        self.broadcast({
            "t": MSG_DELTA,
            "tick": self.tick_id,
            "moves": moves,
            "food": self.food
        })

    def end_game(self):
        self.status = "FINISHED"
        
        # Ranking
        alive = [p for p in self.players.values() if p.alive]
        dead = [self.players[pid] for pid in reversed(self.death_order) if pid in self.players]
        
        rank = 1
        ranks = []
        winner_id = None
        
        for p in alive:
            winner_id = p.player_id
            ranks.append({"id": p.player_id, "rank": rank, "score": p.score})
            rank += 1
            
        for p in dead:
            ranks.append({"id": p.player_id, "rank": rank, "score": p.score})
            rank += 1
            
        self.broadcast({
            "t": MSG_GAME_OVER,
            "ranks": ranks,
            "winner_id": winner_id,
            "ended_tick": self.tick_id
        })
        
        self.status = "WAITING"
        self.host_id = None
        
        # Re-elect host from connected players
        connected = [p for p in self.players.values() if p.connected]
        if connected:
            self.host_id = connected[0].player_id
        else:
            self.status = "IDLE"

# --- Main Server ---

class SnakeServer:
    def __init__(self):
        self.rooms = {}
        
        # Load Model
        self.model = Linear_QNet(20, 128, 3)
        try:
            self.model.load_state_dict(torch.load('./model/model.pth', map_location='cpu'))
            self.model.eval()
            print("Loaded AI Model")
        except Exception as e:
            print(f"Could not load AI model: {e}. Bots will be random?")
            
        for i in range(ROOM_COUNT):
            rid = f"room-{i+1}"
            self.rooms[rid] = Room(rid)
            
            # Auto-add bot to EVERY Room
            bot = BotPlayer(f"bot_{i}", self.model)
            self.rooms[rid].add_player(bot)
        
        self.players = {} # ws -> PlayerState

    async def handler(self, websocket):
        player_id = str(uuid.uuid4())[:8]
        current_room = None
        player = None
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] New connection: {websocket.remote_address[0] if websocket.remote_address else 'unknown'}({player_id})")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                mtype = data.get("t")
                
                if mtype == MSG_JOIN:
                    rid = data.get("room_id")
                    username = data.get("username", "Guest")[:10]
                    
                    if rid not in self.rooms:
                        await websocket.send(json.dumps({"t": MSG_ERROR, "code": "ROOM_NOT_FOUND"}))
                        continue
                        
                    room = self.rooms[rid]
                    player = PlayerState(player_id, username, websocket)
                    success, err = room.add_player(player)
                    
                    if success:
                        current_room = room
                        self.players[websocket] = player
                        
                        plist = [{"id": p.player_id, "name": p.username} for p in room.players.values()]
                        resp = {
                            "t": MSG_JOIN_OK,
                            "room_id": rid,
                            "status": room.status,
                            "map": {"w": MAP_WIDTH, "h": MAP_HEIGHT},
                            "players": plist,
                            "your_id": player_id
                        }
                        
                        if room.status == "RUNNING":
                            snapshot_snakes = {}
                            for p in room.players.values():
                                if p.alive:
                                    snapshot_snakes[p.player_id] = {
                                        "body": list(p.body),
                                        "name": p.username,
                                        "score": p.score,
                                        "alive": True
                                    }
                            resp["snapshot"] = {
                                "snakes": snapshot_snakes,
                                "food": room.food
                            }

                        await websocket.send(json.dumps(resp))
                    else:
                        await websocket.send(json.dumps({"t": MSG_ERROR, "code": err}))
                
                elif mtype == MSG_INPUT:
                    if player and current_room and player.alive:
                        d_str = data.get("d")
                        # 0=Up, 1=Down, 2=Left, 3=Right
                        
                        # Prevent 180 degree reverse
                        old_dir = player.direction
                        new_dir = None
                        if d_str == 'up' and old_dir != (0, 1): new_dir = (0, -1)
                        if d_str == 'down' and old_dir != (0, -1): new_dir = (0, 1)
                        if d_str == 'left' and old_dir != (1, 0): new_dir = (-1, 0)
                        if d_str == 'right' and old_dir != (-1, 0): new_dir = (1, 0)
                        
                        if new_dir:
                            player.direction = new_dir
                            
                elif mtype == "start_request":
                    if current_room and current_room.host_id == player.player_id and current_room.status == "WAITING":
                        # Check min players
                        if len(current_room.players) >= 2:
                            current_room.start_game("MANUAL")
                        else:
                            # Allow 1 player start for debugging
                            current_room.start_game("MANUAL_DEBUG")

                elif mtype == MSG_EXIT:
                    print(f"Explicit exit from {player_id}")
                    break 

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if current_room and player:
                current_room.remove_player(player.player_id)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Connection closed: {player_id}")

    async def game_loop(self):
        while True:
            start_t = time.time()
            
            for room in self.rooms.values():
                # Auto Start Logic
                if room.status == "WAITING":
                    if len(room.players) >= room.capacity:
                        room.start_game("REF_FULL")
                    elif len(room.players) >= 2:
                         if room.countdown_deadline is None:
                             room.countdown_deadline = time.time() + 5
                         elif time.time() >= room.countdown_deadline:
                             room.start_game("COUNTDOWN")
                    else:
                        room.countdown_deadline = None
                
                elif room.status == "RUNNING":
                    room.step()
            
            elapsed = time.time() - start_t
            sleep_t = max(0, TICK_DT_MS/1000.0 - elapsed)
            await asyncio.sleep(sleep_t)

    async def start(self):
        # Start WebSocket Server
        # Increase ping_timeout to avoid 1011 errors on laggy networks
        async with websockets.serve(self.handler, "0.0.0.0", 8765, ping_interval=20, ping_timeout=60):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Server started on ws://0.0.0.0:8765")
            await self.game_loop()

if __name__ == "__main__":
    server = SnakeServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("Server stopped.")
