"""
AI Snake multiplayer GUI client.

Connects to snake_server.py.
"""
import random
import pygame
import pygame_menu
import argparse
import threading
import time
import asyncio
import json
import websockets
from snake_client import SnakeClient
from snake_protocol import *

# --- Constants Match Server ---
# Map is 50x50. Cell size 20 => 1000x1000 pixels.
# That's quite large for some screens. Let's see.
# If we keep 20px, it's 1000px height.
SCREEN_WIDTH = MAP_WIDTH * CELL_SIZE
SCREEN_HEIGHT = MAP_HEIGHT * CELL_SIZE
BGCOLOR = (20, 20, 20)
RED = (255, 0, 0)
FPS = 60 # Client render FPS (server is 30)
MENU_FPS = 30
MENU_REFRESH_SEC = 0.5
ROOM_STATS_POLL_SEC = 2.0
ROOM_STATS_TIMEOUT_SEC = 1.0

class RoomStatsPoller:
    """Poll lightweight room stats API in a background thread."""

    def __init__(self, room_count=ROOM_COUNT, poll_interval=2.0, timeout=1.0):
        self.room_count = room_count
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.server_ip = "127.0.0.1"
        self.thread = None
        self.stop_event = threading.Event()
        self.stats_lock = threading.Lock()
        self.stats_by_room = {}

    def start(self, server_ip):
        self.server_ip = server_ip
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def set_server_ip(self, server_ip):
        server_ip = (server_ip or "").strip()
        if server_ip:
            self.server_ip = server_ip

    async def _fetch_room_stats_once(self, uri):
        async with websockets.connect(uri, open_timeout=self.timeout, close_timeout=0.2) as ws:
            await ws.send(json.dumps({"t": MSG_ROOM_STATS_REQ}))
            raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
            data = json.loads(raw)
            if data.get("t") != MSG_ROOM_STATS:
                return None
            rooms = data.get("rooms", [])
            if not isinstance(rooms, list):
                return None
            return rooms

    def _worker(self):
        while not self.stop_event.is_set():
            uri = f"ws://{self.server_ip}:8765"
            try:
                rooms = asyncio.run(self._fetch_room_stats_once(uri))
                if rooms is not None:
                    next_stats = {}
                    for room in rooms:
                        rid = room.get("room_id")
                        if rid:
                            next_stats[rid] = room
                    with self.stats_lock:
                        self.stats_by_room = next_stats
            except Exception:
                # Keep menu responsive even if backend is temporarily unreachable.
                pass

            self.stop_event.wait(self.poll_interval)

    def _format_label(self, room_number, stat):
        if not stat:
            return f"{room_number} (Loading)"

        if (stat.get("available_slots", 0) or 0) <= 0:
            return f"{room_number} (Full)"

        display_players = stat.get("display_players")
        if display_players is None:
            display_players = stat.get("connected_players", 0)

        if stat.get("status") == "RUNNING":
            return f"{room_number} (Running {display_players})"
        return f"{room_number} (Waiting {display_players})"

    def get_room_items(self):
        with self.stats_lock:
            snapshot = dict(self.stats_by_room)

        items = []
        for i in range(1, self.room_count + 1):
            room_id = f"room-{i}"
            label = self._format_label(i, snapshot.get(room_id))
            items.append((label, room_id))
        return items

def update_room_selector_items(room_selector, room_items):
    """Update selector options while preserving current selected room."""
    selected_room = None
    try:
        selected_room = room_selector.get_value()[0][1]
    except Exception:
        pass

    updated = False
    try:
        room_selector.update_items(room_items)
        updated = True
    except Exception:
        try:
            room_selector.set_items(room_items)
            updated = True
        except Exception:
            return False

    if updated and selected_room is not None:
        selected_idx = next((i for i, item in enumerate(room_items) if item[1] == selected_room), 0)
        try:
            room_selector.set_value(selected_idx)
        except Exception:
            pass
    return True

class NetworkGame:
    def __init__(self, username, room_id, server_ip="127.0.0.1"):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(f'Snake Connect - Room: {room_id} - User: {username}')
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font = pygame.font.SysFont('arial', 20, bold=True)
        
        # Network
        self.client = SnakeClient()
        uri = f"ws://{server_ip}:8765"
        self.client.connect_and_start(uri, username, room_id)
        
        self.running = True

    def draw_grid(self):
        # Draw some subtle grid lines
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, (40, 40, 40), (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, (40, 40, 40), (0, y), (SCREEN_WIDTH, y))

    def run(self):
        while self.running:
            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    
                    # Input
                    d = None
                    if event.key == pygame.K_w or event.key == pygame.K_UP: d = "up"
                    elif event.key == pygame.K_s or event.key == pygame.K_DOWN: d = "down"
                    elif event.key == pygame.K_a or event.key == pygame.K_LEFT: d = "left"
                    elif event.key == pygame.K_d or event.key == pygame.K_RIGHT: d = "right"
                    
                    if d:
                        self.client.send_input(d)
                        
                    # Manual Start (Space)
                    if event.key == pygame.K_SPACE:
                        self.client.send_start_request()

            # 2. Get State
            state = self.client.get_render_state()
            my_id = state.get("my_id")
            status = state.get("status")
            
            # 3. Draw
            self.screen.fill(BGCOLOR)
            self.draw_grid()
            
            # Draw Food
            if self.client.food:
                for fx, fy in self.client.food:
                    rect = pygame.Rect(fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, RED, rect) # Red Food
                
            # Draw Snakes
            snakes = state.get("snakes", {})
            for pid, s in snakes.items():
                body = s["body"]
                if not body: continue
                
                is_me = (pid == my_id)
                # Color: Green if me, Blue/Other if enemy
                # Simple hash color for enemies
                if is_me:
                    color = (0, 255, 0) if s["alive"] else (0, 100, 0)
                    head_color = (100, 255, 100)
                else:
                    if s["alive"]:
                        # Generate consistent color from PID
                        random.seed(pid)
                        r = random.randint(50, 255)
                        g = random.randint(50, 150) # Less green to distinguish
                        b = random.randint(100, 255)
                        color = (r, g, b)
                        head_color = (min(r+50,255), min(g+50,255), min(b+50,255))
                    else:
                        color = (80, 80, 80)
                        head_color = (100, 100, 100)
                
                # Draw Body
                for i, part in enumerate(body):
                    px, py = part
                    rect = pygame.Rect(px * CELL_SIZE, py * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    if i == 0: # Head
                        pygame.draw.rect(self.screen, head_color, rect)
                    else:
                        pygame.draw.rect(self.screen, color, rect)
                        # Inner detail
                # Name Tag - Name Only
                if body:
                    hx, hy = body[0]
                    name = s.get("name", "Unknown")
                    # score = s.get("score", 0)
                    tag = f"{name}"
                    txt = self.font.render(tag, True, (255, 255, 255))
                    self.screen.blit(txt, (hx * CELL_SIZE, hy * CELL_SIZE - 20))

            # HUD
            room_disp = self.client.target_room_id.replace("room-", "")
            
            display_status = status
            if len(snakes) >= ROOM_CAPACITY and status not in ("RUNNING", "FINISHED"):
                display_status = "FULL"
            
            state_txt = self.font.render(f"Status: {display_status} | Room: {room_disp} | Players: {len(snakes)}", True, (255, 255, 255))
            self.screen.blit(state_txt, (10, 10))
            
            # Start Button (Visual)
            if status == "WAITING":
                # Draw Button
                btn_rect = pygame.Rect(SCREEN_WIDTH//2 - 60, SCREEN_HEIGHT - 60, 120, 40)
                mouse_pos = pygame.mouse.get_pos()
                click = pygame.mouse.get_pressed()
                
                color = (0, 200, 0)
                if btn_rect.collidepoint(mouse_pos):
                    color = (0, 255, 0)
                    if click[0]: # Left Click
                        self.client.send_start_request()
                        
                pygame.draw.rect(self.screen, color, btn_rect)
                btn_txt = self.font.render("START", True, (0, 0, 0))
                self.screen.blit(btn_txt, (btn_rect.x + 30, btn_rect.y + 10))
                
                info = self.font.render("Waiting... Press START or SPACE", True, (255, 255, 0))
                self.screen.blit(info, (SCREEN_WIDTH//2 - 170, SCREEN_HEIGHT//2))
                delay_hint = self.font.render("Game starts about 5 seconds after start request", True, (255, 220, 120))
                self.screen.blit(delay_hint, (SCREEN_WIDTH//2 - 250, SCREEN_HEIGHT//2 + 30))
                
            elif status == "FINISHED":
                if state.get("winner"):
                    w_txt = self.font.render(f"Winner: {state['winner']}", True, (0, 255, 255))
                    self.screen.blit(w_txt, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2))
            
            # Scoreboard (Top Right)
            # Create list of (score, name)
            scores = []
            for pid, s in snakes.items():
                scores.append((s.get("score", 0), s.get("name", "Unknown")))
            scores.sort(key=lambda x: x[0], reverse=True)
            
            start_y = 10
            for sc, nm in scores[:10]: # Top 10
                 txt = self.font.render(f"{nm}: {sc}", True, (200, 200, 200))
                 self.screen.blit(txt, (SCREEN_WIDTH - 150, start_y))
                 start_y += 25
            
            
            pygame.display.flip()
            self.clock.tick(FPS)
            
        # pygame.quit() # Removed to prevent crash
        self.client.stop() # Send Exit Signal


# --- Menu ---

def start_client(room_value, user_input, ip_input):
    # room_value format from Selector: (('Label', 'value'), index)
    # user_input: text input widget
    room_id = room_value[0][1]
    username = user_input.get_value()
    ip = ip_input.get_value()
    
    game = NetworkGame(username, room_id, ip)
    game.run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", type=str, default=f"20.239.90.85", help="Server IP")
    args = parser.parse_args()

    pygame.init()
    # Resize to 1000x1000
    surface = pygame.display.set_mode((1000, 1000))
    pygame.display.set_caption('Snake Connect - Menu')
    
    my_theme = pygame_menu.themes.THEME_BLUE.copy()
    menu = pygame_menu.Menu('Snake Connect', 1000, 1000, theme=my_theme)
    
    user_input = menu.add.text_input('Name: ', default='Player1')
    
    # Room dropdown with live occupancy labels.
    room_poller = RoomStatsPoller(
        room_count=ROOM_COUNT,
        poll_interval=ROOM_STATS_POLL_SEC,
        timeout=ROOM_STATS_TIMEOUT_SEC
    )
    room_poller.start(args.uri)
    room_items = room_poller.get_room_items()
    room_selector = menu.add.selector('Room: ', room_items)
    
    ip_input = menu.add.text_input('Server IP: ', default=f'{args.uri}')
    
    menu.add.button('Join Game', lambda: start_client(room_selector.get_value(), user_input, ip_input))
    menu.add.button('Quit', pygame_menu.events.EXIT)

    clock = pygame.time.Clock()
    last_refresh = 0.0
    last_room_items = room_items

    try:
        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    return

            room_poller.set_server_ip(ip_input.get_value())

            now = time.time()
            if now - last_refresh >= MENU_REFRESH_SEC:
                current_items = room_poller.get_room_items()
                if current_items != last_room_items:
                    update_room_selector_items(room_selector, current_items)
                    last_room_items = current_items
                last_refresh = now

            if menu.is_enabled():
                menu.update(events)
                menu.draw(surface)
                pygame.display.flip()
                clock.tick(MENU_FPS)
            else:
                break
    finally:
        room_poller.stop()

if __name__ == '__main__':
    main()
