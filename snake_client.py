import asyncio
import json
import threading
import queue
import time
from collections import deque
import websockets
from snake_protocol import *

class SnakeClient:
    def __init__(self):
        self.ws = None
        self.loop = None
        self.thread = None
        self.running = False
        
        # State (Thread Safe via copy or simple types)
        # For simplicity, we'll expose state directly but mutations happen in network thread
        # Reading from GUI thread might have race conditions but for rendering it looks glitchy at worst
        self.state_lock = threading.Lock()
        
        self.my_id = None
        self.room_id = None
        self.status = "IDLE"
        self.snakes = {} # id -> {body: deque([(x,y)]), color: ...}
        self.food = None
        self.ranks = []
        self.winner = None
        
        self.msg_queue = queue.Queue() # For sending out from Main Thread

    def connect_and_start(self, uri, username, room_id):
        self.username = username
        self.target_room_id = room_id
        self.running = True
        
        self.thread = threading.Thread(target=self._run_loop, args=(uri,), daemon=True)
        self.thread.start()

    def _run_loop(self, uri):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._async_connect(uri))

    async def _async_connect(self, uri):
        try:
            async with websockets.connect(uri) as ws:
                self.ws = ws
                
                # Send Join
                join_msg = {
                    "t": MSG_JOIN,
                    "room_id": self.target_room_id,
                    "username": self.username
                }
                await ws.send(json.dumps(join_msg))
                
                # Receive Loop & Send Loop
                # We can use create_task for sending
                send_task = asyncio.create_task(self._sender(ws))
                
                try:
                    async for message in ws:
                         await self._handle_message(message)
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                finally:
                    send_task.cancel()
                    
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            self.running = False

    async def _sender(self, ws):
        while True:
            try:
                # Non-blocking check?
                # We can loop and sleep
                while not self.msg_queue.empty():
                    msg = self.msg_queue.get()
                    await ws.send(json.dumps(msg))
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Sender Error: {e}")
                break

    async def _handle_message(self, message):
        data = json.loads(message)
        mtype = data.get("t")
        
        with self.state_lock:
            if mtype == MSG_JOIN_OK:
                self.my_id = data["your_id"]
                self.room_id = data["room_id"]
                self.status = data["status"]
                
                # Handle Snapshot
                snap = data.get("snapshot")
                if snap:
                    # Food is list of lists, convert to list of tuples
                    self.food = [tuple(f) for f in snap.get("food", [])]
                    raw_snakes = snap.get("snakes", {})
                    self.snakes = {}
                    for pid, info in raw_snakes.items():
                         info["body"] = deque([tuple(x) for x in info["body"]])
                         self.snakes[pid] = info
                
                # Initialize players list if needed
                
            elif mtype == MSG_GAME_START:
                self.status = "RUNNING"
                self.food = [tuple(f) for f in data.get("food", [])]
                self.snakes = {}
                for p in data["players"]:
                    pid = p["id"]
                    body = deque([tuple(c) for c in p["body"]])
                    # Assign a random color based on ID hash or something
                    # For now just store body
                    self.snakes[pid] = {
                        "body": body,
                        "name": p["name"],
                        "alive": True,
                        "score": 0
                    }
                    
            elif mtype == MSG_DELTA:
                # Apply moves
                if "food" in data:
                     self.food = [tuple(f) for f in data.get("food", [])]
                
                moves = data.get("moves", [])
                for m in moves:
                    pid = m["id"]
                    
                    if m.get("dead"):
                        if pid in self.snakes:
                            self.snakes[pid]["alive"] = False
                            self.snakes[pid]["body"].clear() # Clear body immediately
                        continue
                        
                    if pid not in self.snakes:
                         # Should have been in start, but maybe joined late?
                         # Spec says no mid-game join.
                         continue
                         
                    snake = self.snakes[pid]
                    head_add = tuple(m["head_add"])
                    snake["body"].appendleft(head_add)
                    
                    if m.get("tail_remove"):
                        snake["body"].pop()
                        
                    snake["score"] = m.get("score", 0)
                    
            elif mtype == MSG_GAME_OVER:
                self.status = "FINISHED"
                self.ranks = data.get("ranks", [])
                self.winner = data.get("winner_id")
                
            elif mtype == MSG_ERROR:
                print(f"Server Error: {data.get('code')}")

    def send_input(self, direction):
        # direction: 'up', 'down', 'left', 'right'
        msg = {
            "t": MSG_INPUT,
            "d": direction
        }
        self.msg_queue.put(msg)
        
    def send_start_request(self):
        msg = {"t": "start_request"}
        self.msg_queue.put(msg)
        
    def stop(self):
        # Send explicit exit
        msg = {"t": MSG_EXIT}
        self.msg_queue.put(msg)
        self.running = False
        # The thread will eventually close when connection drops or we can force close
        # But letting the queue process the exit msg is best.

    def get_render_state(self):
        with self.state_lock:
            # Return a deep copy or snapshot if needed
            # For performance with many snakes, shallow copy + new reference is ok
            # since we replace body deques or modify them.
            # actually we modify deques in place. 
            # For render, we iterate.
            return {
                "status": self.status,
                "snakes": self.snakes,
                "food": self.food,
                "my_id": self.my_id,
                "ranks": self.ranks,
                "winner": self.winner
            }

