import asyncio
import websockets
import json
import random
import argparse
import signal
import time
from snake_protocol import *

SERVER_IP = "127.0.0.1"

async def stress_client(client_id, server_uri, room_id):
    while True:
        try:
            # close_timeout=0.2 helps to speed up retry loops if server is unresponsive
            async with websockets.connect(server_uri, close_timeout=0.2) as websocket:
                # Join
                msg = {
                    "t": MSG_JOIN,
                    "room_id": room_id,
                    "username": f"Bot_{client_id}"
                }
                await websocket.send(json.dumps(msg))

                print(f"Client {client_id} connected to {room_id}")

                # Game Loop
                # Split Input (Writer) and Output (Reader) to avoid blocking Pings/receiving
                
                async def reader():
                    expected_interval = 1.0 / SIM_TICK_HZ  # e.g. 0.0666s
                    last_time = None
                    jitter_sum = 0.0
                    jitter_count = 0
                    
                    try:
                        async for message in websocket:
                            arrival_time = time.time()
                            
                            # Passive measurement: the server sends MSG_DELTA regularly.
                            if last_time is not None:
                                delta = arrival_time - last_time
                                jitter = abs(delta - expected_interval)
                                
                                jitter_sum += jitter
                                jitter_count += 1
                                
                            last_time = arrival_time
                    except Exception:
                        pass
                    finally:
                        # Report on exit
                        if jitter_count > 0:
                            avg_jitter_ms = (jitter_sum / jitter_count) * 1000.0
                            print(f"[Client {client_id}] Connection Closed. Avg Jitter: {avg_jitter_ms:.2f} ms ({jitter_count} pkts)")
                        else:
                            print(f"[Client {client_id}] Connection Closed. No packets for jitter calc.")

                async def writer():
                    directions = ['up', 'down', 'left', 'right']
                    try:
                        while True:
                            d = random.choice(directions)
                            input_msg = {"t": MSG_INPUT, "d": d}
                            await websocket.send(json.dumps(input_msg))
                            await asyncio.sleep(0.1)  # Send at ~10Hz
                    except Exception:
                        pass

                # Run both until one fails (likely connection closed)
                done, pending = await asyncio.wait(
                    [asyncio.create_task(reader()), asyncio.create_task(writer())],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in pending:
                    task.cancel()

        except websockets.exceptions.ConnectionClosed:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            return
        except Exception:
            await asyncio.sleep(1)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10, help="Number of clients")
    parser.add_argument("--uri", type=str, default=f"ws://{SERVER_IP}:8765", help="Server URI")
    args = parser.parse_args()

    print(f"Starting {args.count} stress clients on {args.uri}...")

    tasks = []
    for i in range(args.count):
        rid = f"room-{random.randint(1, 20)}"
        tasks.append(asyncio.create_task(stress_client(i, args.uri, rid)))

    # Ctrl+C 立刻印訊息 + 取消 tasks，並攔截第二次 Ctrl+C
    loop = asyncio.get_running_loop()
    shutting_down = False

    def on_sigint():
        nonlocal shutting_down
        if shutting_down:
            return  # 第二次 Ctrl+C 直接攔截掉
        shutting_down = True
        print("\nStopping stress test... 請稍等（正在正確關閉連線）", flush=True)
        for t in tasks:
            t.cancel()

    # *nix 最穩：用 event loop 的 signal handler
    try:
        loop.add_signal_handler(signal.SIGINT, on_sigint)
        loop.add_signal_handler(signal.SIGTERM, on_sigint)  # 可選：kill 也走同邏輯
    except NotImplementedError:
        # Windows fallback
        signal.signal(signal.SIGINT, lambda *_: loop.call_soon_threadsafe(on_sigint))

    # 等全部 task 收尾
    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())

