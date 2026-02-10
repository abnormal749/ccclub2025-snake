import asyncio
import websockets
import json
import random
import argparse
import signal
import time
from snake_protocol import *

SERVER_IP = "127.0.0.1"
START_REQUEST = "start_request"

async def stress_client(client_id, server_uri, room_count, input_hz):
    input_interval = 1.0 / max(1.0, input_hz)
    while True:
        room_id = f"room-{random.randint(1, room_count)}"
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

                try:
                    join_raw = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    join_data = json.loads(join_raw)
                except Exception:
                    print(f"[Client {client_id}] Join timeout/error on {room_id}")
                    await asyncio.sleep(0.5)
                    continue

                if join_data.get("t") == MSG_ERROR:
                    print(f"[Client {client_id}] Join rejected on {room_id}: {join_data.get('code')}")
                    await asyncio.sleep(0.5)
                    continue

                if join_data.get("t") != MSG_JOIN_OK:
                    print(f"[Client {client_id}] Unexpected join response on {room_id}: {join_data.get('t')}")
                    await asyncio.sleep(0.5)
                    continue

                print(f"[Client {client_id}] Connected to {room_id} (status={join_data.get('status')})")

                # Game Loop
                # Split Input (Writer) and Output (Reader) to avoid blocking Pings/receiving
                
                async def reader():
                    expected_interval = 1.0 / SIM_TICK_HZ  # e.g. 0.0666s
                    last_delta_time = None
                    jitter_sum = 0.0
                    jitter_count = 0
                    delta_count = 0
                    
                    try:
                        async for message in websocket:
                            data = json.loads(message)
                            mtype = data.get("t")
                            arrival_time = time.time()

                            # Measure jitter only on MSG_DELTA from server tick.
                            if mtype == MSG_DELTA:
                                delta_count += 1
                                if last_delta_time is not None:
                                    delta_t = arrival_time - last_delta_time
                                    jitter = abs(delta_t - expected_interval)
                                    jitter_sum += jitter
                                    jitter_count += 1
                                last_delta_time = arrival_time

                            if mtype == MSG_ERROR:
                                print(f"[Client {client_id}] Server error: {data.get('code')}")
                                break
                    except Exception:
                        pass
                    finally:
                        # Report on exit
                        if jitter_count > 0:
                            avg_jitter_ms = (jitter_sum / jitter_count) * 1000.0
                            print(f"[Client {client_id}] Closed. Avg Jitter: {avg_jitter_ms:.2f} ms ({jitter_count} samples, {delta_count} deltas)")
                        else:
                            print(f"[Client {client_id}] Closed. No delta packets for jitter calc.")

                async def writer():
                    directions = ['up', 'down', 'left', 'right']
                    next_start_request_at = time.time()
                    try:
                        while True:
                            now = time.time()
                            if now >= next_start_request_at:
                                # Host-only on server side; harmless for non-host clients.
                                await websocket.send(json.dumps({"t": START_REQUEST}))
                                next_start_request_at = now + 1.0

                            d = random.choice(directions)
                            input_msg = {"t": MSG_INPUT, "d": d}
                            await websocket.send(json.dumps(input_msg))
                            await asyncio.sleep(input_interval)
                    except Exception:
                        pass

                # Run both until one fails (likely connection closed)
                done, pending = await asyncio.wait(
                    [asyncio.create_task(reader()), asyncio.create_task(writer())],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        except websockets.exceptions.ConnectionClosed:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            return
        except Exception:
            await asyncio.sleep(0.5)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10, help="Number of clients")
    parser.add_argument("--uri", type=str, default=f"ws://{SERVER_IP}:8765", help="Server URI")
    parser.add_argument("--room-count", type=int, default=ROOM_COUNT, help="Room count to randomize from")
    parser.add_argument("--input-hz", type=float, default=10.0, help="Input send rate per client")
    args = parser.parse_args()

    print(f"Starting {args.count} stress clients on {args.uri}...")

    tasks = []
    for i in range(args.count):
        tasks.append(asyncio.create_task(stress_client(i, args.uri, args.room_count, args.input_hz)))

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
