# Snake Connect

多人連線貪吃蛇（Python WebSocket 伺服器 + pygame 客戶端）。

## 快速開始

1. 安裝套件（Python 3.9+）
   `pip install pygame pygame-menu websockets torch numpy`
2. 啟動伺服器
   `python3 snake_server.py`
3. 啟動客戶端
   `python3 snake_gui.py`

## 遊戲說明

- 選擇名稱、房間（`room-1` ~ `room-20`）與伺服器 IP 後加入。
- 操作：`WASD` 或方向鍵。
- 主機可按 `SPACE` 或 `START` 開始。
- 等待中畫面會提示開始資訊。

## 目前規則（對齊最新程式）

- 地圖：`50 x 50`
- 房間數：`20`
- 每房上限：`10`
- 伺服器 Tick：`15Hz`
- 食物最多同時 `3` 個
- 遊戲開始時：只要有人類玩家，固定會有 `1` 隻 AI 一起玩（`AI`）
- 當所有人類都死亡後：若仍有 AI 存活，才會補上 `AI2` 進行 AI 對戰
- 若當下沒有 AI 存活：不會補新 AI，直接結算
- 勝者顯示為玩家名稱（含 `AI` / `AI2`），以最高分決定
- 每局結束後，所有玩家分數會重設為 `0`

## 壓力測試

`stress_test.py` 會建立多個無視窗客戶端，隨機進房並持續送出隨機方向輸入。

範例：

`python3 stress_test.py --count 50 --uri ws://127.0.0.1:8765`

可選參數：

- `--room-count`：隨機房間範圍（預設 `20`）
- `--input-hz`：每個壓測客戶端送輸入頻率（預設 `10`）

## 房間人數 API（WebSocket）

可在尚未 `join` 房間前直接查詢所有房間目前人數。

- 請求：`{"t":"room_stats_req"}`
- 回應：`{"t":"room_stats","rooms":[...]}`

`rooms` 內每個物件欄位：

- `room_id`：房間 ID（例如 `room-1`）
- `status`：`IDLE / WAITING / RUNNING / FINISHED`
- `connected_players`：目前連線中的玩家數（真人 + AI 全部分開計）
- `display_players`：顯示用玩家數（若無真人，AI/AI2 合併顯示為 `1`；否則與 `connected_players` 相同）
- `used_slots`：目前占用名額（與 server `ROOM_FULL` 判斷一致）
- `capacity`：房間上限
- `available_slots`：剩餘可加入名額

## 其他程式碼
- AI訓練部分：https://github.com/happyjimmy8964/SnakeAI_game_Interface
- 網頁版前端：https://github.com/abnormal749/snake_ai_frontend