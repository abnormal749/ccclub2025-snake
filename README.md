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
