import os
import time
import torch
import argparse
import numpy as np
from snake_env import Game
from snake_agent import Agent
from helper import plot

def train(visualization):
    plot_scores = []
    plot_mean_scores = []
    record = 0
    total_step = 0
    game = Game(tick_rate=10000)
    agent = Agent(game.nS,game.nA)
    state_new = game.get_state()

    # 0.檢查是否有舊模型，若有則載入繼續訓練，避免覆蓋
    if os.path.exists('./model/model.pth'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent.trainer.model.load_state_dict(torch.load('./model/model.pth', map_location=device, weights_only=True))
        agent.trainer.copy_model()
        print(f"已載入現有模型至 {device}，繼續訓練...")
    
    try:
        print(f"[{time.strftime('%H:%M:%S')}] 開始新訓練...")
        start_time = time.time()

        while True:
            state_old = state_new
            final_move = agent.get_action(state_old,agent.n_game)
            reward, done, score = game.play_step(final_move, visualization=visualization)
            state_new = game.get_state()

            # 1. 訓練短期記憶 (Short Memory): 每一步都學習
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            agent.remember(state_old, final_move, reward, state_new, done)
            total_step += 1

            if done:
                game.reset()
                agent.n_game += 1
                
                # 2. 訓練長期記憶 (Long Memory): 遊戲結束後進行經驗回放
                agent.train_long_memory(batch_size=512)
                # 更新目標網路
                agent.trainer.copy_model()

                if score > record:
                    record = score
                    agent.trainer.model.save()
                print('Game', agent.n_game, 'Score', score, 'Record:', record)
                plot_scores.append(score)
                mean_scores = np.mean(plot_scores[-10:])
                plot_mean_scores.append(mean_scores)
                plot(plot_scores, plot_mean_scores)
    
    except KeyboardInterrupt:
        print("\n正在強制存檔...")
        agent.trainer.model.save(file_name='interrupted_model.pth')
        duration = time.time() - start_time
        print(f"[{time.strftime('%H:%M:%S')}] 訓練完成，總耗時: {duration:.2f} 秒")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualization", type=bool, default=False)
    args = parser.parse_args()
    train(args.visualization)


