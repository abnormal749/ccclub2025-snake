import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import os

# --- 新增：檢查是否有 GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        # 儲存模型時建議先轉回 CPU，這樣以後在沒 GPU 的電腦也能載入
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, lr, gamma, input_dim, hidden_dim, output_dim):
        self.gamma = gamma
        self.hidden_size = hidden_dim
        # --- 修改：將模型移至 GPU ---
        self.model = Linear_QNet(input_dim, self.hidden_size, output_dim).to(device)
        self.target_model = Linear_QNet(input_dim, self.hidden_size, output_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.copy_model()

    def copy_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, state, action, reward, next_state, done):
        # --- 修改：將所有 Tensor 移至 GPU ---
        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.long).to(device)
        action = torch.unsqueeze(action, -1)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        done = torch.tensor(done, dtype=torch.long).to(device)

        # 這裡的計算會在 GPU 上進行
        Q_value = self.model(state).gather(-1, action).squeeze()
        Q_value_next = self.target_model(next_state).detach().max(-1)[0]
        target = (reward + self.gamma * Q_value_next * (1 - done)).squeeze()

        self.optimizer.zero_grad()
        loss = self.criterion(Q_value, target)
        loss.backward()
        self.optimizer.step()

class Agent:
    def __init__(self, nS, nA, max_explore=100, gamma=0.9,
                 max_memory=50000, lr=0.001, hidden_dim=128):
        self.max_explore = max_explore 
        self.memory = deque(maxlen=max_memory) 
        self.nS = nS
        self.nA = nA
        self.n_game = 0
        self.trainer = QTrainer(lr, gamma, self.nS, hidden_dim, self.nA)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 

    def train_long_memory(self, batch_size, repeat=5): # 新增 repeat 參數
        for _ in range(repeat): # 讓 GPU 連續運算多次
            if len(self.memory) > batch_size:
                mini_sample = random.sample(self.memory, batch_size)
            else:
                mini_sample = self.memory
        
            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, n_game, explore=True):
        # --- 修改：將輸入 state 移至 GPU，並在最後轉回 CPU 轉成 numpy ---
        state = torch.tensor(state, dtype=torch.float).to(device)
        prediction = self.trainer.model(state).detach().cpu().numpy().squeeze()
        
        epsilon = self.max_explore - n_game
        if explore and random.randint(0, self.max_explore) < epsilon:
            # 這裡使用 numpy 計算 softmax， prediction 已在 CPU
            prob = np.exp(prediction)/np.exp(prediction).sum()
            final_move = np.random.choice(len(prob), p=prob)
        else:
            final_move = prediction.argmax()
        return final_move
