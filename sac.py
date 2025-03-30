# sac.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class SACAgent:
    def __init__(self, obs_dim, action_dim, action_bound, device, lr=3e-4, alpha=0.2):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.device = device

        self.actor = MLP(obs_dim, action_dim).to(device)
        self.q1 = MLP(obs_dim + action_dim, 1).to(device)
        self.q2 = MLP(obs_dim + action_dim, 1).to(device)
        self.target_q1 = MLP(obs_dim + action_dim, 1).to(device)
        self.target_q2 = MLP(obs_dim + action_dim, 1).to(device)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=3e-4)

        self.replay_buffer = ReplayBuffer(1000000)

        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = alpha
        self.batch_size = 256

    def select_action(self, state):
        # print("select_action - raw state type/shape:", type(state), np.shape(state)) # Debug
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # print("select_action - tensor shape:", state.shape)  # Debug   
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action = self.actor(next_state)
            target_q1 = self.target_q1(torch.cat([next_state, next_action], dim=1))
            target_q2 = self.target_q2(torch.cat([next_state, next_action], dim=1))
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_action.pow(2).sum(1, keepdim=True)
            target_value = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.q1(torch.cat([state, action], dim=1))
        current_q2 = self.q2(torch.cat([state, action], dim=1))

        q1_loss = F.mse_loss(current_q1, target_value)
        q2_loss = F.mse_loss(current_q2, target_value)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        new_action = self.actor(state)
        actor_loss = -(self.q1(torch.cat([state, new_action], dim=1)) - self.alpha * new_action.pow(2).sum(1, keepdim=True)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Update targets
        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
