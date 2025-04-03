# sac.py

# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # Stores experience tuples

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))  # Store a transition

    def sample(self, batch_size):
        # Randomly sample batch_size experiences from buffer
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        # Convert to numpy arrays for training
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)  # Return number of stored transitions

# Simple feed-forward neural network for actor and critic
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hiddin_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hiddin_dim),  # Input layer
            nn.ReLU(),                         # Activation
            nn.Linear(hiddin_dim, hiddin_dim), # Hidden layer
            nn.ReLU(),                         # Activation
            nn.Linear(hiddin_dim, output_dim)  # Output layer
        )

    def forward(self, x):
        return self.net(x)  # Forward pass through the network

# Main SAC Agent class
class SACAgent:
    def __init__(self, obs_dim, action_dim, action_bound, device, lr=3e-4, alpha=0.2):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.device = device

        # Actor network (policy)
        self.actor = MLP(obs_dim, action_dim).to(device)

        # Critic networks (Q1 and Q2) - use 2 to reduce overestimation bias
        self.q1 = MLP(obs_dim + action_dim, 1).to(device)
        self.q2 = MLP(obs_dim + action_dim, 1).to(device)

        # Target networks for Q1 and Q2 
        self.target_q1 = MLP(obs_dim + action_dim, 1).to(device)
        self.target_q2 = MLP(obs_dim + action_dim, 1).to(device)

        # Initialize target networks with same weights
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        # Optimizers for each network
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=lr)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

        # Discount factor
        self.gamma = 0.99
        # Target network soft update factor
        self.tau = 0.005
        # Entropy coefficient
        self.alpha = alpha
        # Training batch size
        self.batch_size = 256

    def select_action(self, state):
        # Convert state to tensor and add batch dimension
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # Disable gradient tracking for inference
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]  # Return as numpy array

    def update(self):
        # Skip update if not enough data in buffer
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of experiences
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        # Convert to PyTorch tensors and send to device
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # Compute target Q-value using target networks and actor
        with torch.no_grad():
            next_action = self.actor(next_state)
            target_q1 = self.target_q1(torch.cat([next_state, next_action], dim=1))
            target_q2 = self.target_q2(torch.cat([next_state, next_action], dim=1))
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_action.pow(2).sum(1, keepdim=True)
            target_value = reward + (1 - done) * self.gamma * target_q  # Bellman backup

        # Compute current Q-values using online networks
        current_q1 = self.q1(torch.cat([state, action], dim=1))
        current_q2 = self.q2(torch.cat([state, action], dim=1))

        # Compute critic losses (MSE)
        q1_loss = F.mse_loss(current_q1, target_value)
        q2_loss = F.mse_loss(current_q2, target_value)

        # Backprop for Q1
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        # Backprop for Q2
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # Actor loss: encourage actions that maximize Q - α * entropy (squared action penalty)
        new_action = self.actor(state)
        actor_loss = -(self.q1(torch.cat([state, new_action], dim=1)) - self.alpha * new_action.pow(2).sum(1, keepdim=True)).mean()

        # Backprop for actor
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update target networks
        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
