# sac_bg_control/main.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sac import SACAgent
from environment import BloodGlucoseEnv
from pomdp import POMDPWrapper

import gymnasium as gym
import sys

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='Entropy temperature')
    parser.add_argument('--history-length', type=int, default=10, help='POMDP history length')
    parser.add_argument('--episodes', type=int, default=1000, help='Training episodes')
    return parser.parse_args()

def main():
    sys.modules['gym'] = gym
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment and wrappers
    env = BloodGlucoseEnv()
    env = POMDPWrapper(env)

    obs, _ = env.reset()
    obs_dim = obs.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])

    # SAC Agent
    agent = SACAgent(obs_dim, action_dim, action_bound, device,
                     lr=args.lr, alpha=args.alpha)
    num_episodes = 1000
    for episode in range(args.episodes):
        obs, _ = env.reset()
        # print("obs_dim from env:", obs_dim) # Debug
        # print("obs from reset:", obs) # Debug
        episode_reward = 0
        terminated = False
        while not terminated:
            # print("obs shape:", np.shape(obs)) # Debug
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.replay_buffer.push(obs, action, reward, next_obs, terminated)
            agent.update()
            obs = next_obs
            episode_reward += reward

        print(f"Episode {episode}: Reward = {episode_reward:.2f}")

if __name__ == "__main__":
    main()