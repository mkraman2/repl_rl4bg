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

def main():
    sys.modules['gym'] = gym
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment and wrappers
    env = BloodGlucoseEnv()
    env = POMDPWrapper(env)
    obs, _ = env.reset()
    # obs_dim = env.observation_space.shape[0]
    obs_dim = obs.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])

    # SAC Agent
    agent = SACAgent(obs_dim, action_dim, action_bound, device)

    num_episodes = 1000
    for episode in range(num_episodes):
        obs, _ = env.reset()
        print("obs_dim from env:", obs_dim) # Debug
        print("obs from reset:", obs) # Debug
        episode_reward = 0
        done = False
        while not done:
            print("obs shape:", np.shape(obs)) # Debug
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            agent.update()
            obs = next_obs
            episode_reward += reward

        print(f"Episode {episode}: Reward = {episode_reward:.2f}")

if __name__ == "__main__":
    main()