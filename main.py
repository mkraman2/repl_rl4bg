# sac_bg_control/main.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sac import SACAgent
from environment import BloodGlucoseEnv
from pomdp import POMDPWrapper
from logger import GlucoseLogger
from pid_controller import PIDController
from plot_logs import plot_from_log

import gymnasium as gym
import sys
import os
import csv

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='Entropy temperature')
    parser.add_argument('--history-length', type=int, default=10, help='POMDP history length')
    parser.add_argument('--episodes', type=int, default=1000, help='Training episodes')
    parser.add_argument("--controller", choices=["sac", "sac-t", "pid"], default="sac",
                        help="Choose the controller type: 'sac', 'sac-t', 'pid'")
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

    # Create Agent (sac or pid)
    if args.controller == "sac":
        agent = SACAgent(obs_dim, action_dim, action_bound, device,
                        lr=args.lr, alpha=args.alpha)
    elif args.controller == "sac-t":
        agent = SACAgent(obs_dim, action_dim, action_bound, device,
                        lr=args.lr, alpha=args.alpha, use_transformer=True, history_length=args.history_length)        
    else:
        agent = PIDController(Kp=0.05, Ki=0.001, Kd=0.01)

    log_file = "glucose_log.csv"
    if os.path.exists(log_file):
        os.remove(log_file)

    # Create logger
    logger = GlucoseLogger()

    for episode in range(args.episodes):
        obs, _ = env.reset()
        episode_reward = 0
        terminated = False
        t = 0
        while not terminated:
            if args.controller in ["sac", "sac-t"]:
                action = agent.select_action(obs)
            else:
                action = np.array([agent.compute_action(obs[0])])

            next_obs, reward, terminated, truncated, info = env.step(action)
            glucose = info.get("CGM", next_obs[-1])
            insulin = float(action[0].item())
            logger.log_step(episode, t, glucose, insulin, reward)

            if args.controller in ["sac", "sac-t"]:
                agent.replay_buffer.push(obs, action, reward, next_obs, terminated)
                agent.update()

            obs = next_obs
            episode_reward += reward
            t += 1

        print(f"Episode {episode}: Reward = {episode_reward:.2f}")

    plot_from_log()


if __name__ == "__main__":
    main()
