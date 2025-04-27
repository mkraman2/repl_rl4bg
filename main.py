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
from gymnasium.vector import AsyncVectorEnv

import sys
import os
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='Entropy temperature')
    parser.add_argument('--history-length', type=int, default=10, help='POMDP history length')
    parser.add_argument('--episodes', type=int, default=1000, help='Training episodes')
    parser.add_argument("--controller", choices=["sac", "sac-t", "pid"], default="sac",
                        help="Choose the controller type: 'sac', 'sac-t', 'pid'")
    parser.add_argument('--num-envs', type=int, default=32, help='Number of parallel environments')
    parser.add_argument('--dnn-hidden-size', type=int, default=128, help='DNN hidden layer size')

    return parser.parse_args()

def make_env_fn(patient_name, history_length):
    def _init():
        env = BloodGlucoseEnv(patient_name=patient_name)
        env = POMDPWrapper(env, history_length=history_length)
        return env
    return _init

def main():
    sys.modules['gym'] = gym
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gradient_steps_per_env_step = 4  # You can tune this

    # Build patient list
    patient_list = []
    patient_list += [f"adult#{i:03d}" for i in range(1, 11)]
    # patient_list += [f"adolescent#{i:03d}" for i in range(1, 11)]
    # patient_list += [f"child#{i:03d}" for i in range(1, 11)]

    # Create vectorized environments
    env_fns = [
        make_env_fn(random.choice(patient_list), args.history_length)
        for _ in range(args.num_envs)
    ]
    envs = AsyncVectorEnv(env_fns)

    # Get obs/action dims from env
    obs, _ = envs.reset()
    obs_dim = obs.shape[1]  # (num_envs, obs_dim)
    action_dim = envs.single_action_space.shape[0]
    action_bound = float(envs.single_action_space.high[0])

    feature_dim = obs_dim // args.history_length

    # Initialize controller
    if args.controller == "sac":
        agent = SACAgent(obs_dim, action_dim, action_bound, device,
                         lr=args.lr, alpha=args.alpha, hidden_dim=args.dnn_hidden_size)
    elif args.controller == "sac-t":
        agent = SACAgent(obs_dim, action_dim, action_bound, device,
                         lr=args.lr, alpha=args.alpha, use_transformer=True,
                         feature_dim=feature_dim, history_length=args.history_length, hidden_dim=args.dnn_hidden_size)
    else:
        agent = PIDController(Kp=0.05, Ki=0.001, Kd=0.01)

    # Prepare logging
    log_file = "glucose_log.csv"
    if os.path.exists(log_file):
        os.remove(log_file)

    logger = GlucoseLogger()

    for episode in range(args.episodes):
        obs, _ = envs.reset()
        episode_rewards = np.zeros(args.num_envs)
        dones = np.zeros(args.num_envs, dtype=bool)
        t = 0

        while not np.all(dones):
            if args.controller in ["sac", "sac-t"]:
                actions = agent.select_action_batch(obs)
            else:
                actions = np.array([[agent.compute_action(o[0])] for o in obs])

            actions = np.clip(actions, 0.0, action_bound)

            next_obs, rewards, terminated, truncated, infos = envs.step(actions)
            if isinstance(infos, tuple):
                infos, final_infos = infos

            for i in range(args.num_envs):
                if not dones[i]:
                    # Corrected glucose fallback
                    if isinstance(infos, (list, tuple)) and infos[i] is not None and isinstance(infos[i], dict) and "CGM" in infos[i]:
                        glucose = infos[i]["CGM"]
                    else:
                        glucose = next_obs[i, 0]  # CGM is always first index

                    insulin = float(actions[i][0])
                    logger.log_step(episode, t, glucose, insulin, rewards[i], env_id=i)
                
                    # if episode == 0 and t < 10:  # first few steps
                        # print(f"[Debug] Action: {actions[i][0]}, Applied insulin: {float(actions[i][0])}")

            # Multiple gradient steps per environment step
            for _ in range(gradient_steps_per_env_step):
                agent.update()

            obs = next_obs
            dones |= terminated
            episode_rewards += rewards
            t += 1

        print(f"Episode {episode}: Average Reward = {np.mean(episode_rewards):.2f}")

    plot_from_log()

if __name__ == "__main__":
    main()
