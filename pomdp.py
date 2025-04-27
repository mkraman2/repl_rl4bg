# pomdp.py

import gymnasium as gym
import numpy as np

class POMDPWrapper(gym.Wrapper):
    def __init__(self, env, history_length=10):
        super().__init__(env)
        self.history_length = history_length
        self.feature_dim = env.observation_space.shape[0]

        # New observation space: (feature_dim * history_length,)
        obs_low = np.tile(env.observation_space.low, history_length)
        obs_high = np.tile(env.observation_space.high, history_length)

        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        self.action_space = env.action_space

        # Initialize history buffer
        self.history = np.zeros((self.history_length, self.feature_dim), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Back-propagate: Fill history with the first real observation
        self.history = np.tile(obs, (self.history_length, 1))

        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Shift history and append the new observation
        self.history = np.roll(self.history, shift=-1, axis=0)
        self.history[-1] = obs

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return self.history.flatten()
