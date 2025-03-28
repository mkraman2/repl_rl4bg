# pomdp.py

import gym
import numpy as np
from collections import deque

class POMDPWrapper(gym.ObservationWrapper):
    def __init__(self, env, history_length=10):
        super().__init__(env)
        self.history_length = history_length
        self.obs_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim * history_length,),
            dtype=np.float32
        )
        self.history = deque(maxlen=history_length)

    def reset(self):
        obs, info = self.env.reset()
        self.history.clear()
        for _ in range(self.history_length - 1):
            self.history.append(np.zeros(self.obs_dim, dtype=np.float32))
        self.history.append(obs)
        return self._get_obs(), info

    def observation(self, obs):
        self.history.append(obs)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(self.history)