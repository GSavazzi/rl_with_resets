import collections

import gymnasium as gym                               # CHANGE 1
import numpy as np
from gymnasium.spaces import Box                      # CHANGE 1


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack: int, stack_axis=-1):
        super().__init__(env)
        self._num_stack = num_stack
        self._stack_axis = stack_axis

        self._frames = collections.deque([], maxlen=num_stack)

        low = np.repeat(self.observation_space.low, num_stack, axis=stack_axis)
        high = np.repeat(self.observation_space.high, num_stack, axis=stack_axis)
        self.observation_space = Box(low=low,
                                     high=high,
                                     dtype=self.observation_space.dtype)

    def reset(self, seed=None, options=None):          # CHANGE 2
        obs, info = self.env.reset(seed=seed, options=options)  # CHANGE 2
        for _ in range(self._num_stack):
            self._frames.append(obs)
        return self._get_obs(), info                   # CHANGE 2

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)  # CHANGE 3
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info        # CHANGE 3

    def _get_obs(self):
        assert len(self._frames) == self._num_stack
        return np.concatenate(list(self._frames), axis=self._stack_axis)