import time

import gymnasium as gym                              # CHANGE 1
import numpy as np

from continuous_control.wrappers.common import TimeStep


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray) -> TimeStep:
        observation, reward, terminated, truncated, info = self.env.step(action)  # CHANGE 2
        done = terminated or truncated                    # CHANGE 2

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if done:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self, 'get_normalized_score'):
                info['episode']['return'] = self.get_normalized_score(
                    info['episode']['return']) * 100.0

        return observation, reward, terminated, truncated, info  # CHANGE 2

    def reset(self, seed=None, options=None) -> tuple:   # CHANGE 3
        self._reset_stats()
        return self.env.reset(seed=seed, options=options) # CHANGE 3