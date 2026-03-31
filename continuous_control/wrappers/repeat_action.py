import gymnasium as gym                              # CHANGE 1
import numpy as np

from continuous_control.wrappers.common import TimeStep


class RepeatAction(gym.Wrapper):
    def __init__(self, env, action_repeat=4):
        super().__init__(env)
        self._action_repeat = action_repeat

    def step(self, action: np.ndarray) -> TimeStep:
        total_reward = 0.0
        done = None
        combined_info = {}

        for _ in range(self._action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)  # CHANGE 2
            done = terminated or truncated                                      # CHANGE 2
            total_reward += reward
            combined_info.update(info)
            if done:
                break

        return obs, total_reward, terminated, truncated, combined_info         # CHANGE 2