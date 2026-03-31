import gymnasium as gym                          # CHANGE 1
import numpy as np
from gymnasium import Wrapper                    # CHANGE 1


def make_non_absorbing(observation):
    return np.concatenate([observation, [0.0]], -1)


class AbsorbingStatesWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        low = env.observation_space.low
        high = env.observation_space.high
        self._absorbing_state = np.concatenate([np.zeros_like(low), [1.0]], 0)
        low = np.concatenate([low, [0]], 0)
        high = np.concatenate([high, [1]], 0)

        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        self._done = False
        self._absorbing = False
        self._info = {}
        obs, info = self.env.reset(**kwargs)                   # CHANGE 2
        return make_non_absorbing(obs), info                   # CHANGE 2

    def step(self, action):
        if not self._done:
            observation, reward, terminated, truncated, info = self.env.step(action)  # CHANGE 3
            observation = make_non_absorbing(observation)
            self._done = terminated or truncated               # CHANGE 3
            self._info = info
            # Absorbing states logic: on true termination, return done=False
            # so the caller continues and receives the absorbing state next step.
            # On truncation (time limit), end the episode normally.
            return observation, reward, False, truncated, info # CHANGE 3
        else:
            if not self._absorbing:
                self._absorbing = True
                return self._absorbing_state, 0.0, False, False, self._info  # CHANGE 3
            else:
                return self._absorbing_state, 0.0, True, False, self._info   # CHANGE 3


if __name__ == '__main__':
    env = gym.make('Hopper-v4')                               # CHANGE 4: v2 → v4
    env = AbsorbingStatesWrapper(env)
    obs, info = env.reset()                                   # CHANGE 4

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)  # CHANGE 4
        done = terminated or truncated                        # CHANGE 4
        print(obs, done)