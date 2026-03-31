from typing import Dict

import flax.linen as nn
import gymnasium as gym                              # CHANGE 1
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    successes = None

    for _ in range(num_episodes):
        observation, info = env.reset()              # CHANGE 2
        done = False                                 # CHANGE 2
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, terminated, truncated, info = env.step(action)  # CHANGE 3
            done = terminated or truncated           # CHANGE 3
        for k in stats.keys():
            stats[k].append(info['episode'][k])

        if 'is_success' in info:
            if successes is None:
                successes = 0.0
            successes += info['is_success']

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    return stats