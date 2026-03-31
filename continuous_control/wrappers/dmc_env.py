# Taken from
# https://github.com/denisyarats/dmc2gym
# and modified to exclude duplicated code.

import copy
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
from dm_control import suite
from dm_env import specs
from gymnasium import core, spaces

from continuous_control.wrappers.common import TimeStep


def dmc_spec2gym_space(spec):
    if isinstance(spec, OrderedDict):
        spec = copy.copy(spec)
        for k, v in spec.items():
            spec[k] = dmc_spec2gym_space(v)
        return spaces.Dict(spec)
    elif isinstance(spec, specs.BoundedArray):
        return spaces.Box(low=spec.minimum,
                          high=spec.maximum,
                          shape=spec.shape,
                          dtype=spec.dtype)
    elif isinstance(spec, specs.Array):
        return spaces.Box(low=-float('inf'),
                          high=float('inf'),
                          shape=spec.shape,
                          dtype=spec.dtype)
    else:
        raise NotImplementedError


class DMCEnv(core.Env):
    # Declare supported render modes for Gymnasium v1.0 compliance.
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self,
                 domain_name: str,
                 task_name: str,
                 task_kwargs: Optional[Dict] = None,   # avoid mutable default {}
                 environment_kwargs=None,
                 render_mode: Optional[str] = None,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 camera_id: int = 0):

        # task_kwargs mutable default guard
        if task_kwargs is None:
            task_kwargs = {}
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'

        # Store render config BEFORE suite.load() so that if anything inside
        # suite.load() triggers __getattr__ while self._env is still unset,
        # the guard in __getattr__ raises AttributeError cleanly instead of
        # recursing infinitely trying to resolve self._env.
        self.render_mode = render_mode
        self._height = height       # None = let dm_control use its default (240)
        self._width = width         # None = let dm_control use its default (320)
        self._camera_id = camera_id

        self._env = suite.load(domain_name=domain_name,
                               task_name=task_name,
                               task_kwargs=task_kwargs,
                               environment_kwargs=environment_kwargs)

        self.action_space = dmc_spec2gym_space(self._env.action_spec())
        self.observation_space = dmc_spec2gym_space(self._env.observation_spec())

    def __getattr__(self, name):
        # Guard: _env may not exist yet during __init__ (e.g. if an exception
        # fires before self._env = suite.load(...)). Without this check,
        # Python would call __getattr__('_env') which calls __getattr__('_env')
        # again, causing infinite recursion and a RuntimeError.
        if '_env' not in self.__dict__:
            raise AttributeError(name)
        return getattr(self._env, name)

    def step(self, action: np.ndarray) -> TimeStep:
        assert self.action_space.contains(action)

        time_step = self._env.step(action)
        reward = time_step.reward or 0

        # dm_control encodes episode end via discount:
        #   discount == 0.0  → terminal state (e.g. fell over) → terminated=True
        #   discount == 1.0  → time limit hit                  → truncated=True
        terminated = time_step.last() and time_step.discount == 0.0
        truncated = time_step.last() and time_step.discount == 1.0

        return time_step.observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        # dm_control seeding is handled via task_kwargs["random"] at construction;
        # the seed parameter here is accepted for Gymnasium API compliance only.
        time_step = self._env.reset()
        return time_step.observation, {}

    def render(self):
        # Gymnasium v1.0: render() takes no arguments.
        # Size and camera are controlled via constructor params stored on self.
        if self.render_mode != 'rgb_array':
            raise ValueError(
                f'render_mode must be "rgb_array", got {self.render_mode!r}. '
                'Pass render_mode="rgb_array" when constructing DMCEnv.'
            )
        # Only override dm_control defaults when explicitly set at construction.
        # Passing None to physics.render() would cause a MuJoCo type error.
        render_kwargs = {'camera_id': self._camera_id}
        if self._height is not None:
            render_kwargs['height'] = self._height
        if self._width is not None:
            render_kwargs['width'] = self._width
        return self._env.physics.render(**render_kwargs)