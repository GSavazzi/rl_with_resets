from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import RescaleAction, AddRenderObservation

from continuous_control import wrappers
from continuous_control.wrappers import VideoRecorder


def make_env(env_name: str,
             seed: int,
             save_folder: Optional[str] = None,
             add_episode_monitor: bool = True,
             action_repeat: int = 1,
             frame_stack: int = 1,
             from_pixels: bool = False,
             pixels_only: bool = True,
             image_size: int = 84,
             sticky: bool = False,
             gray_scale: bool = False,
             flatten: bool = True) -> gym.Env:

    if env_name in gym.envs.registry:
        camera_id = 0
        # width/height only passed when from_pixels=True.
        # Non-MuJoCo envs (e.g. CartPole) do not accept these kwargs and would crash.
        make_kwargs = {'render_mode': 'rgb_array' if from_pixels else None}
        if from_pixels:
            make_kwargs['width'] = image_size
            make_kwargs['height'] = image_size
        env = gym.make(env_name, **make_kwargs)
    else:
        domain_name, task_name = env_name.split('-')
        camera_id = 2 if domain_name == 'quadruped' else 0
        env = wrappers.DMCEnv(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={'random': seed},
            render_mode='rgb_array' if from_pixels else None,
            # height/width/camera_id stored as instance state in DMCEnv;
            # None when not rendering so dm_control uses its own defaults
            # if render() is ever called unexpectedly.
            height=image_size if from_pixels else None,
            width=image_size if from_pixels else None,
            camera_id=camera_id,
        )

    # Flatten Dict observation spaces before any wrapper inspects obs shape.
    # Must happen early — wrappers like EpisodeMonitor may read observation_space.
    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if add_episode_monitor:
        env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    # RescaleAction requires a Box action space — valid for all DMC and MuJoCo envs.
    env = RescaleAction(env, -1.0, 1.0)

    # VideoRecorder wraps before AddRenderObservation so it records the raw
    # environment render, not the augmented pixel observation.
    if save_folder is not None:
        env = VideoRecorder(env, save_folder=save_folder)

    if from_pixels:
        env = AddRenderObservation(env, render_only=pixels_only, render_key='pixels')

        # TakeKey is only needed when render_only=False (pixels_only=False).
        # In that case AddRenderObservation returns a Dict with 'pixels' mixed in
        # alongside the original obs — TakeKey extracts just the image.
        # When render_only=True, the obs is already a plain Box — no key to extract.
        if not pixels_only:
            env = wrappers.TakeKey(env, take_key='pixels')

        if gray_scale:
            env = wrappers.RGB2Gray(env)
    else:
        env = wrappers.SinglePrecision(env)

    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    if sticky:
        env = wrappers.StickyActionEnv(env)

    # env.seed() removed in Gymnasium v1.0 — seed action/obs spaces directly.
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env