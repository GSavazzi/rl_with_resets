import os

import cv2                                                     # ADD
import gymnasium as gym
import imageio
import numpy as np

from continuous_control.wrappers.common import TimeStep


class VideoRecorder(gym.Wrapper):
    def __init__(self,
                 env: gym.Env,
                 save_folder: str = '',
                 height: int = 128,
                 width: int = 128,
                 fps: int = 30):
        super().__init__(env)

        self.current_episode = 0
        self.save_folder = save_folder
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []

        try:
            os.makedirs(save_folder, exist_ok=True)
        except:
            pass

    def step(self, action: np.ndarray) -> TimeStep:
        frame = self.env.render()                              # FIX: no kwargs in Gymnasium ≥0.26

        if frame is None:
            raise NotImplementedError('Rendering is not implemented.')

        # Resize if the env renders at a different resolution than requested
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height),
                               interpolation=cv2.INTER_AREA)   # ADD

        self.frames.append(frame)

        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if done:
            save_file = os.path.join(self.save_folder,
                                     f'{self.current_episode}.mp4')
            imageio.mimsave(save_file, self.frames, fps=self.fps)
            self.frames = []
            self.current_episode += 1

        return observation, reward, terminated, truncated, info