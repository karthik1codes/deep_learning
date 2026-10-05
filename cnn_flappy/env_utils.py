"""Environment helpers: fix rgb_array frames, resize for CNN, frame stacking."""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional

import flappy_bird_env  # noqa: F401 — registers FlappyBird-v0
import gymnasium as gym
import numpy as np
from gymnasium import spaces


FRAME_SIZE = 84
STACK_SIZE = 4


class RenderFixWrapper(gym.Wrapper):
    """flappy-bird-env only redraws for human mode; force rgb_array redraws."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.last_rgb: Optional[np.ndarray] = None

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        frame = self.env.render()
        obs = frame if frame is not None else self.env.unwrapped.observation
        self.last_rgb = np.asarray(obs).copy()
        return obs, info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        frame = self.env.render()
        obs = frame if frame is not None else self.env.unwrapped.observation
        self.last_rgb = np.asarray(obs).copy()
        return obs, reward, terminated, truncated, info


class PreprocessFrame(gym.ObservationWrapper):
    """RGB (H, W, 3) -> grayscale float32 (1, 84, 84) in [0, 1]."""

    def __init__(self, env: gym.Env, size: int = FRAME_SIZE):
        super().__init__(env)
        self.size = size
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1, size, size), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # obs is (H, W, 3) uint8
        if obs.ndim == 3 and obs.shape[-1] == 3:
            gray = obs.mean(axis=-1)
        else:
            gray = obs.astype(np.float32)
        # nearest-neighbor resize without cv2 dependency
        h, w = gray.shape
        ys = (np.linspace(0, h - 1, self.size)).astype(np.int32)
        xs = (np.linspace(0, w - 1, self.size)).astype(np.int32)
        small = gray[ys][:, xs]
        return (small.astype(np.float32) / 255.0)[None, ...]


class FrameStack(gym.Wrapper):
    """Stack last N grayscale frames -> (N, H, W)."""

    def __init__(self, env: gym.Env, n: int = STACK_SIZE):
        super().__init__(env)
        self.n = n
        self.frames: deque[np.ndarray] = deque(maxlen=n)
        c, h, w = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n * c, h, w), dtype=np.float32
        )

    def _stack(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n):
            self.frames.append(obs)
        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._stack(), reward, terminated, truncated, info


def make_env(
    render_mode: Optional[str] = "rgb_array",
    frame_size: int = FRAME_SIZE,
    stack_size: int = STACK_SIZE,
) -> gym.Env:
    """Build CNN-ready Flappy Bird env."""
    env = gym.make("FlappyBird-v0", render_mode=render_mode or "rgb_array")
    env = RenderFixWrapper(env)
    env = PreprocessFrame(env, size=frame_size)
    env = FrameStack(env, n=stack_size)
    return env


def heuristic_action(info: dict) -> int:
    """Simple rule: flap if bird is below the next pipe gap center."""
    bird = info["bird"]
    bird_y = float(bird["y"])
    bird_x = float(bird["x"])
    pipes = info.get("pipes") or []
    ahead = [p for p in pipes if p["x"] + 50 >= bird_x]
    if not ahead:
        return 1 if bird_y > 400 else 0
    nxt = min(ahead, key=lambda p: p["x"])
    # gap runs from height to height+200; center ≈ height + 100
    gap_center = float(nxt["height"]) + 100.0
    return 1 if bird_y > gap_center - 10 else 0


def run_episode(
    env: gym.Env,
    policy: Callable,
    max_steps: int = 2000,
) -> tuple[list, float, int]:
    """
    Run one episode.
    policy(obs, info) -> action
    Returns (transitions, total_reward, score).
    Each transition: dict with obs, action, reward, next_obs, done, info.
    """
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    transitions = []
    steps = 0

    while not done and steps < max_steps:
        action = int(policy(obs, info))
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = bool(terminated or truncated)
        transitions.append(
            {
                "obs": obs,
                "action": action,
                "reward": float(reward),
                "next_obs": next_obs,
                "done": done,
            }
        )
        total_reward += float(reward)
        obs, info = next_obs, next_info
        steps += 1

    score = int(info.get("score", 0)) if info else 0
    return transitions, total_reward, score
