"""Shared live training state + DQN loop for pygame / Next.js dashboards."""

from __future__ import annotations

import base64
import io
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cnn_flappy.cnn_model import FlappyCNN, select_device
from cnn_flappy.env_utils import RenderFixWrapper, make_env


@dataclass
class LiveState:
    running: bool = False
    status: str = "idle"
    step: int = 0
    episode: int = 0
    epsilon: float = 1.0
    last_loss: float = 0.0
    episode_reward: float = 0.0
    last_score: int = 0
    mean_score_10: float = 0.0
    max_score: int = 0
    fps: float = 0.0
    action: int = 0
    message: str = "Ready"
    scores: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    frame_jpeg_b64: str = ""
    updated_at: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "status": self.status,
            "step": self.step,
            "episode": self.episode,
            "epsilon": round(self.epsilon, 4),
            "last_loss": round(self.last_loss, 5),
            "episode_reward": round(self.episode_reward, 3),
            "last_score": self.last_score,
            "mean_score_10": round(self.mean_score_10, 3),
            "max_score": self.max_score,
            "fps": round(self.fps, 1),
            "action": self.action,
            "message": self.message,
            "scores": self.scores[-100:],
            "losses": self.losses[-200:],
            "rewards": self.rewards[-100:],
            "frame_jpeg_b64": self.frame_jpeg_b64,
            "updated_at": self.updated_at,
        }


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.stack(obs),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_obs),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


def _find_render_fix(env) -> Optional[RenderFixWrapper]:
    cur = env
    while cur is not None:
        if isinstance(cur, RenderFixWrapper):
            return cur
        cur = getattr(cur, "env", None)
    return None


def rgb_to_jpeg_b64(rgb: np.ndarray, max_width: int = 400) -> str:
    if rgb is None or rgb.size == 0:
        return ""
    img = Image.fromarray(rgb.astype(np.uint8))
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class LiveDQNTrainer:
    """Background DQN trainer that publishes frames + metrics."""

    def __init__(self):
        self.state = LiveState()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return self.state.snapshot()

    def _update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self.state, k, v)
            self.state.updated_at = time.time()

    def start(
        self,
        timesteps: int = 20_000,
        warmup: int = 500,
        batch_size: int = 32,
        init_bc: str = "cnn_flappy/checkpoints/bc_cnn.pt",
        out: str = "cnn_flappy/checkpoints/dqn_cnn.pt",
        frame_every: int = 2,
    ) -> bool:
        with self._lock:
            if self.state.running:
                return False
            self.state = LiveState(running=True, status="starting", message="Starting DQN…")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            kwargs={
                "timesteps": timesteps,
                "warmup": warmup,
                "batch_size": batch_size,
                "init_bc": init_bc,
                "out": out,
                "frame_every": frame_every,
            },
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        self._update(message="Stop requested…", status="stopping")

    def _run(
        self,
        timesteps: int,
        warmup: int,
        batch_size: int,
        init_bc: str,
        out: str,
        frame_every: int,
    ):
        try:
            env = make_env(render_mode="rgb_array")
            render_fix = _find_render_fix(env)
            obs, info = env.reset()
            in_channels = obs.shape[0]
            n_actions = env.action_space.n

            device = select_device()
            policy_net = FlappyCNN(in_channels, n_actions).to(device)
            target_net = FlappyCNN(in_channels, n_actions).to(device)

            if init_bc and Path(init_bc).exists():
                ckpt = torch.load(init_bc, map_location=device, weights_only=False)
                policy_net.load_state_dict(ckpt["model_state"])
                self._update(message=f"Warm-started from {init_bc}")

            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
            opt = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
            buffer = ReplayBuffer()

            episode_scores: list[int] = []
            ep_reward = 0.0
            t0 = time.time()
            steps_window = 0

            eps_start, eps_end, eps_decay = 1.0, 0.05, max(1, timesteps // 2)
            gamma = 0.99
            target_update = 500

            self._update(status="training", message="Training CNN-DQN")

            for t in range(1, timesteps + 1):
                if self._stop.is_set():
                    break

                eps = eps_end + (eps_start - eps_end) * max(
                    0.0, (eps_decay - t) / eps_decay
                )
                if random.random() < eps:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        q = policy_net(
                            torch.from_numpy(obs).unsqueeze(0).to(device)
                        )
                        action = int(q.argmax(1).item())

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                buffer.push(obs, action, float(reward), next_obs, done)
                ep_reward += float(reward)
                obs = next_obs
                steps_window += 1

                loss_val = None
                if len(buffer) >= warmup:
                    o, a, r, no, d = buffer.sample(batch_size)
                    o_t = torch.from_numpy(o).to(device)
                    a_t = torch.from_numpy(a).to(device)
                    r_t = torch.from_numpy(r).to(device)
                    no_t = torch.from_numpy(no).to(device)
                    d_t = torch.from_numpy(d).to(device)
                    q = policy_net(o_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_q = target_net(no_t).max(1).values
                        target = r_t + gamma * next_q * (1.0 - d_t)
                    loss = F.smooth_l1_loss(q, target)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                    opt.step()
                    loss_val = float(loss.item())
                    if t % target_update == 0:
                        target_net.load_state_dict(policy_net.state_dict())

                frame_b64 = None
                if t % frame_every == 0 and render_fix is not None and render_fix.last_rgb is not None:
                    frame_b64 = rgb_to_jpeg_b64(render_fix.last_rgb)

                now = time.time()
                if now - t0 >= 1.0:
                    fps = steps_window / (now - t0)
                    t0 = now
                    steps_window = 0
                else:
                    fps = self.state.fps

                with self._lock:
                    self.state.running = True
                    self.state.status = "training"
                    self.state.step = t
                    self.state.epsilon = eps
                    self.state.episode_reward = ep_reward
                    self.state.action = int(action)
                    self.state.fps = fps
                    if loss_val is not None:
                        self.state.last_loss = loss_val
                        self.state.losses.append(loss_val)
                        if len(self.state.losses) > 2000:
                            self.state.losses = self.state.losses[-2000:]
                    if frame_b64:
                        self.state.frame_jpeg_b64 = frame_b64
                    self.state.updated_at = time.time()

                if done:
                    score = int(info.get("score", 0))
                    episode_scores.append(score)
                    with self._lock:
                        self.state.episode += 1
                        self.state.last_score = score
                        self.state.max_score = max(self.state.max_score, score)
                        self.state.mean_score_10 = float(
                            np.mean(episode_scores[-10:])
                        )
                        self.state.scores.append(float(score))
                        self.state.rewards.append(ep_reward)
                        if len(self.state.scores) > 500:
                            self.state.scores = self.state.scores[-500:]
                            self.state.rewards = self.state.rewards[-500:]
                        self.state.message = (
                            f"Episode {self.state.episode} score={score}"
                        )
                    ep_reward = 0.0
                    obs, info = env.reset()

            Path(out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": policy_net.state_dict(),
                    "in_channels": in_channels,
                    "n_actions": n_actions,
                    "kind": "dqn",
                },
                out,
            )
            env.close()
            self._update(
                running=False,
                status="finished",
                message=f"Saved checkpoint → {out}",
            )
        except Exception as exc:  # noqa: BLE001
            self._update(
                running=False,
                status="error",
                message=f"Error: {exc}",
            )


# singleton used by API + pygame
trainer = LiveDQNTrainer()
