"""Play Flappy Bird with a trained CNN (BC or DQN) and optionally show window."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cnn_flappy.cnn_model import FlappyCNN, select_device
from cnn_flappy.env_utils import make_env, run_episode


def load_model(path: str, device: torch.device) -> FlappyCNN:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = FlappyCNN(
        in_channels=ckpt.get("in_channels", 4),
        n_actions=ckpt.get("n_actions", 2),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Play with trained CNN")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn_flappy/checkpoints/dqn_cnn.pt",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open pygame window (slower)",
    )
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(
            f"Model not found: {args.model}\n"
            "Train first: python -m cnn_flappy.train_bc  or  python -m cnn_flappy.train_rl"
        )

    device = select_device()
    model = load_model(args.model, device)
    render_mode = "human" if args.render else "rgb_array"
    env = make_env(render_mode=render_mode)

    def policy(obs, info):
        with torch.no_grad():
            q = model(torch.from_numpy(obs).unsqueeze(0).to(device))
            return int(q.argmax(1).item())

    scores, rewards = [], []
    for ep in range(args.episodes):
        transitions, total_r, score = run_episode(env, policy)
        scores.append(score)
        rewards.append(total_r)
        print(
            f"[play] episode {ep + 1}/{args.episodes} "
            f"steps={len(transitions)} reward={total_r:.2f} score={score}"
        )

    env.close()
    print(
        f"[play] mean score={np.mean(scores):.2f} "
        f"max={np.max(scores)} mean reward={np.mean(rewards):.2f}"
    )


if __name__ == "__main__":
    main()
