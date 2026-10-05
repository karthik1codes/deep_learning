"""Showcase CNN: save input frames + conv feature-map grids."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cnn_flappy.cnn_model import FlappyCNN, select_device
from cnn_flappy.env_utils import heuristic_action, make_env


def _grid(maps: np.ndarray, max_maps: int = 16) -> np.ndarray:
    """maps: (C, H, W) -> tiled grayscale image."""
    maps = maps[:max_maps]
    n = maps.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    h, w = maps.shape[1], maps.shape[2]
    canvas = np.zeros((rows * h, cols * w), dtype=np.float32)
    for i, m in enumerate(maps):
        r, c = divmod(i, cols)
        # normalize each map for visibility
        m = m - m.min()
        if m.max() > 0:
            m = m / m.max()
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = m
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize CNN feature maps")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn_flappy/checkpoints/bc_cnn.pt",
    )
    parser.add_argument(
        "--out", type=str, default="cnn_flappy/outputs/cnn_features.png"
    )
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    device = select_device()
    if Path(args.model).exists():
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        model = FlappyCNN(
            in_channels=ckpt.get("in_channels", 4),
            n_actions=ckpt.get("n_actions", 2),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[viz] loaded {args.model}")
    else:
        model = FlappyCNN().to(device)
        print("[viz] no checkpoint — using randomly initialized CNN")

    model.eval()
    env = make_env(render_mode="rgb_array")
    obs, info = env.reset()

    for _ in range(args.steps):
        action = heuristic_action(info)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    x = torch.from_numpy(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, feats = model.forward_with_features(x)
    env.close()

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    action_names = ["do nothing (0)", "flap (1)"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    # stacked input frames
    for i in range(min(4, obs.shape[0])):
        ax = axes[0, 0] if i == 0 else None
    axes[0, 0].imshow(np.concatenate([obs[i] for i in range(obs.shape[0])], axis=1), cmap="gray")
    axes[0, 0].set_title("Input: 4 stacked frames (CNN sees motion)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(_grid(feats["conv1"][0].cpu().numpy()), cmap="viridis")
    axes[0, 1].set_title("Conv1 feature maps\n(edges / bird / pipes)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(_grid(feats["conv2"][0].cpu().numpy()), cmap="viridis")
    axes[0, 2].set_title("Conv2 feature maps\n(mid-level patterns)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(_grid(feats["conv3"][0].cpu().numpy(), max_maps=16), cmap="viridis")
    axes[1, 0].set_title("Conv3 feature maps\n(task-relevant concepts)")
    axes[1, 0].axis("off")

    axes[1, 1].bar(action_names, probs, color=["#4C78A8", "#F58518"])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("CNN output action probabilities")
    axes[1, 1].set_ylabel("softmax(logits)")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0,
        0.5,
        (
            "How the CNN works here\n"
            "─────────────────────\n"
            "1. Raw game pixels → grayscale 84×84\n"
            "2. Stack 4 frames so the net sees velocity\n"
            "3. Conv layers detect bird / pipes / gaps\n"
            "4. Fully-connected layer → action scores\n"
            "5. BC: imitate teacher flaps\n"
            "6. DQN/RL: improve by maximizing reward\n"
            f"\nChosen action: {action_names[int(probs.argmax())]}"
        ),
        family="monospace",
        fontsize=9,
        va="center",
    )

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[viz] saved CNN showcase -> {out}")


if __name__ == "__main__":
    main()
