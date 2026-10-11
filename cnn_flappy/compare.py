"""Compare random / heuristic / BC CNN / DQN CNN on Flappy Bird."""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from cnn_flappy.cnn_model import FlappyCNN, select_device
from cnn_flappy.env_utils import heuristic_action, make_env, run_episode

DEFAULT_BC = "cnn_flappy/checkpoints/bc_cnn.pt"
DEFAULT_DQN = "cnn_flappy/checkpoints/dqn_cnn.pt"


def eval_policy(name, policy, episodes: int):
    env = make_env(render_mode="rgb_array")
    scores, rewards = [], []
    for _ in range(episodes):
        _, total_r, score = run_episode(env, policy)
        scores.append(int(score))
        rewards.append(float(total_r))
    env.close()
    summary = {
        "name": name,
        "scores": scores,
        "rewards": rewards,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "max_score": int(np.max(scores)) if scores else 0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "episodes": len(scores),
    }
    print(
        f"[compare] {name:12s} mean_score={summary['mean_score']:.2f} "
        f"max={summary['max_score']} mean_reward={summary['mean_reward']:.2f}"
    )
    return summary


def cnn_policy(model, device):
    def policy(obs, info):
        with torch.no_grad():
            q = model(torch.from_numpy(obs).unsqueeze(0).to(device))
            return int(q.argmax(1).item())

    return policy


def maybe_load(path: str, device):
    if not Path(path).exists():
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = FlappyCNN(
        in_channels=ckpt.get("in_channels", 4),
        n_actions=ckpt.get("n_actions", 2),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def model_status(
    bc_path: str = DEFAULT_BC,
    dqn_path: str = DEFAULT_DQN,
) -> dict[str, Any]:
    return {
        "bc": {"path": bc_path, "exists": Path(bc_path).exists()},
        "dqn": {"path": dqn_path, "exists": Path(dqn_path).exists()},
    }


def _chart_b64(results: list[dict[str, Any]]) -> str:
    """Thread-safe bar chart via Pillow (no Matplotlib/Tk)."""
    if not results:
        return ""

    width, height = 720, 380
    margin_l, margin_r, margin_t, margin_b = 70, 30, 50, 70
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_title = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        font_title = font

    colors = {
        "random": (158, 202, 225),
        "heuristic": (161, 217, 155),
        "BC-CNN": (253, 174, 107),
        "DQN-CNN": (230, 85, 13),
    }

    means = [float(r["mean_score"]) for r in results]
    max_mean = max(means) if means else 1.0
    if max_mean <= 0:
        max_mean = 1.0

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    n = len(results)
    gap = 20
    bar_w = max(20, (plot_w - gap * (n + 1)) // max(n, 1))

    draw.text((margin_l, 16), "BC-CNN vs DQN-CNN (and baselines)", fill=(30, 30, 30), font=font_title)
    draw.text((12, margin_t + plot_h // 2), "Mean\nscore", fill=(80, 80, 80), font=font)

    # axes
    draw.line(
        [(margin_l, margin_t), (margin_l, margin_t + plot_h), (margin_l + plot_w, margin_t + plot_h)],
        fill=(60, 60, 60),
        width=2,
    )

    for i, r in enumerate(results):
        x0 = margin_l + gap + i * (bar_w + gap)
        bar_h = int((r["mean_score"] / max_mean) * (plot_h - 10))
        y0 = margin_t + plot_h - bar_h
        color = colors.get(r["name"], (120, 120, 120))
        draw.rectangle([x0, y0, x0 + bar_w, margin_t + plot_h], fill=color)
        label = f"{r['mean_score']:.2f}"
        draw.text((x0 + 4, max(margin_t, y0 - 18)), label, fill=(20, 20, 20), font=font)
        draw.text((x0, margin_t + plot_h + 10), r["name"][:10], fill=(40, 40, 40), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_comparison(
    episodes: int = 10,
    bc_path: str = DEFAULT_BC,
    dqn_path: str = DEFAULT_DQN,
    include_baselines: bool = True,
    save_plot: Optional[str] = "cnn_flappy/outputs/compare_agents.png",
) -> dict[str, Any]:
    """Evaluate agents and return JSON-serializable comparison analytics."""
    device = select_device()
    results: list[dict[str, Any]] = []
    skipped: list[str] = []

    if include_baselines:
        results.append(
            eval_policy(
                "random",
                lambda obs, info: int(np.random.randint(0, 2)),
                episodes,
            )
        )
        results.append(
            eval_policy(
                "heuristic",
                lambda obs, info: heuristic_action(info),
                episodes,
            )
        )

    bc = maybe_load(bc_path, device)
    if bc is not None:
        results.append(eval_policy("BC-CNN", cnn_policy(bc, device), episodes))
    else:
        skipped.append(f"BC missing: {bc_path}")

    dqn = maybe_load(dqn_path, device)
    if dqn is not None:
        results.append(eval_policy("DQN-CNN", cnn_policy(dqn, device), episodes))
    else:
        skipped.append(f"DQN missing: {dqn_path}")

    chart = _chart_b64(results) if results else ""
    if save_plot and chart:
        out = Path(save_plot)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(chart))

    bc_row = next((r for r in results if r["name"] == "BC-CNN"), None)
    dqn_row = next((r for r in results if r["name"] == "DQN-CNN"), None)
    winner = None
    if bc_row and dqn_row:
        if dqn_row["mean_score"] > bc_row["mean_score"]:
            winner = "DQN-CNN"
        elif bc_row["mean_score"] > dqn_row["mean_score"]:
            winner = "BC-CNN"
        else:
            winner = "tie"

    return {
        "episodes": episodes,
        "models": model_status(bc_path, dqn_path),
        "results": results,
        "skipped": skipped,
        "winner": winner,
        "chart_png_b64": chart,
        "bc_vs_dqn": {
            "bc_mean_score": bc_row["mean_score"] if bc_row else None,
            "dqn_mean_score": dqn_row["mean_score"] if dqn_row else None,
            "bc_max_score": bc_row["max_score"] if bc_row else None,
            "dqn_max_score": dqn_row["max_score"] if dqn_row else None,
            "bc_mean_reward": bc_row["mean_reward"] if bc_row else None,
            "dqn_mean_reward": dqn_row["mean_reward"] if dqn_row else None,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--bc", type=str, default=DEFAULT_BC)
    parser.add_argument("--dqn", type=str, default=DEFAULT_DQN)
    parser.add_argument(
        "--plot", type=str, default="cnn_flappy/outputs/compare_agents.png"
    )
    args = parser.parse_args()

    payload = run_comparison(
        episodes=args.episodes,
        bc_path=args.bc,
        dqn_path=args.dqn,
        save_plot=args.plot,
    )
    print(f"[compare] winner={payload['winner']}")
    print(f"[compare] saved {args.plot}")


if __name__ == "__main__":
    main()
