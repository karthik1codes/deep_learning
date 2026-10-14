"""DQN self-improvement: CNN Q-network learns by playing Flappy Bird.

Supports mid-training PyTorch checkpoints so Colab/Kaggle sessions can resume.
Replay buffer is not serialized (too large); weights/optimizer/step resume.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from cnn_flappy.cnn_model import FlappyCNN, select_device
from cnn_flappy.env_utils import make_env
from cnn_flappy.numpy_compat import ensure_numpy_core_alias


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


def _ckpt_path(out: Path) -> Path:
    return out.with_name(out.stem + "_train_ckpt.pt")


def save_train_checkpoint(
    path: Path,
    *,
    policy_net,
    target_net,
    opt,
    step: int,
    episode_rewards,
    episode_scores,
    losses,
    in_channels: int,
    n_actions: int,
    args_dict: dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "dqn_train",
            "step": step,
            "policy_state": policy_net.state_dict(),
            "target_state": target_net.state_dict(),
            "optimizer_state": opt.state_dict(),
            "episode_rewards": episode_rewards,
            "episode_scores": episode_scores,
            "losses": losses[-5000:],  # keep file small
            "in_channels": in_channels,
            "n_actions": n_actions,
            "args": args_dict,
        },
        path,
    )
    print(f"[dqn] checkpoint saved -> {path} (step {step})")


def save_final_model(out: Path, policy_net, in_channels: int, n_actions: int, episode_scores):
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": policy_net.state_dict(),
            "in_channels": in_channels,
            "n_actions": n_actions,
            "kind": "dqn",
            "episode_scores": episode_scores,
        },
        out,
    )


def write_plots(plot_path: Path, episode_scores, losses, timesteps: int):
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    if episode_scores:
        window = max(1, min(20, len(episode_scores) // 5 or 1))
        smooth = np.convolve(episode_scores, np.ones(window) / window, mode="valid")
        plt.plot(episode_scores, alpha=0.35, label="raw")
        plt.plot(range(window - 1, len(episode_scores)), smooth, label="smoothed")
        plt.legend()
    plt.title("DQN episode score (self-improvement)")
    plt.xlabel("episode")
    plt.ylabel("pipes passed")
    plt.subplot(1, 2, 2)
    if losses:
        ls = losses[:: max(1, len(losses) // 200)]
        plt.plot(ls)
    plt.title("DQN TD loss")
    plt.xlabel("updates (subsampled)")
    plt.ylabel("SmoothL1")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()

    summary = {
        "timesteps": timesteps,
        "episodes": len(episode_scores),
        "mean_score_last_20": float(np.mean(episode_scores[-20:])) if episode_scores else 0.0,
        "max_score": int(max(episode_scores)) if episode_scores else 0,
    }
    plot_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train CNN-DQN on Flappy Bird")
    parser.add_argument("--timesteps", type=int, default=25_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=15_000)
    parser.add_argument(
        "--init-bc",
        type=str,
        default="",
        help="Optional BC checkpoint to warm-start the Q-network",
    )
    parser.add_argument("--out", type=str, default="cnn_flappy/checkpoints/dqn_cnn.pt")
    parser.add_argument("--plot", type=str, default="cnn_flappy/outputs/dqn_progress.png")
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=10_000,
        help="Save resumable training checkpoint every N steps (0=disable mid saves)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <out>_train_ckpt.pt if present",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Explicit training checkpoint path to resume from",
    )
    args = parser.parse_args()

    env = make_env(render_mode="rgb_array")
    obs, _ = env.reset()
    in_channels = obs.shape[0]
    n_actions = env.action_space.n

    device = select_device()
    policy_net = FlappyCNN(in_channels, n_actions).to(device)
    target_net = FlappyCNN(in_channels, n_actions).to(device)

    out = Path(args.out)
    train_ckpt = Path(args.resume_from) if args.resume_from else _ckpt_path(out)
    start_t = 1
    episode_rewards, episode_scores, losses = [], [], []
    resumed = False

    if args.resume or args.resume_from:
        if train_ckpt.exists():
            ensure_numpy_core_alias()
            ckpt = torch.load(train_ckpt, map_location=device, weights_only=False)
            policy_net.load_state_dict(ckpt["policy_state"])
            target_net.load_state_dict(ckpt.get("target_state", ckpt["policy_state"]))
            opt = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
            if "optimizer_state" in ckpt:
                opt.load_state_dict(ckpt["optimizer_state"])
            start_t = int(ckpt.get("step", 0)) + 1
            episode_rewards = list(ckpt.get("episode_rewards", []))
            episode_scores = list(ckpt.get("episode_scores", []))
            losses = list(ckpt.get("losses", []))
            resumed = True
            print(
                f"[dqn] resumed from {train_ckpt} at step {start_t}/{args.timesteps} "
                f"(episodes so far={len(episode_scores)})"
            )
        else:
            print(f"[dqn] no resume checkpoint at {train_ckpt}; training from scratch")
            opt = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    else:
        opt = torch.optim.Adam(policy_net.parameters(), lr=args.lr)

    if not resumed and args.init_bc and Path(args.init_bc).exists():
        ensure_numpy_core_alias()
        ckpt = torch.load(args.init_bc, map_location=device, weights_only=False)
        policy_net.load_state_dict(ckpt["model_state"])
        print(f"[dqn] warm-started from BC checkpoint {args.init_bc}")

    if not resumed:
        target_net.load_state_dict(policy_net.state_dict())

    target_net.eval()
    buffer = ReplayBuffer()

    ep_reward, ep_score = 0.0, 0
    obs, info = env.reset()
    args_dict = vars(args)

    if start_t > args.timesteps:
        print(f"[dqn] already finished ({start_t - 1} >= {args.timesteps}); nothing to do")
        env.close()
        save_final_model(out, policy_net, in_channels, n_actions, episode_scores)
        summary = write_plots(Path(args.plot), episode_scores, losses, args.timesteps)
        print(f"[dqn] saved model -> {out}")
        print(f"[dqn] summary: {summary}")
        return

    for t in range(start_t, args.timesteps + 1):
        eps = args.eps_end + (args.eps_start - args.eps_end) * max(
            0.0, (args.eps_decay - t) / args.eps_decay
        )
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q = policy_net(torch.from_numpy(obs).unsqueeze(0).to(device))
                action = int(q.argmax(1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        buffer.push(obs, action, float(reward), next_obs, done)
        ep_reward += float(reward)
        obs = next_obs

        if done:
            ep_score = int(info.get("score", 0))
            episode_rewards.append(ep_reward)
            episode_scores.append(ep_score)
            if len(episode_scores) % 5 == 0:
                print(
                    f"[dqn] step={t} episodes={len(episode_scores)} "
                    f"last_score={ep_score} mean10={np.mean(episode_scores[-10:]):.2f} "
                    f"eps={eps:.2f}"
                )
            ep_reward = 0.0
            obs, info = env.reset()

        if len(buffer) >= args.warmup:
            o, a, r, no, d = buffer.sample(args.batch_size)
            o_t = torch.from_numpy(o).to(device)
            a_t = torch.from_numpy(a).to(device)
            r_t = torch.from_numpy(r).to(device)
            no_t = torch.from_numpy(no).to(device)
            d_t = torch.from_numpy(d).to(device)

            q = policy_net(o_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target_net(no_t).max(1).values
                target = r_t + args.gamma * next_q * (1.0 - d_t)
            loss = F.smooth_l1_loss(q, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
            opt.step()
            losses.append(float(loss.item()))

            if t % args.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if t % 5000 == 0:
            print(f"[dqn] progress {t}/{args.timesteps}")

        if args.ckpt_every > 0 and (t % args.ckpt_every == 0 or t == args.timesteps):
            save_final_model(out, policy_net, in_channels, n_actions, episode_scores)
            save_train_checkpoint(
                train_ckpt,
                policy_net=policy_net,
                target_net=target_net,
                opt=opt,
                step=t,
                episode_rewards=episode_rewards,
                episode_scores=episode_scores,
                losses=losses,
                in_channels=in_channels,
                n_actions=n_actions,
                args_dict=args_dict,
            )

    env.close()

    save_final_model(out, policy_net, in_channels, n_actions, episode_scores)
    save_train_checkpoint(
        train_ckpt,
        policy_net=policy_net,
        target_net=target_net,
        opt=opt,
        step=args.timesteps,
        episode_rewards=episode_rewards,
        episode_scores=episode_scores,
        losses=losses,
        in_channels=in_channels,
        n_actions=n_actions,
        args_dict=args_dict,
    )
    summary = write_plots(Path(args.plot), episode_scores, losses, args.timesteps)
    print(f"[dqn] saved model -> {out}")
    print(f"[dqn] saved plot  -> {args.plot}")
    print(f"[dqn] summary: {summary}")
    print(f"[dqn] resume later with: --resume --out {out} --timesteps {args.timesteps}")


if __name__ == "__main__":
    main()
