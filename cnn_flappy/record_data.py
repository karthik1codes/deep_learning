"""Record (frame-stack, action) demos using a heuristic teacher policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cnn_flappy.env_utils import heuristic_action, make_env, run_episode


def main():
    parser = argparse.ArgumentParser(description="Record Flappy Bird CNN demos")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--out", type=str, default="cnn_flappy/data/demos.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy",
        choices=["heuristic", "random"],
        default="heuristic",
        help="Teacher policy for recording state-actions",
    )
    args = parser.parse_args()

    env = make_env(render_mode="rgb_array")
    env.reset(seed=args.seed)

    all_obs, all_actions, all_rewards, all_next, all_dones = [], [], [], [], []
    scores = []

    def policy(obs, info):
        if args.policy == "random":
            return env.action_space.sample()
        return heuristic_action(info)

    for ep in range(args.episodes):
        transitions, total_r, score = run_episode(env, policy)
        scores.append(score)
        for t in transitions:
            all_obs.append(t["obs"])
            all_actions.append(t["action"])
            all_rewards.append(t["reward"])
            all_next.append(t["next_obs"])
            all_dones.append(t["done"])
        print(
            f"[record] episode {ep + 1}/{args.episodes} "
            f"steps={len(transitions)} reward={total_r:.2f} score={score}"
        )

    env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        observations=np.asarray(all_obs, dtype=np.float32),
        actions=np.asarray(all_actions, dtype=np.int64),
        rewards=np.asarray(all_rewards, dtype=np.float32),
        next_observations=np.asarray(all_next, dtype=np.float32),
        dones=np.asarray(all_dones, dtype=np.bool_),
        scores=np.asarray(scores, dtype=np.int64),
    )
    print(
        f"[record] saved {len(all_actions)} transitions -> {out}\n"
        f"         mean score={np.mean(scores):.2f} "
        f"obs shape={np.asarray(all_obs).shape}"
    )


if __name__ == "__main__":
    main()
