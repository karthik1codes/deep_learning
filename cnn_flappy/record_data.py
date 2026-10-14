"""Record (frame-stack, action) demos using a heuristic teacher policy.

Saves memory-friendly shards (float16 observations only — enough for BC),
then merges into a final demos.npz so Colab/Kaggle do not OOM after long runs.
"""

from __future__ import annotations

import argparse
import gc
import shutil
from pathlib import Path

import numpy as np

from cnn_flappy.env_utils import heuristic_action, make_env, run_episode


def _flush_shard(
    shard_dir: Path,
    shard_idx: int,
    all_obs: list,
    all_actions: list,
    scores: list,
) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"shard_{shard_idx:04d}.npz"
    obs = np.asarray(all_obs, dtype=np.float16)
    actions = np.asarray(all_actions, dtype=np.int64)
    np.savez_compressed(
        path,
        observations=obs,
        actions=actions,
        scores=np.asarray(scores, dtype=np.int64),
    )
    print(
        f"[record] wrote shard {path.name}: "
        f"{len(actions)} steps, obs={obs.shape}, dtype={obs.dtype}"
    )
    return path


def _merge_shards(shard_paths: list[Path], out: Path) -> Path:
    """Merge shards into demos.npz (float16). Returns path usable by train_bc."""
    if not shard_paths:
        raise RuntimeError("no shards to merge — recording produced no data")

    out.parent.mkdir(parents=True, exist_ok=True)
    obs_parts, act_parts, score_parts = [], [], []
    total_steps = 0
    for p in shard_paths:
        with np.load(p) as d:
            obs_parts.append(np.asarray(d["observations"], dtype=np.float16))
            act_parts.append(np.asarray(d["actions"], dtype=np.int64))
            score_parts.append(np.asarray(d["scores"], dtype=np.int64))
            total_steps += int(d["actions"].shape[0])
            print(f"[record] loaded shard {p.name} for merge ({d['actions'].shape[0]} steps)")
        gc.collect()

    observations = np.concatenate(obs_parts, axis=0)
    actions = np.concatenate(act_parts, axis=0)
    scores = np.concatenate(score_parts, axis=0) if score_parts else np.zeros(0, dtype=np.int64)
    del obs_parts, act_parts, score_parts
    gc.collect()

    # Uncompressed .npz is faster/safer on low-RAM hosts than savez_compressed
    np.savez(
        out,
        observations=observations,
        actions=actions,
        scores=scores,
    )
    print(
        f"[record] saved {total_steps} transitions -> {out.resolve()}\n"
        f"         mean score={float(scores.mean()) if len(scores) else 0:.2f} "
        f"obs shape={observations.shape} dtype={observations.dtype}"
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Record Flappy Bird CNN demos")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--out", type=str, default="cnn_flappy/data/demos.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shard-every",
        type=int,
        default=100,
        help="Flush a disk shard every N episodes (avoids OOM on long runs)",
    )
    parser.add_argument(
        "--policy",
        choices=["heuristic", "random"],
        default="heuristic",
        help="Teacher policy for recording state-actions",
    )
    args = parser.parse_args()

    out = Path(args.out).resolve()
    shard_dir = out.parent / (out.stem + "_shards")
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(render_mode="rgb_array")
    env.reset(seed=args.seed)

    all_obs, all_actions = [], []
    ep_scores: list[int] = []
    shard_scores: list[int] = []
    shard_paths: list[Path] = []
    shard_idx = 0

    def policy(obs, info):
        if args.policy == "random":
            return env.action_space.sample()
        return heuristic_action(info)

    for ep in range(args.episodes):
        transitions, total_r, score = run_episode(env, policy)
        ep_scores.append(int(score))
        shard_scores.append(int(score))
        for t in transitions:
            # float16 ASAP to cut RAM ~2x vs float32 lists of large frames
            all_obs.append(np.asarray(t["obs"], dtype=np.float16))
            all_actions.append(int(t["action"]))
        print(
            f"[record] episode {ep + 1}/{args.episodes} "
            f"steps={len(transitions)} reward={total_r:.2f} score={score}"
        )

        if args.shard_every > 0 and (ep + 1) % args.shard_every == 0:
            shard_paths.append(
                _flush_shard(shard_dir, shard_idx, all_obs, all_actions, shard_scores)
            )
            shard_idx += 1
            all_obs, all_actions, shard_scores = [], [], []
            gc.collect()

    env.close()

    if all_obs:
        shard_paths.append(
            _flush_shard(shard_dir, shard_idx, all_obs, all_actions, shard_scores)
        )
        all_obs, all_actions, shard_scores = [], [], []
        gc.collect()

    try:
        _merge_shards(shard_paths, out)
        # Remove shards only after successful merge
        try:
            shutil.rmtree(shard_dir)
        except OSError:
            print(f"[record] warning: could not remove shard dir {shard_dir}")
    except Exception as exc:
        # Fallback: point train_bc at shard directory via a tiny index file
        index = out.with_suffix(".shards.txt")
        index.write_text("\n".join(str(p.resolve()) for p in shard_paths) + "\n")
        print(
            f"[record] WARNING: merge to {out} failed ({exc}).\n"
            f"         Shards kept in {shard_dir}\n"
            f"         Train BC with: --data {shard_dir}"
        )
        # Also copy first-shard style path for discovery
        marker = out.parent / (out.stem + "_SHARDS_DIR.txt")
        marker.write_text(str(shard_dir.resolve()) + "\n")
        raise SystemExit(
            f"[record] demos.npz not written. Use shards: "
            f"python -m cnn_flappy.train_bc --data {shard_dir} ..."
        ) from exc

    if not out.exists():
        raise SystemExit(f"[record] ERROR: expected output missing: {out}")


if __name__ == "__main__":
    main()
