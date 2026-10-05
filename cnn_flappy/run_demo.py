#!/usr/bin/env python
"""One-shot demo: record -> BC -> visualize -> short DQN -> compare."""

from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main():
    py = sys.executable
    run([py, "-m", "cnn_flappy.record_data", "--episodes", "20"])
    run([py, "-m", "cnn_flappy.train_bc", "--epochs", "10"])
    run([py, "-m", "cnn_flappy.visualize_cnn"])
    run(
        [
            py,
            "-m",
            "cnn_flappy.train_rl",
            "--timesteps",
            "8000",
            "--warmup",
            "500",
            "--init-bc",
            "cnn_flappy/checkpoints/bc_cnn.pt",
        ]
    )
    run([py, "-m", "cnn_flappy.compare", "--episodes", "8"])
    run([py, "-m", "cnn_flappy.play", "--model", "cnn_flappy/checkpoints/dqn_cnn.pt", "--episodes", "3"])
    print("\nDemo finished. See cnn_flappy/outputs/ and cnn_flappy/checkpoints/")


if __name__ == "__main__":
    main()
