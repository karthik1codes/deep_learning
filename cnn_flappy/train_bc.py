"""Behavioral cloning: train CNN to imitate recorded (state, action) pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cnn_flappy.cnn_model import FlappyCNN, select_device


def main():
    parser = argparse.ArgumentParser(description="Train CNN via behavioral cloning")
    parser.add_argument("--data", type=str, default="cnn_flappy/data/demos.npz")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--out", type=str, default="cnn_flappy/checkpoints/bc_cnn.pt"
    )
    parser.add_argument(
        "--plot", type=str, default="cnn_flappy/outputs/bc_loss.png"
    )
    args = parser.parse_args()

    data = np.load(args.data)
    obs = data["observations"]
    actions = data["actions"]
    print(f"[bc] loaded {len(actions)} samples, obs={obs.shape}")

    device = select_device()
    model = FlappyCNN(in_channels=obs.shape[1], n_actions=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ds = TensorDataset(
        torch.from_numpy(obs),
        torch.from_numpy(actions.astype(np.int64)),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    history = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
        avg_loss = total_loss / max(n, 1)
        acc = correct / max(n, 1)
        history.append({"epoch": epoch, "loss": avg_loss, "acc": acc})
        print(f"[bc] epoch {epoch}/{args.epochs} loss={avg_loss:.4f} acc={acc:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_channels": obs.shape[1],
            "n_actions": 2,
            "kind": "bc",
        },
        out,
    )

    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [h["loss"] for h in history], marker="o")
    plt.title("BC CNN training loss")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy")
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [h["acc"] for h in history], marker="o", color="green")
    plt.title("BC action accuracy")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()

    metrics_path = plot_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(history, indent=2))
    print(f"[bc] saved model -> {out}")
    print(f"[bc] saved plot  -> {plot_path}")


if __name__ == "__main__":
    main()
