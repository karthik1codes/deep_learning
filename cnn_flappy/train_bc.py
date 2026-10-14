"""Behavioral cloning: train CNN to imitate recorded (state, action) pairs.

Supports mid-training PyTorch checkpoints so Colab/Kaggle sessions can resume.
"""

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
from cnn_flappy.numpy_compat import ensure_numpy_core_alias


def _ckpt_path(out: Path) -> Path:
    return out.with_name(out.stem + "_train_ckpt.pt")


def save_train_checkpoint(
    path: Path,
    *,
    model,
    opt,
    epoch: int,
    history: list,
    in_channels: int,
    n_actions: int,
    args_dict: dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "bc_train",
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "history": history,
            "in_channels": in_channels,
            "n_actions": n_actions,
            "args": args_dict,
        },
        path,
    )
    print(f"[bc] checkpoint saved -> {path} (epoch {epoch})")


def save_final_model(out: Path, model, in_channels: int, n_actions: int):
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_channels": in_channels,
            "n_actions": n_actions,
            "kind": "bc",
        },
        out,
    )


def main():
    parser = argparse.ArgumentParser(description="Train CNN via behavioral cloning")
    parser.add_argument("--data", type=str, default="cnn_flappy/data/demos.npz")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="cnn_flappy/checkpoints/bc_cnn.pt")
    parser.add_argument("--plot", type=str, default="cnn_flappy/outputs/bc_loss.png")
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=5,
        help="Save a resumable training checkpoint every N epochs (0=only final)",
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

    data = np.load(args.data)
    obs = data["observations"]
    actions = data["actions"]
    print(f"[bc] loaded {len(actions)} samples, obs={obs.shape}")

    device = select_device()
    in_channels = int(obs.shape[1])
    n_actions = 2
    model = FlappyCNN(in_channels=in_channels, n_actions=n_actions).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    counts = np.bincount(actions.astype(np.int64), minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (2.0 * counts)
    weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[bc] class counts={counts.tolist()} weights={weights.tolist()}")

    ds = TensorDataset(
        torch.from_numpy(obs),
        torch.from_numpy(actions.astype(np.int64)),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    out = Path(args.out)
    train_ckpt = Path(args.resume_from) if args.resume_from else _ckpt_path(out)
    start_epoch = 1
    history: list = []

    if args.resume or args.resume_from:
        if train_ckpt.exists():
            ensure_numpy_core_alias()
            ckpt = torch.load(train_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                opt.load_state_dict(ckpt["optimizer_state"])
            history = list(ckpt.get("history", []))
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(f"[bc] resumed from {train_ckpt} at epoch {start_epoch}/{args.epochs}")
        else:
            print(f"[bc] no resume checkpoint at {train_ckpt}; training from scratch")

    args_dict = vars(args)
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=weight_t)
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

        # Always keep a latest inference-ready model + periodic full train ckpt
        save_final_model(out, model, in_channels, n_actions)
        if args.ckpt_every > 0 and (epoch % args.ckpt_every == 0 or epoch == args.epochs):
            save_train_checkpoint(
                train_ckpt,
                model=model,
                opt=opt,
                epoch=epoch,
                history=history,
                in_channels=in_channels,
                n_actions=n_actions,
                args_dict=args_dict,
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
    print(f"[bc] resume later with: --resume --out {out}")


if __name__ == "__main__":
    main()
