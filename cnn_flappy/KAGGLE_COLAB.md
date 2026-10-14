# Colab / Kaggle training (with resume checkpoints)

BC and DQN now save **mid-training PyTorch checkpoints** so a session timeout does not wipe progress.

| File | Purpose |
|------|---------|
| `cnn_flappy/checkpoints/bc_cnn.pt` | Inference-ready BC weights (updated every epoch) |
| `cnn_flappy/checkpoints/bc_cnn_train_ckpt.pt` | Full BC resume state (model + optimizer + epoch) |
| `cnn_flappy/checkpoints/dqn_cnn.pt` | Inference-ready DQN weights |
| `cnn_flappy/checkpoints/dqn_cnn_train_ckpt.pt` | Full DQN resume state (policy/target/optimizer/step) |

## Fresh run (Colab-style)

```python
%cd /content
!git clone https://github.com/karthik1codes/deep_learning.git
%cd deep_learning

import sys
print("Python", sys.version)

# 1) Install deps WITHOUT touching numpy/torch
!pip -q install "gymnasium>=0.29.0,<1.0" "pygame>=2.5.0" matplotlib pillow

# 2) Install flappy env without its pinned numpy==1.25.2
!pip -q install "flappy-bird-env==0.1.6" --no-deps

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

# 3) Must print ok before continuing
import flappy_bird_env, gymnasium, numpy, torch
print("ok", "numpy", numpy.__version__, "cuda", torch.cuda.is_available())

# 4) Pipeline (checkpoints auto-save during training)
!python -m cnn_flappy.record_data --episodes 1000 --policy heuristic --out cnn_flappy/data/demos.npz
!python -m cnn_flappy.train_bc --epochs 40 --batch-size 128 --ckpt-every 5 --data cnn_flappy/data/demos.npz --out cnn_flappy/checkpoints/bc_cnn.pt
!python -m cnn_flappy.train_rl --timesteps 500000 --warmup 1000 --batch-size 64 --ckpt-every 10000 --init-bc cnn_flappy/checkpoints/bc_cnn.pt --out cnn_flappy/checkpoints/dqn_cnn.pt
!python -m cnn_flappy.compare --episodes 30

# Colab download
from google.colab import files
files.download("cnn_flappy/checkpoints/bc_cnn.pt")
files.download("cnn_flappy/checkpoints/dqn_cnn.pt")
files.download("cnn_flappy/checkpoints/bc_cnn_train_ckpt.pt")
files.download("cnn_flappy/checkpoints/dqn_cnn_train_ckpt.pt")
```

## Resume after Kaggle/Colab disconnect

Re-open the notebook, reinstall deps (steps 1–3), then:

```python
%cd /content/deep_learning   # or your Kaggle working copy path
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Resume BC from last saved epoch (same --epochs target)
!python -m cnn_flappy.train_bc --resume --epochs 40 --batch-size 128 --ckpt-every 5 --data cnn_flappy/data/demos.npz --out cnn_flappy/checkpoints/bc_cnn.pt

# Resume DQN from last saved step (same --timesteps target)
!python -m cnn_flappy.train_rl --resume --timesteps 500000 --warmup 1000 --batch-size 64 --ckpt-every 10000 --out cnn_flappy/checkpoints/dqn_cnn.pt

!python -m cnn_flappy.compare --episodes 30
```

Do **not** pass `--init-bc` when resuming DQN (weights already come from the train checkpoint).

## How to finish without losing work

1. **Use `--resume`** after every reconnect — training continues from `*_train_ckpt.pt`.
2. **Train in chunks** if sessions die often, e.g. run DQN to 100k, reconnect, resume to 200k, … up to 500k (always keep the same final `--timesteps`).
3. **Kaggle:** save/download `*_train_ckpt.pt` and `*.pt` to your local machine or a Kaggle Dataset after each chunk, then upload them back into `/kaggle/working/...` before `--resume`.
4. **Keep the session alive:** leave the tab focused, or use a small keep-alive snippet; idle timeouts still happen on free tiers.
5. **Prefer GPU + shorter idle risk:** Colab T4 / Kaggle GPU is much faster than CPU for 300k–500k DQN steps.
6. **Recommended targets:** demos **1000** eps · BC **40** epochs · DQN **300k–500k** steps with `--ckpt-every 10000`.

## Kaggle path tip

If your notebook clones into `/kaggle/working/deep_learning`, replace `/content` with `/kaggle/working` in the cells above.
