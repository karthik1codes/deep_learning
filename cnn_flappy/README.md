# CNN + RL Flappy Bird Showcase

End-to-end demo of how a **CNN** reads game pixels (state), how **state–action pairs** are recorded, how the CNN is **trained**, how it **plays**, and how **DQN (RL)** self-improves the same model.

## What this shows

| Stage | Idea |
|-------|------|
| **State** | RGB frame → grayscale 84×84 → stack of 4 frames (motion) |
| **Action** | `0` = do nothing, `1` = flap |
| **Record** | Heuristic teacher saves `(obs, action, reward, next_obs)` |
| **BC (CNN)** | Supervised learning: CNN imitates teacher flaps |
| **CNN viz** | Feature maps from conv1/2/3 — see what the net “looks at” |
| **DQN (RL)** | Same CNN as Q-network; improves by maximizing reward |
| **Compare** | Random vs heuristic vs BC-CNN vs DQN-CNN scores |

## Setup (use your Python 3.11 venv)

```powershell
cd E:\dlaat
.\.venv\Scripts\activate
python --version
python -m pip install -r cnn_flappy\requirements.txt
```

Confirm `Python 3.11.x`. Do **not** use 3.13 for `flappy-bird-env`.

## Commands (step by step)

### 1) Record state–actions (pixels + flap labels)

```powershell
python -m cnn_flappy.record_data --episodes 30
```

Output: `cnn_flappy/data/demos.npz`

### 2) Train CNN with behavioral cloning

```powershell
python -m cnn_flappy.train_bc --epochs 15
```

Outputs:
- `cnn_flappy/checkpoints/bc_cnn.pt`
- `cnn_flappy/outputs/bc_loss.png`

### 3) Showcase how the CNN works (feature maps)

```powershell
python -m cnn_flappy.visualize_cnn --model cnn_flappy/checkpoints/bc_cnn.pt
```

Output: `cnn_flappy/outputs/cnn_features.png`

### 4) Self-improve with RL (CNN-DQN)

Warm-start from BC (recommended):

```powershell
python -m cnn_flappy.train_rl --timesteps 25000 --init-bc cnn_flappy/checkpoints/bc_cnn.pt
```

Quick smoke test (~few minutes):

```powershell
python -m cnn_flappy.train_rl --timesteps 8000 --warmup 500 --init-bc cnn_flappy/checkpoints/bc_cnn.pt
```

Outputs:
- `cnn_flappy/checkpoints/dqn_cnn.pt`
- `cnn_flappy/outputs/dqn_progress.png`

### 5) Play with the trained CNN

```powershell
python -m cnn_flappy.play --model cnn_flappy/checkpoints/dqn_cnn.pt --episodes 5
```

With window:

```powershell
python -m cnn_flappy.play --model cnn_flappy/checkpoints/dqn_cnn.pt --episodes 3 --render
```

Also try the BC-only model:

```powershell
python -m cnn_flappy.play --model cnn_flappy/checkpoints/bc_cnn.pt --episodes 5
```

### 6) Compare agents (shows RL improvement)

```powershell
python -m cnn_flappy.compare --episodes 10
```

Output: `cnn_flappy/outputs/compare_agents.png`

### 6b) Dual race — two birds, who wins?

```powershell
python -m cnn_flappy.dual_play --episodes 5 --render
```

Or use the dashboard panel **Dual race — two Flappy Birds** after starting `api_server`.

## One-command pipeline

```powershell
python -m cnn_flappy.run_demo
```

## Project layout

```
cnn_flappy/
  env_utils.py       # frame fix, resize, stack, heuristic teacher
  cnn_model.py       # FlappyCNN (conv layers + head)
  record_data.py     # record state-actions
  train_bc.py        # CNN imitation learning
  train_rl.py        # CNN-DQN self-improvement
  play.py            # run trained policy
  visualize_cnn.py   # feature-map showcase
  compare.py         # score chart across agents
  run_demo.py        # short end-to-end demo
  data/              # demos.npz
  checkpoints/       # bc_cnn.pt, dqn_cnn.pt
  outputs/           # plots
```

## How CNN + RL fit together

1. **CNN** turns stacked frames into action scores (BC) or Q-values (DQN).
2. **Recording** builds a dataset of good flaps for a warm start.
3. **RL** keeps playing; the replay buffer stores new transitions; TD loss updates the **same CNN** so scores rise over episodes (see `dqn_progress.png`).

## AWS EC2 training

One-shot script + instance guide:

- **[EC2.md](./EC2.md)** — recommended `g4dn.xlarge`, ideal steps, demo BC vs DQN  
- Script: `cnn_flappy/scripts/ec2_setup_and_train.sh`

```bash
chmod +x cnn_flappy/scripts/ec2_setup_and_train.sh
./cnn_flappy/scripts/ec2_setup_and_train.sh
```

## Live UI (Pygame + Next.js / shadcn)

Side-by-side gameplay and live training metrics:

- See **[LIVE_UI.md](./LIVE_UI.md)**
- Pygame HUD: `python -m cnn_flappy.pygame_ui --autostart`
- API: `python -m cnn_flappy.api_server`
- Next.js dashboard: `cd dashboard && npm.cmd run dev` → http://localhost:3000

> shadcn/ui is installed via **npm** (`npx shadcn@latest`), not pip.

## Notes

- `flappy-bird-env` needs `render_mode="rgb_array"`; this project wraps it so frames are not blank.
- Training without a display is fine. Use `--render` only when watching play.
- Longer `--timesteps` (50k–200k) usually gives clearer RL gains.
