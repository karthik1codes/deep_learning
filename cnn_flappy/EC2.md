# Train CNN Flappy on AWS EC2

## Recommended instance

| Choice | Instance | Notes |
|--------|----------|--------|
| **Best balance** | **`g4dn.xlarge`** | 1× NVIDIA T4, 4 vCPU, 16 GB RAM |
| Faster env stepping | `g4dn.2xlarge` | Same GPU, more CPU |
| Avoid | CPU-only (`t3`, `c6i`) | DQN will be very slow |

- **OS:** Ubuntu 22.04 LTS (or Deep Learning AMI PyTorch)
- **Disk:** 50–100 GB gp3
- **Security group (for dashboard demo):** inbound TCP **22** (SSH), **8000** (API), **3000** (Next.js) — restrict to your IP

## Ideal training sizes (script defaults)

| Step | Default | Purpose |
|------|---------|---------|
| Record episodes | **1000** | Heuristic demos → `demos.npz` |
| BC epochs | **40** | Imitation CNN |
| DQN timesteps | **300000** | RL self-improve (visible gains often need this+) |
| Compare episodes | **30** | BC vs DQN analytics |

For a stronger demo, use `DQN_TIMESTEPS=500000`.

## One-command train (on the instance)

```bash
sudo apt update && sudo apt install -y git
git clone https://github.com/karthik1codes/deep_learning.git
cd deep_learning

chmod +x cnn_flappy/scripts/ec2_setup_and_train.sh

# keep session alive if SSH drops
sudo apt install -y tmux
tmux new -s flappy

./cnn_flappy/scripts/ec2_setup_and_train.sh
# detach: Ctrl+B then D
# reattach: tmux attach -t flappy
```

### Custom sizes

```bash
RECORD_EPISODES=2000 BC_EPOCHS=50 DQN_TIMESTEPS=500000 \
  ./cnn_flappy/scripts/ec2_setup_and_train.sh
```

### Already set up venv?

```bash
SKIP_SETUP=1 DQN_TIMESTEPS=300000 ./cnn_flappy/scripts/ec2_setup_and_train.sh
```

## Demo: difference between BC and DQN

### A) Terminal scores (headless)

```bash
cd ~/deep_learning
source .venv/bin/activate

python -m cnn_flappy.play --model cnn_flappy/checkpoints/bc_cnn.pt --episodes 20
python -m cnn_flappy.play --model cnn_flappy/checkpoints/dqn_cnn.pt --episodes 20
python -m cnn_flappy.compare --episodes 30
```

Open `cnn_flappy/outputs/compare_agents.png` for the report.

### B) Live dashboard on EC2

```bash
# terminal 1
cd ~/deep_learning && source .venv/bin/activate
python -m cnn_flappy.api_server

# terminal 2 (Node 18+ required)
cd ~/deep_learning/dashboard
npm install
npm run dev -- -H 0.0.0.0 -p 3000
```

Browser: `http://<EC2-PUBLIC-IP>:3000`  
→ **Refresh models** → **Run BC vs DQN comparison**

### C) Copy weights to your Windows PC

On your PC (CMD):

```cmd
scp -i your-key.pem ubuntu@<EC2-IP>:~/deep_learning/cnn_flappy/checkpoints/bc_cnn.pt E:\dlaat\cnn_flappy\checkpoints\
scp -i your-key.pem ubuntu@<EC2-IP>:~/deep_learning/cnn_flappy/checkpoints/dqn_cnn.pt E:\dlaat\cnn_flappy\checkpoints\
```

Then locally:

```cmd
cd /d E:\dlaat
.venv\Scripts\activate.bat
python -m cnn_flappy.api_server
```

Other window:

```cmd
cd /d E:\dlaat\dashboard
npm.cmd run dev
```

## Manual step-by-step (without the script)

```bash
cd ~/deep_learning && source .venv/bin/activate

python -m cnn_flappy.record_data --episodes 1000 --policy heuristic --out cnn_flappy/data/demos.npz
python -m cnn_flappy.train_bc --epochs 40 --batch-size 128 --data cnn_flappy/data/demos.npz --out cnn_flappy/checkpoints/bc_cnn.pt
python -m cnn_flappy.train_rl --timesteps 300000 --warmup 1000 --batch-size 64 --init-bc cnn_flappy/checkpoints/bc_cnn.pt --out cnn_flappy/checkpoints/dqn_cnn.pt
python -m cnn_flappy.compare --episodes 30
```

## Notes

- Recording uses **CPU**; BC/DQN use **GPU** when CUDA is available.
- Heuristic demos are often weak (many score-0 episodes). Longer DQN helps, but a better teacher helps more.
- Expect several hours for 300k–500k DQN steps on a T4.
