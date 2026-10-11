#!/usr/bin/env bash
# EC2 one-shot: setup (optional) → record → BC → DQN → compare
# Recommended instance: g4dn.xlarge (Tesla T4), Ubuntu 22.04, Python 3.10/3.11
#
# Usage (from repo root):
#   chmod +x cnn_flappy/scripts/ec2_setup_and_train.sh
#   ./cnn_flappy/scripts/ec2_setup_and_train.sh
#
# Or with custom sizes:
#   RECORD_EPISODES=1000 BC_EPOCHS=40 DQN_TIMESTEPS=300000 ./cnn_flappy/scripts/ec2_setup_and_train.sh
#
# Skip apt/venv setup if already done:
#   SKIP_SETUP=1 ./cnn_flappy/scripts/ec2_setup_and_train.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RECORD_EPISODES="${RECORD_EPISODES:-1000}"
BC_EPOCHS="${BC_EPOCHS:-40}"
BC_BATCH="${BC_BATCH:-128}"
DQN_TIMESTEPS="${DQN_TIMESTEPS:-300000}"
DQN_WARMUP="${DQN_WARMUP:-1000}"
DQN_BATCH="${DQN_BATCH:-64}"
COMPARE_EPISODES="${COMPARE_EPISODES:-30}"
SKIP_SETUP="${SKIP_SETUP:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

DEMOS="cnn_flappy/data/demos.npz"
BC_OUT="cnn_flappy/checkpoints/bc_cnn.pt"
DQN_OUT="cnn_flappy/checkpoints/dqn_cnn.pt"

echo "=============================================="
echo " CNN Flappy — EC2 train pipeline"
echo " Root: $ROOT"
echo " Record episodes : $RECORD_EPISODES"
echo " BC epochs       : $BC_EPOCHS"
echo " DQN timesteps   : $DQN_TIMESTEPS"
echo "=============================================="

if [[ "$SKIP_SETUP" != "1" ]]; then
  echo "[setup] system packages..."
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
      git "$PYTHON_BIN" "${PYTHON_BIN}-venv" python3-pip \
      libsdl2-2.0-0 libsdl2-image-2.0-0 libsdl2-mixer-2.0-0 libsdl2-ttf-2.0-0 \
      || true
  fi

  if [[ ! -d .venv ]]; then
    echo "[setup] creating venv with $PYTHON_BIN..."
    "$PYTHON_BIN" -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip wheel
  pip install -r cnn_flappy/requirements.txt
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[gpu] checking CUDA..."
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU — training will be slow on CPU")
PY

mkdir -p cnn_flappy/data cnn_flappy/checkpoints cnn_flappy/outputs

echo "[1/4] recording heuristic demos → $DEMOS"
python -m cnn_flappy.record_data \
  --episodes "$RECORD_EPISODES" \
  --policy heuristic \
  --out "$DEMOS"

echo "[2/4] training BC-CNN → $BC_OUT"
python -m cnn_flappy.train_bc \
  --epochs "$BC_EPOCHS" \
  --batch-size "$BC_BATCH" \
  --data "$DEMOS" \
  --out "$BC_OUT"

echo "[3/4] training DQN-CNN → $DQN_OUT"
python -m cnn_flappy.train_rl \
  --timesteps "$DQN_TIMESTEPS" \
  --warmup "$DQN_WARMUP" \
  --batch-size "$DQN_BATCH" \
  --init-bc "$BC_OUT" \
  --out "$DQN_OUT"

echo "[4/4] comparing agents ($COMPARE_EPISODES episodes)"
python -m cnn_flappy.compare --episodes "$COMPARE_EPISODES" \
  --bc "$BC_OUT" --dqn "$DQN_OUT"

echo ""
echo "=============================================="
echo " DONE"
echo " Weights:"
echo "   $BC_OUT"
echo "   $DQN_OUT"
echo " Compare plot: cnn_flappy/outputs/compare_agents.png"
echo ""
echo " Demo play (scores only, headless):"
echo "   source .venv/bin/activate"
echo "   python -m cnn_flappy.play --model $BC_OUT --episodes 10"
echo "   python -m cnn_flappy.play --model $DQN_OUT --episodes 10"
echo ""
echo " Dashboard (open SG ports 8000 + 3000):"
echo "   python -m cnn_flappy.api_server"
echo "   cd dashboard && npm install && npm run dev -- -H 0.0.0.0 -p 3000"
echo "=============================================="
