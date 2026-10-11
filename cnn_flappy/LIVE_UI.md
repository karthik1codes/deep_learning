# CNN Flappy — Live UI (Pygame + Next.js / shadcn)

Two UIs share the same live trainer:

1. **Pygame window** — local side-by-side game + metrics HUD  
2. **Next.js + shadcn/ui** — browser dashboard fed by FastAPI WebSocket  

> Note: **shadcn/ui is not a pip package**. It is installed in the Next.js app with `npx shadcn@latest`.  
> Python side uses `fastapi`, `uvicorn`, `pillow`, and **pygame**.

## 1) Python API + trainer

```powershell
cd E:\dlaat
.\.venv\Scripts\activate
python -m pip install fastapi uvicorn pillow
python -m cnn_flappy.api_server
```

API: `http://127.0.0.1:8000`  
WebSocket: `ws://127.0.0.1:8000/ws/live`

## 2) Next.js dashboard (shadcn)

```powershell
cd E:\dlaat\dashboard
npm.cmd install
npm.cmd run dev
```

Open: [http://localhost:3000](http://localhost:3000)

Click **Start Training** — left = live gameplay frames, right = live metrics.

Scroll to **Model comparison — BC-CNN vs DQN-CNN** and click **Run BC vs DQN comparison**
to evaluate both trained models (table + chart) for AAT result analysis.

API helpers:
- `GET /api/models` — checkpoint availability
- `POST /api/compare` — start evaluation (`{"episodes": 10}`)
- `GET /api/compare` — poll status + results
- `POST /api/dual/start` — side-by-side BC vs DQN race
- `GET /api/dual` / `WS /ws/dual` — live dual frames + winner

Pygame dual window (local):

```powershell
python -m cnn_flappy.dual_play --episodes 5 --render
```

## 3) Optional: custom Pygame UI

```powershell
cd E:\dlaat
.\.venv\Scripts\activate
python -m cnn_flappy.pygame_ui --autostart --timesteps 15000
```

Keys: `SPACE` start/stop · `S` stop · `ESC` quit

## Architecture

```
pygame env (rgb frames)
        │
        ▼
 LiveDQNTrainer  ──WebSocket/HTTP──►  Next.js + shadcn dashboard
        │
        └──► pygame_ui.py (local HUD)
```

## Tips

- Prefer Python **3.11** venv (already used for `flappy-bird-env`).
- Warm-start works if `cnn_flappy/checkpoints/bc_cnn.pt` exists (`python -m cnn_flappy.train_bc`).
- Rendering every frame slows training; the stream samples frames (`frame_every`).
