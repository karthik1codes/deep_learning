"""FastAPI backend: stream live training frames + metrics to Next.js."""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any, Optional

# Headless charts from worker threads (avoids Tk "main thread" crashes).
os.environ.setdefault("MPLBACKEND", "Agg")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cnn_flappy.compare import DEFAULT_BC, DEFAULT_DQN, model_status, run_comparison
from cnn_flappy.dual_play import dual_race
from cnn_flappy.live_trainer import trainer

app = FastAPI(title="CNN Flappy Live API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_compare_lock = threading.Lock()
_compare_job: dict[str, Any] = {
    "running": False,
    "error": None,
    "result": None,
}


class StartRequest(BaseModel):
    timesteps: int = Field(default=20000, ge=100, le=500_000)
    warmup: int = Field(default=500, ge=10)
    init_bc: str = "cnn_flappy/checkpoints/bc_cnn.pt"
    frame_every: int = Field(default=2, ge=1, le=30)


class CompareRequest(BaseModel):
    episodes: int = Field(default=10, ge=2, le=50)
    include_baselines: bool = True
    bc_path: str = DEFAULT_BC
    dqn_path: str = DEFAULT_DQN


class DualRaceRequest(BaseModel):
    episodes: int = Field(default=5, ge=1, le=30)
    bc_path: str = DEFAULT_BC
    dqn_path: str = DEFAULT_DQN
    frame_every: int = Field(default=2, ge=1, le=10)


@app.get("/api/health")
def health():
    return {"ok": True, "service": "cnn-flappy-live"}


@app.get("/api/state")
def get_state():
    return trainer.get_state()


@app.get("/api/models")
def get_models():
    return model_status()


@app.post("/api/train/start")
def start_train(body: StartRequest):
    started = trainer.start(
        timesteps=body.timesteps,
        warmup=body.warmup,
        init_bc=body.init_bc,
        frame_every=body.frame_every,
    )
    return {"started": started, "state": trainer.get_state()}


@app.post("/api/train/stop")
def stop_train():
    trainer.stop()
    return {"stopped": True, "state": trainer.get_state()}


def _run_compare_job(body: CompareRequest):
    global _compare_job
    try:
        result = run_comparison(
            episodes=body.episodes,
            bc_path=body.bc_path,
            dqn_path=body.dqn_path,
            include_baselines=body.include_baselines,
            save_plot="cnn_flappy/outputs/compare_agents.png",
        )
        with _compare_lock:
            _compare_job["result"] = result
            _compare_job["error"] = None
            _compare_job["running"] = False
    except Exception as exc:  # noqa: BLE001
        with _compare_lock:
            _compare_job["error"] = str(exc)
            _compare_job["running"] = False


@app.post("/api/compare")
def start_compare(body: CompareRequest):
    with _compare_lock:
        if _compare_job["running"]:
            return {"started": False, "message": "Comparison already running"}
        _compare_job["running"] = True
        _compare_job["error"] = None
        # keep previous result until new one finishes
    thread = threading.Thread(target=_run_compare_job, args=(body,), daemon=True)
    thread.start()
    return {"started": True, "message": "Comparison started"}


@app.get("/api/compare")
def get_compare():
    with _compare_lock:
        return {
            "running": _compare_job["running"],
            "error": _compare_job["error"],
            "result": _compare_job["result"],
        }


@app.post("/api/dual/start")
def dual_start(body: DualRaceRequest):
    started = dual_race.start(
        episodes=body.episodes,
        bc_path=body.bc_path,
        dqn_path=body.dqn_path,
        frame_every=body.frame_every,
    )
    return {"started": started, "state": dual_race.get_state()}


@app.post("/api/dual/stop")
def dual_stop():
    dual_race.stop()
    return {"stopped": True, "state": dual_race.get_state()}


@app.get("/api/dual")
def dual_get():
    return dual_race.get_state()


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(trainer.get_state())
            await asyncio.sleep(0.1)  # ~10 Hz
    except WebSocketDisconnect:
        return


@app.websocket("/ws/dual")
async def ws_dual(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(dual_race.get_state())
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        return


def main(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn

    uvicorn.run(
        "cnn_flappy.api_server:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
