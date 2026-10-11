"""Side-by-side BC-CNN vs DQN-CNN race: render both birds and declare a winner."""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from cnn_flappy.cnn_model import FlappyCNN, select_device
from cnn_flappy.compare import DEFAULT_BC, DEFAULT_DQN
from cnn_flappy.env_utils import RenderFixWrapper, make_env
from cnn_flappy.live_trainer import rgb_to_jpeg_b64
from cnn_flappy.play import load_model


def _find_rgb(env) -> Optional[np.ndarray]:
    cur = env
    while cur is not None:
        if isinstance(cur, RenderFixWrapper) and cur.last_rgb is not None:
            return cur.last_rgb
        cur = getattr(cur, "env", None)
    return None


def _policy(model: FlappyCNN, device: torch.device, obs: np.ndarray) -> int:
    with torch.no_grad():
        q = model(torch.from_numpy(obs).unsqueeze(0).to(device))
        return int(q.argmax(1).item())


@dataclass
class DualRaceState:
    running: bool = False
    status: str = "idle"
    message: str = "Ready for BC vs DQN dual race"
    episode: int = 0
    total_episodes: int = 0
    bc_score: int = 0
    dqn_score: int = 0
    bc_alive: bool = False
    dqn_alive: bool = False
    bc_total: int = 0
    dqn_total: int = 0
    bc_frame_b64: str = ""
    dqn_frame_b64: str = ""
    winner: str = ""
    episode_winners: list[str] = field(default_factory=list)
    updated_at: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "status": self.status,
            "message": self.message,
            "episode": self.episode,
            "total_episodes": self.total_episodes,
            "bc_score": self.bc_score,
            "dqn_score": self.dqn_score,
            "bc_alive": self.bc_alive,
            "dqn_alive": self.dqn_alive,
            "bc_total": self.bc_total,
            "dqn_total": self.dqn_total,
            "bc_frame_b64": self.bc_frame_b64,
            "dqn_frame_b64": self.dqn_frame_b64,
            "winner": self.winner,
            "episode_winners": self.episode_winners[-20:],
            "updated_at": self.updated_at,
        }


class DualRaceRunner:
    """Background BC vs DQN races for API / dashboard streaming."""

    def __init__(self):
        self.state = DualRaceState()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return self.state.snapshot()

    def _update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self.state, k, v)
            self.state.updated_at = time.time()

    def start(
        self,
        episodes: int = 5,
        bc_path: str = DEFAULT_BC,
        dqn_path: str = DEFAULT_DQN,
        frame_every: int = 2,
        max_steps: int = 2000,
    ) -> bool:
        with self._lock:
            if self.state.running:
                return False
            self.state = DualRaceState(
                running=True,
                status="starting",
                message="Starting dual race…",
                total_episodes=episodes,
            )
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            kwargs={
                "episodes": episodes,
                "bc_path": bc_path,
                "dqn_path": dqn_path,
                "frame_every": frame_every,
                "max_steps": max_steps,
            },
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        self._update(message="Stop requested…", status="stopping")

    def _run(self, episodes, bc_path, dqn_path, frame_every, max_steps):
        try:
            if not Path(bc_path).exists() or not Path(dqn_path).exists():
                self._update(
                    running=False,
                    status="error",
                    message="Missing bc_cnn.pt or dqn_cnn.pt in checkpoints/",
                )
                return

            device = select_device()
            bc_model = load_model(bc_path, device)
            dqn_model = load_model(dqn_path, device)

            bc_env = make_env(render_mode="rgb_array")
            dqn_env = make_env(render_mode="rgb_array")

            bc_totals: list[int] = []
            dqn_totals: list[int] = []
            winners: list[str] = []

            self._update(status="racing", message="Racing BC vs DQN")

            for ep in range(1, episodes + 1):
                if self._stop.is_set():
                    break

                bc_obs, bc_info = bc_env.reset()
                dqn_obs, dqn_info = dqn_env.reset()
                bc_done = dqn_done = False
                steps = 0

                while (not bc_done or not dqn_done) and steps < max_steps:
                    if self._stop.is_set():
                        break

                    if not bc_done:
                        a = _policy(bc_model, device, bc_obs)
                        bc_obs, _, term, trunc, bc_info = bc_env.step(a)
                        bc_done = bool(term or trunc)
                    if not dqn_done:
                        a = _policy(dqn_model, device, dqn_obs)
                        dqn_obs, _, term, trunc, dqn_info = dqn_env.step(a)
                        dqn_done = bool(term or trunc)

                    steps += 1
                    if steps % frame_every == 0:
                        bc_rgb = _find_rgb(bc_env)
                        dqn_rgb = _find_rgb(dqn_env)
                        self._update(
                            episode=ep,
                            bc_score=int(bc_info.get("score", 0)),
                            dqn_score=int(dqn_info.get("score", 0)),
                            bc_alive=not bc_done,
                            dqn_alive=not dqn_done,
                            bc_frame_b64=rgb_to_jpeg_b64(bc_rgb) if bc_rgb is not None else "",
                            dqn_frame_b64=rgb_to_jpeg_b64(dqn_rgb) if dqn_rgb is not None else "",
                            message=f"Episode {ep}/{episodes}  BC={bc_info.get('score', 0)}  DQN={dqn_info.get('score', 0)}",
                        )

                bc_s = int(bc_info.get("score", 0))
                dqn_s = int(dqn_info.get("score", 0))
                bc_totals.append(bc_s)
                dqn_totals.append(dqn_s)
                if dqn_s > bc_s:
                    ep_winner = "DQN-CNN"
                elif bc_s > dqn_s:
                    ep_winner = "BC-CNN"
                else:
                    ep_winner = "tie"
                winners.append(ep_winner)

                self._update(
                    episode=ep,
                    bc_score=bc_s,
                    dqn_score=dqn_s,
                    bc_total=sum(bc_totals),
                    dqn_total=sum(dqn_totals),
                    episode_winners=winners,
                    message=f"Episode {ep} winner: {ep_winner}",
                )

            bc_env.close()
            dqn_env.close()

            if sum(dqn_totals) > sum(bc_totals):
                overall = "DQN-CNN"
            elif sum(bc_totals) > sum(dqn_totals):
                overall = "BC-CNN"
            else:
                overall = "tie"

            self._update(
                running=False,
                status="finished",
                winner=overall,
                bc_total=sum(bc_totals),
                dqn_total=sum(dqn_totals),
                episode_winners=winners,
                message=(
                    f"FINAL WINNER: {overall}  "
                    f"(BC total pipes={sum(bc_totals)}, DQN total pipes={sum(dqn_totals)})"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._update(running=False, status="error", message=f"Error: {exc}")


dual_race = DualRaceRunner()


def _compose_panel(bc_rgb, dqn_rgb, bc_score, dqn_score, title: str) -> np.ndarray:
    """Build a side-by-side RGB panel for pygame."""
    h = 360
    def fit(rgb):
        if rgb is None:
            return np.zeros((h, 400, 3), dtype=np.uint8)
        img = Image.fromarray(rgb.astype(np.uint8))
        ratio = h / img.height
        img = img.resize((int(img.width * ratio), h), Image.BILINEAR)
        return np.asarray(img)

    left = fit(bc_rgb)
    right = fit(dqn_rgb)
    gap = 12
    header = 48
    w = left.shape[1] + right.shape[1] + gap + 40
    canvas = Image.new("RGB", (w, h + header), (18, 20, 28))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((16, 12), title, fill=(240, 240, 245), font=font)
    draw.text((16, header - 22), f"BC-CNN  score={bc_score}", fill=(253, 174, 107), font=font)
    draw.text(
        (left.shape[1] + gap + 24, header - 22),
        f"DQN-CNN  score={dqn_score}",
        fill=(230, 85, 13),
        font=font,
    )
    canvas.paste(Image.fromarray(left), (16, header))
    canvas.paste(Image.fromarray(right), (16 + left.shape[1] + gap, header))
    return np.asarray(canvas)


def run_pygame_dual(
    episodes: int,
    bc_path: str,
    dqn_path: str,
    max_steps: int = 2000,
):
    import pygame

    device = select_device()
    bc_model = load_model(bc_path, device)
    dqn_model = load_model(dqn_path, device)
    bc_env = make_env(render_mode="rgb_array")
    dqn_env = make_env(render_mode="rgb_array")

    pygame.init()
    screen = None
    clock = pygame.time.Clock()
    bc_totals, dqn_totals = [], []

    for ep in range(1, episodes + 1):
        bc_obs, bc_info = bc_env.reset()
        dqn_obs, dqn_info = dqn_env.reset()
        bc_done = dqn_done = False
        steps = 0

        while (not bc_done or not dqn_done) and steps < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    bc_env.close()
                    dqn_env.close()
                    pygame.quit()
                    return

            if not bc_done:
                bc_obs, _, term, trunc, bc_info = bc_env.step(
                    _policy(bc_model, device, bc_obs)
                )
                bc_done = bool(term or trunc)
            if not dqn_done:
                dqn_obs, _, term, trunc, dqn_info = dqn_env.step(
                    _policy(dqn_model, device, dqn_obs)
                )
                dqn_done = bool(term or trunc)

            panel = _compose_panel(
                _find_rgb(bc_env),
                _find_rgb(dqn_env),
                int(bc_info.get("score", 0)),
                int(dqn_info.get("score", 0)),
                f"Episode {ep}/{episodes} — BC vs DQN",
            )
            if screen is None:
                screen = pygame.display.set_mode((panel.shape[1], panel.shape[0]))
                pygame.display.set_caption("BC-CNN vs DQN-CNN Dual Race")
            surf = pygame.surfarray.make_surface(np.transpose(panel, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)
            steps += 1

        bc_s = int(bc_info.get("score", 0))
        dqn_s = int(dqn_info.get("score", 0))
        bc_totals.append(bc_s)
        dqn_totals.append(dqn_s)
        ep_w = "DQN" if dqn_s > bc_s else ("BC" if bc_s > dqn_s else "tie")
        print(f"[dual] episode {ep}: BC={bc_s} DQN={dqn_s} winner={ep_w}")

    overall = (
        "DQN-CNN"
        if sum(dqn_totals) > sum(bc_totals)
        else ("BC-CNN" if sum(bc_totals) > sum(dqn_totals) else "tie")
    )
    print(
        f"[dual] FINAL WINNER: {overall}  "
        f"BC total={sum(bc_totals)} DQN total={sum(dqn_totals)}"
    )
    # hold final screen a moment
    time.sleep(2.5)
    bc_env.close()
    dqn_env.close()
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="BC vs DQN dual Flappy race")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--bc", type=str, default=DEFAULT_BC)
    parser.add_argument("--dqn", type=str, default=DEFAULT_DQN)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Pygame side-by-side window",
    )
    args = parser.parse_args()

    if not Path(args.bc).exists() or not Path(args.dqn).exists():
        raise SystemExit("Need both checkpoints: bc_cnn.pt and dqn_cnn.pt")

    if args.render:
        run_pygame_dual(args.episodes, args.bc, args.dqn)
    else:
        # headless race via DualRaceRunner (blocking)
        runner = DualRaceRunner()
        runner.start(episodes=args.episodes, bc_path=args.bc, dqn_path=args.dqn)
        while runner.get_state()["running"]:
            time.sleep(0.5)
            st = runner.get_state()
            print(st["message"])
        st = runner.get_state()
        print(st["message"])
        print("winner:", st["winner"])


if __name__ == "__main__":
    main()
