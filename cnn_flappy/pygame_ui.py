"""Custom pygame UI: live Flappy Bird + training metrics side-by-side."""

from __future__ import annotations

import argparse
import base64
import io
import sys
import time

import numpy as np
import pygame
from PIL import Image

from cnn_flappy.live_trainer import trainer


W, H = 1100, 640
GAME_W = 520
PANEL_X = 540


def _decode_frame(b64: str) -> np.ndarray | None:
    if not b64:
        return None
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img)


def _blit_frame(screen, frame: np.ndarray | None, rect: pygame.Rect):
    pygame.draw.rect(screen, (24, 28, 36), rect, border_radius=8)
    if frame is None:
        return
    surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    surf = pygame.transform.smoothscale(surf, (rect.w - 16, rect.h - 16))
    screen.blit(surf, (rect.x + 8, rect.y + 8))


def _sparkline(screen, values, rect: pygame.Rect, color):
    pygame.draw.rect(screen, (30, 34, 42), rect, border_radius=6)
    if len(values) < 2:
        return
    vals = np.asarray(values[-80:], dtype=np.float32)
    vmin, vmax = float(vals.min()), float(vals.max())
    if abs(vmax - vmin) < 1e-6:
        vmax = vmin + 1.0
    pts = []
    for i, v in enumerate(vals):
        x = rect.x + 6 + int((i / (len(vals) - 1)) * (rect.w - 12))
        y = rect.y + rect.h - 6 - int(((v - vmin) / (vmax - vmin)) * (rect.h - 12))
        pts.append((x, y))
    if len(pts) >= 2:
        pygame.draw.lines(screen, color, False, pts, 2)


def _label(screen, font, text, pos, color=(230, 230, 235)):
    screen.blit(font.render(text, True, color), pos)


def main():
    parser = argparse.ArgumentParser(description="Pygame live training UI")
    parser.add_argument("--timesteps", type=int, default=15000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument(
        "--init-bc", type=str, default="cnn_flappy/checkpoints/bc_cnn.pt"
    )
    parser.add_argument("--autostart", action="store_true")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("CNN Flappy — Live Training (Pygame)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    title = pygame.font.SysFont("consolas", 26, bold=True)
    small = pygame.font.SysFont("consolas", 14)

    if args.autostart:
        trainer.start(
            timesteps=args.timesteps,
            warmup=args.warmup,
            init_bc=args.init_bc,
        )

    running_ui = True
    while running_ui:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_ui = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running_ui = False
                elif event.key == pygame.K_SPACE:
                    st = trainer.get_state()
                    if st["running"]:
                        trainer.stop()
                    else:
                        trainer.start(
                            timesteps=args.timesteps,
                            warmup=args.warmup,
                            init_bc=args.init_bc,
                        )
                elif event.key == pygame.K_s:
                    trainer.stop()

        st = trainer.get_state()
        screen.fill((14, 16, 22))

        _label(screen, title, "CNN + DQN Live Training", (20, 16), (250, 250, 252))
        _label(
            screen,
            small,
            "SPACE start/stop   S stop   ESC quit",
            (20, 48),
            (160, 165, 175),
        )

        game_rect = pygame.Rect(20, 80, GAME_W, 520)
        frame = _decode_frame(st.get("frame_jpeg_b64", ""))
        _blit_frame(screen, frame, game_rect)
        _label(screen, font, "Gameplay (pygame / rgb stream)", (28, 88), (200, 210, 220))

        x = PANEL_X
        y = 80
        _label(screen, font, f"Status: {st['status']}", (x, y), (120, 220, 160))
        y += 28
        _label(screen, font, f"Message: {st['message'][:48]}", (x, y))
        y += 32
        metrics = [
            ("Step", st["step"]),
            ("Episode", st["episode"]),
            ("Epsilon", st["epsilon"]),
            ("Loss", st["last_loss"]),
            ("Ep reward", st["episode_reward"]),
            ("Last score", st["last_score"]),
            ("Mean10", st["mean_score_10"]),
            ("Max score", st["max_score"]),
            ("FPS", st["fps"]),
            ("Action", "FLAP" if st["action"] == 1 else "IDLE"),
        ]
        for name, val in metrics:
            _label(screen, font, f"{name:12s} {val}", (x, y), (220, 225, 230))
            y += 24

        y += 10
        _label(screen, small, "Episode scores", (x, y), (180, 180, 190))
        y += 20
        _sparkline(screen, st["scores"], pygame.Rect(x, y, 520, 70), (80, 180, 255))
        y += 90
        _label(screen, small, "TD loss", (x, y), (180, 180, 190))
        y += 20
        _sparkline(screen, st["losses"], pygame.Rect(x, y, 520, 70), (255, 160, 80))

        pygame.display.flip()
        clock.tick(30)

    trainer.stop()
    time.sleep(0.2)
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
