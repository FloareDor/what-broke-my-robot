"""Renders actual MuJoCo frames for the README: the robot standing, and a healthy vs faulty trot side by side."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.controller import TrotController
from src.env import Go1Env
from src.perturbations import set_actuator_strength_scale

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "renders"
WIDTH, HEIGHT = 640, 480


def render_frame(env: Go1Env, renderer: mujoco.Renderer, camera: str | None = None) -> np.ndarray:
    if camera:
        renderer.update_scene(env.data, camera=camera)
    else:
        renderer.update_scene(env.data)
    return renderer.render()


def label(img: np.ndarray, text: str) -> Image.Image:
    pic = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(pic)
    try:
        font = ImageFont.truetype("arialbd.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    pad = 10
    bbox = draw.textbbox((0, 0), text, font=font)
    box_w, box_h = bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2
    draw.rectangle([0, 0, box_w, box_h], fill=(20, 24, 20, 220))
    draw.text((pad, pad - bbox[1]), text, fill=(240, 240, 235), font=font)
    return pic


def hstack(images: list[Image.Image], gap: int = 6, bg=(255, 255, 255)) -> Image.Image:
    w = sum(im.width for im in images) + gap * (len(images) - 1)
    h = max(im.height for im in images)
    canvas = Image.new("RGB", (w, h), bg)
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


def vstack(images: list[Image.Image], gap: int = 6, bg=(255, 255, 255)) -> Image.Image:
    w = max(im.width for im in images)
    h = sum(im.height for im in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (w, h), bg)
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height + gap
    return canvas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- hero shot: the robot standing, nominal ----
    env = Go1Env()
    renderer = mujoco.Renderer(env.model, height=HEIGHT, width=WIDTH)
    env.reset()
    frame = render_frame(env, renderer)
    Image.fromarray(frame).save(OUT_DIR / "hero_standing.png")
    print("wrote hero_standing.png")

    # ---- healthy vs faulty trot, same moments in time, side by side ----
    fault_actuator = "RR_hip"
    fault_scale = 0.35  # dramatic, chosen for visibility, not sampled from the real range

    healthy_env = Go1Env()
    faulty_model = healthy_env.clone_model()
    set_actuator_strength_scale(faulty_model, fault_actuator, fault_scale)
    faulty_env = Go1Env(model=faulty_model)

    healthy_renderer = mujoco.Renderer(healthy_env.model, height=HEIGHT, width=WIDTH)
    faulty_renderer = mujoco.Renderer(faulty_env.model, height=HEIGHT, width=WIDTH)

    healthy_env.reset()
    faulty_env.reset()

    controller = TrotController()
    checkpoints_t = [0.5, 1.5, 2.5]
    next_checkpoint = 0
    healthy_shots, faulty_shots = [], []

    n_steps = int(3.0 / healthy_env.control_dt)
    for i in range(n_steps):
        t = i * healthy_env.control_dt
        ctrl = controller(t)
        healthy_env.step(ctrl)
        faulty_env.step(ctrl)

        if next_checkpoint < len(checkpoints_t) and t >= checkpoints_t[next_checkpoint]:
            healthy_shots.append(label(render_frame(healthy_env, healthy_renderer, camera="tracking"), f"healthy  t={t:.1f}s"))
            faulty_shots.append(label(render_frame(faulty_env, faulty_renderer, camera="tracking"), f"RR_hip @ {fault_scale}x  t={t:.1f}s"))
            next_checkpoint += 1

    top_row = hstack(healthy_shots)
    bottom_row = hstack(faulty_shots)
    comparison = vstack([top_row, bottom_row])
    comparison.save(OUT_DIR / "healthy_vs_faulty.png")
    print("wrote healthy_vs_faulty.png")


if __name__ == "__main__":
    main()
