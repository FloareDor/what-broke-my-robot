"""Renders the healthy and faulty trot as GIFs, same run used for the report's side by side comparison."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from PIL import Image

from src.controller import TrotController
from src.env import Go1Env
from src.perturbations import set_actuator_strength_scale

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "renders"
WIDTH, HEIGHT = 480, 360
DURATION = 4.5
FPS = 20


def render_frame(env: Go1Env, renderer: mujoco.Renderer, camera: str | None = None) -> np.ndarray:
    if camera:
        renderer.update_scene(env.data, camera=camera)
    else:
        renderer.update_scene(env.data)
    return renderer.render()


def save_gif(frames: list[np.ndarray], path: Path, fps: int = FPS) -> None:
    images = [Image.fromarray(f).convert("RGB") for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    healthy_frames, faulty_frames = [], []

    render_every = max(1, int(round((1.0 / FPS) / healthy_env.control_dt)))
    n_steps = int(DURATION / healthy_env.control_dt)
    for i in range(n_steps):
        t = i * healthy_env.control_dt
        ctrl = controller(t)
        healthy_env.step(ctrl)
        faulty_env.step(ctrl)

        if i % render_every == 0:
            healthy_frames.append(render_frame(healthy_env, healthy_renderer, camera="tracking"))
            faulty_frames.append(render_frame(faulty_env, faulty_renderer, camera="tracking"))

    save_gif(healthy_frames, OUT_DIR / "healthy.gif")
    print("wrote healthy.gif")
    save_gif(faulty_frames, OUT_DIR / "faulty.gif")
    print("wrote faulty.gif")


if __name__ == "__main__":
    main()
