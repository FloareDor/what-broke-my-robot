"""Checks the robot stands and walks without falling, prints height and forward distance for both."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.controller import StandingController, TrotController
from src.env import Go1Env

FALL_HEIGHT_M = 0.15  # nominal standing height is ~0.27m; well below this means it fell


def run(controller, duration_s: float, seed: int = 0) -> dict:
    env = Go1Env()
    env.reset(seed=seed)
    n_steps = int(duration_s / env.control_dt)
    heights = np.zeros(n_steps)
    x_pos = np.zeros(n_steps)
    for i in range(n_steps):
        ctrl = controller(env.data.time)
        record = env.step(ctrl)
        heights[i] = record.torso_pos[2]
        x_pos[i] = record.torso_pos[0]
    fell = bool(np.any(heights < FALL_HEIGHT_M))
    return {
        "controller": controller.name,
        "fell": fell,
        "min_height": float(heights.min()),
        "final_height": float(heights[-1]),
        "forward_distance_m": float(x_pos[-1] - x_pos[0]),
    }


def main():
    for controller, duration in [(StandingController(), 3.0), (TrotController(), 5.0)]:
        result = run(controller, duration)
        status = "FELL" if result["fell"] else "stable"
        print(
            f"[{result['controller']:8s}] {status:6s}  "
            f"min_height={result['min_height']:.3f}m  "
            f"final_height={result['final_height']:.3f}m  "
            f"forward={result['forward_distance_m']:+.3f}m"
        )


if __name__ == "__main__":
    main()
