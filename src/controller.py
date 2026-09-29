"""Simple hand-coded controllers for the Go1, a standing pose and a sine-wave trot, no training."""

from __future__ import annotations

import numpy as np

# Per-leg home ctrl (hip, thigh, knee). Same triplet repeated for all 4 legs.
HOME_LEG_CTRL = np.array([0.0, 0.9, -1.8])
HOME_CTRL = np.tile(HOME_LEG_CTRL, 4)  # order: FR, FL, RR, RL

# Legs in actuator order. In a trot, diagonal pairs move together (FR+RL, FL+RR).
_LEG_ORDER = ["FR", "FL", "RR", "RL"]
_TROT_PHASE = {"FR": 0.0, "RL": 0.0, "FL": np.pi, "RR": np.pi}


class StandingController:
    """Just holds the pose. No brains, no state. It just stands there."""

    name = "standing"

    def __call__(self, t: float, obs=None) -> np.ndarray:
        return HOME_CTRL.copy()


class TrotController:
    """Open-loop trot, sine waves on hip and knee around the home pose. Don't tune it to be more stable, the wobble when something's broken is the whole signal."""

    name = "trot"

    def __init__(
        self,
        frequency_hz: float = 0.8,
        hip_amplitude: float = 0.25,
        knee_amplitude: float = 0.15,
        abduction_amplitude: float = 0.0,
    ):
        self.omega = 2.0 * np.pi * frequency_hz
        self.hip_amplitude = hip_amplitude
        self.knee_amplitude = knee_amplitude
        self.abduction_amplitude = abduction_amplitude

    def __call__(self, t: float, obs=None) -> np.ndarray:
        ctrl = HOME_CTRL.copy()
        for leg_idx, leg in enumerate(_LEG_ORDER):
            phase = _TROT_PHASE[leg] + self.omega * t
            s = np.sin(phase)
            base = leg_idx * 3
            ctrl[base + 0] += self.abduction_amplitude * s
            ctrl[base + 1] += self.hip_amplitude * s
            ctrl[base + 2] += self.knee_amplitude * s
        return ctrl


CONTROLLERS = {
    "standing": StandingController,
    "trot": TrotController,
}
