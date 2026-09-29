"""Hidden physical faults, applied by editing MjModel arrays on a cloned model, the nominal one never gets touched."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from src.env import ACTUATOR_NAMES, FOOT_GEOM_NAMES

# Ranges from build_plan.md "MVP Setup".
FRICTION_RANGE = (0.3, 1.2)          # absolute floor sliding-friction coefficient
TORSO_MASS_SCALE_RANGE = (0.7, 1.4)  # multiplier on nominal trunk mass
ACTUATOR_STRENGTH_SCALE_RANGE = (0.5, 1.0)  # multiplier on one actuator's kp

PERTURBATION_KINDS = ("friction", "torso_mass", "actuator_strength")


@dataclass
class Perturbation:
    kind: str            # one of PERTURBATION_KINDS
    value: float          # resulting absolute value (friction) or scale factor (mass/actuator)
    target: str | None = None  # actuator name, only set for "actuator_strength"

    def as_dict(self) -> dict:
        return {"kind": self.kind, "value": self.value, "target": self.target}


def sample_perturbation(rng: np.random.Generator, kind: str | None = None) -> Perturbation:
    """Sample one perturbation. Only one parameter changes at a time (MVP)."""
    if kind is None:
        kind = rng.choice(PERTURBATION_KINDS)

    if kind == "friction":
        value = rng.uniform(*FRICTION_RANGE)
        return Perturbation(kind=kind, value=float(value))

    if kind == "torso_mass":
        scale = rng.uniform(*TORSO_MASS_SCALE_RANGE)
        return Perturbation(kind=kind, value=float(scale))

    if kind == "actuator_strength":
        scale = rng.uniform(*ACTUATOR_STRENGTH_SCALE_RANGE)
        actuator = rng.choice(ACTUATOR_NAMES)
        return Perturbation(kind=kind, value=float(scale), target=str(actuator))

    raise ValueError(f"unknown perturbation kind: {kind}")


def set_friction(model: mujoco.MjModel, value: float) -> None:
    # foot geoms have priority=1, so the floor's friction gets ignored, has to be set on the feet
    for name in FOOT_GEOM_NAMES:
        foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        model.geom_friction[foot_id, 0] = value


def set_torso_mass_scale(model: mujoco.MjModel, scale: float) -> None:
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    model.body_mass[trunk_id] *= scale
    model.body_inertia[trunk_id] *= scale


def set_actuator_strength_scale(model: mujoco.MjModel, actuator_name: str, scale: float) -> None:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    model.actuator_gainprm[actuator_id, 0] *= scale
    # Position actuators store biasprm[1] as -kp, so scale that too to match the gain.
    model.actuator_biasprm[actuator_id, 1] *= scale


def apply_perturbation(model: mujoco.MjModel, perturbation: Perturbation) -> None:
    """Mutate `model` in place so it matches the sampled hidden perturbation."""
    if perturbation.kind == "friction":
        set_friction(model, perturbation.value)
    elif perturbation.kind == "torso_mass":
        set_torso_mass_scale(model, perturbation.value)
    elif perturbation.kind == "actuator_strength":
        set_actuator_strength_scale(model, perturbation.target, perturbation.value)
    else:
        raise ValueError(f"unknown perturbation kind: {perturbation.kind}")


def make_hidden_model(env, rng: np.random.Generator, kind: str | None = None) -> tuple[mujoco.MjModel, Perturbation]:
    """Convenience: clone the env's nominal model and apply one sampled perturbation."""
    perturbation = sample_perturbation(rng, kind=kind)
    hidden_model = env.clone_model()
    apply_perturbation(hidden_model, perturbation)
    return hidden_model, perturbation
