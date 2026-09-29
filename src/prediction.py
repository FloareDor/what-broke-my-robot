"""Turns an agent's parameter guess into a real MuJoCo model, using the same setters that build the hidden fault."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco

from src.perturbations import set_actuator_strength_scale, set_friction, set_torso_mass_scale


@dataclass
class ParameterEstimate:
    """What the agent thinks changed on the hidden robot. A field left as None means "no change"."""

    friction: float | None = None
    torso_mass_scale: float | None = None
    actuator_scale: dict[str, float] = field(default_factory=dict)  # {actuator_name: scale}

    def to_model(self, nominal_env) -> mujoco.MjModel:
        model = nominal_env.clone_model()
        if self.friction is not None:
            set_friction(model, self.friction)
        if self.torso_mass_scale is not None:
            set_torso_mass_scale(model, self.torso_mass_scale)
        for actuator_name, scale in self.actuator_scale.items():
            set_actuator_strength_scale(model, actuator_name, scale)
        return model

    def as_dict(self) -> dict:
        return {
            "friction": self.friction,
            "torso_mass_scale": self.torso_mass_scale,
            "actuator_scale": dict(self.actuator_scale),
        }
