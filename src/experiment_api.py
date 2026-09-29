"""The tool interface an agent uses to diagnose a hidden robot: inspect_model, run_experiment, inspect_trajectory, submit_model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.env import ACTUATOR_NAMES, Go1Env, Trajectory
from src.perturbations import make_hidden_model
from src.prediction import ParameterEstimate

PolicyFn = Callable[[float, dict], np.ndarray]


@dataclass
class ExperimentRecord:
    experiment_id: int
    controller_code: str
    duration: float
    trajectory: dict[str, np.ndarray]


class ExperimentSession:
    """A single diagnosis episode against one hidden robot."""

    def __init__(self, seed: int | None = None, budget: int = 8, hidden_kind: str | None = None):
        self.nominal_env = Go1Env()
        rng = np.random.default_rng(seed)
        hidden_model, self._true_perturbation = make_hidden_model(self.nominal_env, rng, kind=hidden_kind)
        self.hidden_env = Go1Env(model=hidden_model)

        self.budget = budget
        self.experiments_used = 0
        self._experiments: dict[int, ExperimentRecord] = {}
        self._next_id = 0
        self.parameter_estimate: ParameterEstimate | None = None

    # the tools the agent actually calls

    def inspect_model(self) -> dict:
        """Static description of the *nominal* model. Reveals nothing hidden."""
        m = self.nominal_env.model
        return {
            "actuator_names": list(ACTUATOR_NAMES),
            "actuator_ctrlrange": m.actuator_ctrlrange.tolist(),
            "control_dt": self.nominal_env.control_dt,
            "home_ctrl": self.nominal_env.home_ctrl.tolist(),
            "home_qpos": self.nominal_env.home_qpos.tolist(),
            "nominal_torso_mass": float(m.body_mass[self.nominal_env._torso_body_id]),
            "nominal_foot_friction": 0.8,  # go1.xml default class "foot"
            "experiments_remaining": self.budget - self.experiments_used,
        }

    def run_experiment(self, controller_code: str, duration: float = 3.0, seed: int = 0) -> dict:
        """Runs agent-written controller code against the hidden robot, `controller_code` must define `policy(t, obs)`."""
        if self.experiments_used >= self.budget:
            raise RuntimeError(f"experiment budget exhausted ({self.budget} used)")

        namespace: dict = {"np": np}
        exec(controller_code, namespace)  # noqa: S102, this is a trusted local harness with one user
        policy: PolicyFn = namespace["policy"]

        self.hidden_env.reset(seed=seed)
        n_steps = int(duration / self.hidden_env.control_dt)
        records = []
        for _ in range(n_steps):
            obs = {"qpos": self.hidden_env.data.qpos.copy(), "qvel": self.hidden_env.data.qvel.copy()}
            ctrl = np.asarray(policy(self.hidden_env.data.time, obs), dtype=float)
            records.append(self.hidden_env.step(ctrl))

        trajectory = Trajectory(records=records).to_dict()
        experiment_id = self._next_id
        self._next_id += 1
        self._experiments[experiment_id] = ExperimentRecord(
            experiment_id=experiment_id, controller_code=controller_code, duration=duration, trajectory=trajectory
        )
        self.experiments_used += 1

        return {
            "experiment_id": experiment_id,
            "n_steps": n_steps,
            "experiments_remaining": self.budget - self.experiments_used,
        }

    def inspect_trajectory(self, experiment_id: int) -> dict[str, np.ndarray]:
        """Full recorded trajectory for a past experiment, free, costs no budget."""
        if experiment_id not in self._experiments:
            raise KeyError(f"no experiment with id {experiment_id}")
        return self._experiments[experiment_id].trajectory

    def submit_model(self, parameter_estimate: ParameterEstimate) -> None:
        """Freeze the agent's self-model. Ends the diagnosis phase of the episode."""
        self.parameter_estimate = parameter_estimate

    # bookkeeping for evaluation and baselines

    def true_perturbation(self) -> dict:
        """Ground truth, for evaluation and logging only, never expose this to the agent."""
        return self._true_perturbation.as_dict()

    def experiment_log(self) -> list[dict]:
        return [
            {"experiment_id": rec.experiment_id, "controller_code": rec.controller_code, "duration": rec.duration}
            for rec in self._experiments.values()
        ]
