"""Baseline B: random safe probing, then fit the result with scipy.optimize.least_squares, same budget as the agent."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from src.env import ACTUATOR_NAMES
from src.experiment_api import ExperimentSession
from src.metrics import rollout_fixed_controls
from src.prediction import ParameterEstimate

FRICTION_BOUNDS = (0.3, 1.2)
TORSO_MASS_BOUNDS = (0.7, 1.4)
ACTUATOR_BOUNDS = (0.5, 1.0)


def _random_probe_code(rng: np.random.Generator, n_steps: int, control_dt: float, ctrlrange: np.ndarray, amplitude: float = 0.25) -> str:
    """Smoothed random excitation around the home pose, clipped to each actuator's ctrlrange."""
    from src.controller import HOME_CTRL

    noise = rng.normal(0.0, amplitude, size=(n_steps, len(ACTUATOR_NAMES)))
    kernel = np.ones(5) / 5.0
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=noise)
    ctrl_table = HOME_CTRL + smoothed
    ctrl_table = np.clip(ctrl_table, ctrlrange[:, 0], ctrlrange[:, 1])

    return (
        f"_ctrl_table = np.array({ctrl_table.tolist()})\n"
        f"_dt = {control_dt}\n"
        "def policy(t, obs):\n"
        "    idx = min(int(round(t / _dt)), len(_ctrl_table) - 1)\n"
        "    return _ctrl_table[idx]\n"
    )


def collect_random_experiments(session: ExperimentSession, rng: np.random.Generator, duration: float = 3.0) -> list[dict]:
    """Spend the session's full remaining budget on random probing."""
    ctrlrange = session.nominal_env.model.actuator_ctrlrange
    n_steps = int(duration / session.nominal_env.control_dt)
    collected = []
    while session.experiments_used < session.budget:
        code = _random_probe_code(rng, n_steps, session.nominal_env.control_dt, ctrlrange)
        result = session.run_experiment(code, duration=duration, seed=int(rng.integers(0, 1_000_000)))
        collected.append(session.inspect_trajectory(result["experiment_id"]))
    return collected


def fit_parameters(nominal_env, collected_trajectories: list[dict], reg_weight: float = 0.15) -> ParameterEstimate:
    """Fits friction, torso mass, and actuator strength by replaying each trajectory and minimizing the joint-position residual, with a ridge penalty (`reg_weight`) pulling unmoved parameters back toward nominal."""
    from src.perturbations import set_actuator_strength_scale, set_friction, set_torso_mass_scale

    lo = [FRICTION_BOUNDS[0], TORSO_MASS_BOUNDS[0]] + [ACTUATOR_BOUNDS[0]] * len(ACTUATOR_NAMES)
    hi = [FRICTION_BOUNDS[1], TORSO_MASS_BOUNDS[1]] + [ACTUATOR_BOUNDS[1]] * len(ACTUATOR_NAMES)
    theta_nominal = np.array([0.8, 1.0] + [1.0] * len(ACTUATOR_NAMES))
    # actuator nominal sits exactly on the upper bound, a bad starting point for scipy, so nudge off it
    theta0 = theta_nominal.copy()
    theta0[2:] -= 1e-3

    def residual(theta: np.ndarray) -> np.ndarray:
        friction, mass_scale = theta[0], theta[1]
        actuator_scales = theta[2:]
        model = nominal_env.clone_model()
        set_friction(model, friction)
        set_torso_mass_scale(model, mass_scale)
        for name, scale in zip(ACTUATOR_NAMES, actuator_scales):
            set_actuator_strength_scale(model, name, scale)

        residuals = []
        for traj in collected_trajectories:
            sim_traj = rollout_fixed_controls(model, traj["ctrl"])
            residuals.append((sim_traj["qpos"][:, 7:] - traj["qpos"][:, 7:]).ravel())
        residuals.append(reg_weight * (theta - theta_nominal))
        return np.concatenate(residuals)

    result = least_squares(residual, theta0, bounds=(lo, hi), verbose=0)
    friction, mass_scale = result.x[0], result.x[1]
    actuator_scales = dict(zip(ACTUATOR_NAMES, result.x[2:]))
    return ParameterEstimate(friction=float(friction), torso_mass_scale=float(mass_scale), actuator_scale={k: float(v) for k, v in actuator_scales.items()})


def fit_parameters_global(
    nominal_env,
    collected_trajectories: list[dict],
    reg_weight: float = 0.15,
    n_restarts: int = 6,
    seed: int = 0,
) -> ParameterEstimate:
    """Same fit as `fit_parameters`, but restarted from several points instead of one, much slower but never worse, nominal is always included as a restart so this can only improve on it."""
    from src.perturbations import set_actuator_strength_scale, set_friction, set_torso_mass_scale

    lo = np.array([FRICTION_BOUNDS[0], TORSO_MASS_BOUNDS[0]] + [ACTUATOR_BOUNDS[0]] * len(ACTUATOR_NAMES))
    hi = np.array([FRICTION_BOUNDS[1], TORSO_MASS_BOUNDS[1]] + [ACTUATOR_BOUNDS[1]] * len(ACTUATOR_NAMES))
    theta_nominal = np.array([0.8, 1.0] + [1.0] * len(ACTUATOR_NAMES))

    def residual(theta: np.ndarray) -> np.ndarray:
        friction, mass_scale = theta[0], theta[1]
        actuator_scales = theta[2:]
        model = nominal_env.clone_model()
        set_friction(model, friction)
        set_torso_mass_scale(model, mass_scale)
        for name, scale in zip(ACTUATOR_NAMES, actuator_scales):
            set_actuator_strength_scale(model, name, scale)

        residuals = []
        for traj in collected_trajectories:
            sim_traj = rollout_fixed_controls(model, traj["ctrl"])
            residuals.append((sim_traj["qpos"][:, 7:] - traj["qpos"][:, 7:]).ravel())
        residuals.append(reg_weight * (theta - theta_nominal))
        return np.concatenate(residuals)

    rng = np.random.default_rng(seed)
    starts = [theta_nominal] + [rng.uniform(lo, hi) for _ in range(n_restarts - 1)]
    best = None
    for theta0 in starts:
        result = least_squares(residual, theta0, bounds=(lo, hi), verbose=0)
        if best is None or result.cost < best.cost:
            best = result

    friction, mass_scale = best.x[0], best.x[1]
    actuator_scales = dict(zip(ACTUATOR_NAMES, best.x[2:]))
    return ParameterEstimate(friction=float(friction), torso_mass_scale=float(mass_scale), actuator_scale={k: float(v) for k, v in actuator_scales.items()})


def run_random_baseline(session: ExperimentSession, seed: int = 0, duration: float = 3.0) -> ParameterEstimate:
    rng = np.random.default_rng(seed)
    collected = collect_random_experiments(session, rng, duration=duration)
    return fit_parameters(session.nominal_env, collected)
