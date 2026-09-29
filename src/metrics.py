"""Runs held-out control sequences on the hidden robot vs a candidate model and compares the trajectories."""

from __future__ import annotations

import numpy as np

from src.controller import TrotController
from src.env import Go1Env


def rollout_fixed_controls(model, ctrl_sequence: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    """Run an open-loop [T, 12] control sequence on `model`, starting from the home pose."""
    env = Go1Env(model=model)
    env.reset(seed=seed)
    records = [env.step(ctrl) for ctrl in ctrl_sequence]
    from src.env import Trajectory

    traj = Trajectory(records=records)
    return traj.to_dict()


def joint_state_error(traj_a: dict, traj_b: dict) -> float:
    """Mean squared error over the 12 actuated-joint positions (qpos[7:])."""
    a = traj_a["qpos"][:, 7:]
    b = traj_b["qpos"][:, 7:]
    return float(np.mean((a - b) ** 2))


def body_position_error(traj_a: dict, traj_b: dict) -> float:
    """Mean Euclidean error between torso positions over time."""
    a = traj_a["torso_pos"]
    b = traj_b["torso_pos"]
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def evaluate_predictions(
    hidden_model,
    candidate_models: dict[str, object],
    held_out_ctrl_sequences: list[np.ndarray],
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Compares each candidate model's held-out prediction error against the hidden robot, averaged over the sequences."""
    results: dict[str, dict[str, float]] = {label: {"joint_state_error": 0.0, "body_position_error": 0.0} for label in candidate_models}

    for ctrl_seq in held_out_ctrl_sequences:
        hidden_traj = rollout_fixed_controls(hidden_model, ctrl_seq, seed=seed)
        for label, model in candidate_models.items():
            cand_traj = rollout_fixed_controls(model, ctrl_seq, seed=seed)
            results[label]["joint_state_error"] += joint_state_error(hidden_traj, cand_traj)
            results[label]["body_position_error"] += body_position_error(hidden_traj, cand_traj)

    n = len(held_out_ctrl_sequences)
    for label in results:
        results[label]["joint_state_error"] /= n
        results[label]["body_position_error"] /= n

    return results


def controller_to_ctrl_sequence(controller, duration: float, control_dt: float) -> np.ndarray:
    n_steps = int(duration / control_dt)
    return np.stack([controller(i * control_dt) for i in range(n_steps)])


def held_out_control_sequences(control_dt: float, duration: float = 3.0) -> list[np.ndarray]:
    """The fixed held-out sequences every episode is scored against, kept in one place so scripts can't drift apart."""
    return [
        controller_to_ctrl_sequence(TrotController(frequency_hz=0.8), duration, control_dt),
        controller_to_ctrl_sequence(TrotController(frequency_hz=1.1, hip_amplitude=0.2), duration, control_dt),
    ]


def parameter_estimation_error(true_perturbation: dict, estimate) -> float:
    """Absolute error between a guess and the true fault, leaving a field blank is scored against the nominal value."""
    kind = true_perturbation["kind"]
    true_value = true_perturbation["value"]
    if kind == "friction":
        guess = estimate.friction if estimate.friction is not None else 0.8
    elif kind == "torso_mass":
        guess = estimate.torso_mass_scale if estimate.torso_mass_scale is not None else 1.0
    elif kind == "actuator_strength":
        guess = estimate.actuator_scale.get(true_perturbation["target"], 1.0)
    else:
        raise ValueError(f"unknown perturbation kind: {kind}")
    return float(abs(guess - true_value))
