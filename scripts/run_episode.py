"""Runs baseline A (no adaptation) and baseline B (random probing) against one hidden robot, compares held-out error."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from baselines.nominal import run_nominal_baseline
from baselines.random_sysid import run_random_baseline
from src.controller import TrotController
from src.experiment_api import ExperimentSession
from src.metrics import evaluate_predictions


def controller_to_ctrl_sequence(controller, duration: float, control_dt: float) -> np.ndarray:
    n_steps = int(duration / control_dt)
    return np.stack([controller(i * control_dt) for i in range(n_steps)])


def main():
    episode_seed = 42
    budget = 8
    diagnosis_duration = 3.0

    session = ExperimentSession(seed=episode_seed, budget=budget, hidden_kind="actuator_strength")
    print("Hidden perturbation (ground truth, hidden from all baselines):")
    print(" ", session.true_perturbation())
    print()

    nominal_estimate = run_nominal_baseline()

    random_estimate = run_random_baseline(session, seed=episode_seed, duration=diagnosis_duration)
    print(f"Random-probing estimate (after {session.experiments_used}/{budget} experiments):")
    print(" ", random_estimate.as_dict())
    print()

    # neither baseline saw these sequences during diagnosis
    held_out_sequences = [
        controller_to_ctrl_sequence(TrotController(frequency_hz=0.8), 3.0, session.nominal_env.control_dt),
        controller_to_ctrl_sequence(TrotController(frequency_hz=1.1, hip_amplitude=0.2), 3.0, session.nominal_env.control_dt),
    ]

    candidate_models = {
        "nominal (no adaptation)": nominal_estimate.to_model(session.nominal_env),
        "random_sysid": random_estimate.to_model(session.nominal_env),
    }

    results = evaluate_predictions(session.hidden_env.model, candidate_models, held_out_sequences)

    print("Held-out prediction error vs hidden robot:")
    for label, metrics in results.items():
        print(f"  {label:24s} joint_state_error={metrics['joint_state_error']:.6f}  body_position_error={metrics['body_position_error']:.6f}")


if __name__ == "__main__":
    main()
