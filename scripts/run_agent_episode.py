"""Runs nominal, random probing, and the Gemini agent against the same hidden fault. Needs GEMINI_API_KEY set."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.coding_agent import CodingAgent
from baselines.nominal import run_nominal_baseline
from baselines.random_sysid import run_random_baseline
from src.experiment_api import ExperimentSession
from src.metrics import evaluate_predictions, held_out_control_sequences


def main():
    episode_seed = 42
    budget = 8
    hidden_kind = "actuator_strength"
    diagnosis_duration = 3.0

    # same seed and hidden_kind means both sample the identical fault, separate sessions so each gets its own budget
    random_session = ExperimentSession(seed=episode_seed, budget=budget, hidden_kind=hidden_kind)
    agent_session = ExperimentSession(seed=episode_seed, budget=budget, hidden_kind=hidden_kind)
    assert random_session.true_perturbation() == agent_session.true_perturbation()

    print("Hidden perturbation (ground truth, hidden from everything below):")
    print(" ", random_session.true_perturbation())
    print()

    nominal_estimate = run_nominal_baseline()

    random_estimate = run_random_baseline(random_session, seed=episode_seed, duration=diagnosis_duration)
    print(f"Random-probing estimate (after {random_session.experiments_used}/{budget} experiments):")
    print(" ", random_estimate.as_dict())
    print()

    print("Coding agent (Gemini) starting, this calls the real API and may take a while...")
    print()
    agent = CodingAgent(agent_session)
    agent_estimate = agent.run()
    print()
    print(f"Coding agent estimate (after {agent_session.experiments_used}/{budget} experiments):")
    print(" ", agent_estimate.as_dict())
    print()

    held_out_sequences = held_out_control_sequences(agent_session.nominal_env.control_dt)

    candidate_models = {
        "nominal (no adaptation)": nominal_estimate.to_model(agent_session.nominal_env),
        "random_sysid": random_estimate.to_model(agent_session.nominal_env),
        "coding_agent (gemini)": agent_estimate.to_model(agent_session.nominal_env),
    }

    results = evaluate_predictions(agent_session.hidden_env.model, candidate_models, held_out_sequences)

    print("Held-out prediction error vs hidden robot:")
    for label, metrics in results.items():
        print(f"  {label:26s} joint_state_error={metrics['joint_state_error']:.6f}  body_position_error={metrics['body_position_error']:.6f}")

    print()
    print("Experiments the agent chose to run:")
    for rec in agent_session.experiment_log():
        print(f"  experiment {rec['experiment_id']}: duration={rec['duration']}s")


if __name__ == "__main__":
    main()
