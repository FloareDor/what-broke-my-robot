"""Re-fits 12 configs with both the fast and slow optimizer on identical data, to show what the random probes actually supported. Slow: budget-8 configs take 30-90 minutes each."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from baselines.random_sysid import collect_random_experiments, fit_parameters, fit_parameters_global
from src.experiment_api import ExperimentSession
from src.metrics import evaluate_predictions, held_out_control_sequences, parameter_estimation_error

FAULT_TYPES = ["friction", "torso_mass", "actuator_strength"]
SEEDS = [1000, 1001]
BUDGETS = [1, 8]
N_RESTARTS = 6
DIAGNOSIS_DURATION = 3.0
OUTPUT = Path("results/optimizer_comparison.csv")


def run_one(fault_kind: str, seed: int, budget: int) -> list[dict]:
    session = ExperimentSession(seed=seed, budget=budget, hidden_kind=fault_kind)
    true_perturbation = session.true_perturbation()
    sequences = held_out_control_sequences(session.nominal_env.control_dt)

    rng = np.random.default_rng(seed)
    collected = collect_random_experiments(session, rng, duration=DIAGNOSIS_DURATION)

    local_estimate = fit_parameters(session.nominal_env, collected)

    t0 = time.time()
    global_estimate = fit_parameters_global(session.nominal_env, collected, n_restarts=N_RESTARTS, seed=seed)
    global_fit_seconds = time.time() - t0

    estimates = {"local_leastsq": local_estimate, "global_multistart": global_estimate}
    candidates = {label: est.to_model(session.nominal_env) for label, est in estimates.items()}
    results = evaluate_predictions(session.hidden_env.model, candidates, sequences)

    rows = []
    for label, metrics in results.items():
        rows.append({
            "fault_kind": fault_kind,
            "seed": seed,
            "budget": budget,
            "method": label,
            "joint_state_error": metrics["joint_state_error"],
            "body_position_error": metrics["body_position_error"],
            "param_error": parameter_estimation_error(true_perturbation, estimates[label]),
            "fit_seconds": global_fit_seconds if label == "global_multistart" else None,
            "true_perturbation_kind": true_perturbation["kind"],
            "true_perturbation_value": true_perturbation["value"],
            "true_perturbation_target": true_perturbation["target"],
        })
    return rows


def main():
    combos = [(fk, seed, budget) for fk in FAULT_TYPES for seed in SEEDS for budget in BUDGETS]

    all_rows: list[dict] = []
    done: set[tuple] = set()
    if OUTPUT.exists():
        existing = pd.read_csv(OUTPUT)
        all_rows = existing.to_dict("records")
        done = {(r["fault_kind"], r["seed"], r["budget"]) for r in all_rows}
        print(f"resuming: {len(done)} configs already in {OUTPUT}", flush=True)

    print(f"running {len(combos)} configs total (this is slow, budget=8 configs are 30 to 90 min each)")
    for i, (fault_kind, seed, budget) in enumerate(combos, start=1):
        if (fault_kind, seed, budget) in done:
            print(f"[{i}/{len(combos)}] skip (already done) fault={fault_kind} seed={seed} budget={budget}", flush=True)
            continue

        print(f"[{i}/{len(combos)}] fault={fault_kind} seed={seed} budget={budget}", flush=True)
        t0 = time.time()
        rows = run_one(fault_kind, seed, budget)
        all_rows.extend(rows)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(OUTPUT, index=False)
        print(f"  done in {time.time() - t0:.1f}s", flush=True)

    print(f"\nwrote {len(all_rows)} rows to {OUTPUT}")
    df = pd.DataFrame(all_rows)
    print("\nmean param_error by method x budget:")
    print(df.pivot_table(index="budget", columns="method", values="param_error", aggfunc="mean").to_string())
    print("\nmean body_position_error by method x budget:")
    print(df.pivot_table(index="budget", columns="method", values="body_position_error", aggfunc="mean").to_string())


if __name__ == "__main__":
    main()
