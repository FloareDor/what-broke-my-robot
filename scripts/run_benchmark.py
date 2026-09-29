"""Sweeps fault types x seeds x budgets, runs the baselines and optionally the real agent, writes one CSV row per config."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from baselines.nominal import run_nominal_baseline
from baselines.random_sysid import run_random_baseline
from src.experiment_api import ExperimentSession
from src.metrics import evaluate_predictions, held_out_control_sequences, parameter_estimation_error


def run_one(fault_kind: str, seed: int, budget: int, include_agent: bool, agent_verbose: bool, diagnosis_duration: float, traces_dir: Path | None) -> list[dict]:
    random_session = ExperimentSession(seed=seed, budget=budget, hidden_kind=fault_kind)
    true_perturbation = random_session.true_perturbation()
    sequences = held_out_control_sequences(random_session.nominal_env.control_dt)

    nominal_estimate = run_nominal_baseline()
    random_estimate = run_random_baseline(random_session, seed=seed, duration=diagnosis_duration)

    estimates = {"nominal": nominal_estimate, "random_sysid": random_estimate}
    candidates = {label: est.to_model(random_session.nominal_env) for label, est in estimates.items()}
    experiments_used = {"nominal": 0, "random_sysid": random_session.experiments_used}

    if include_agent:
        from agents.coding_agent import CodingAgent

        agent_session = ExperimentSession(seed=seed, budget=budget, hidden_kind=fault_kind)
        agent = CodingAgent(agent_session, verbose=agent_verbose)
        agent_estimate = agent.run()
        estimates["coding_agent"] = agent_estimate
        candidates["coding_agent"] = agent_estimate.to_model(agent_session.nominal_env)
        experiments_used["coding_agent"] = agent_session.experiments_used

        if traces_dir is not None:
            traces_dir.mkdir(parents=True, exist_ok=True)
            trace_path = traces_dir / f"{fault_kind}_seed{seed}_budget{budget}.json"
            trace_path.write_text(json.dumps({
                "true_perturbation": true_perturbation,
                "estimate": agent_estimate.as_dict(),
                "transcript": agent.transcript,
            }, indent=2, default=str))

    results = evaluate_predictions(random_session.hidden_env.model, candidates, sequences)

    rows = []
    for label, metrics in results.items():
        rows.append({
            "fault_kind": fault_kind,
            "seed": seed,
            "budget": budget,
            "method": label,
            "experiments_used": experiments_used[label],
            "joint_state_error": metrics["joint_state_error"],
            "body_position_error": metrics["body_position_error"],
            "param_error": parameter_estimation_error(true_perturbation, estimates[label]),
            "true_perturbation_kind": true_perturbation["kind"],
            "true_perturbation_value": true_perturbation["value"],
            "true_perturbation_target": true_perturbation["target"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault-types", nargs="+", default=["friction", "torso_mass", "actuator_strength"])
    parser.add_argument("--n-seeds", type=int, default=5, help="random hidden-fault values per fault type (each seed samples one)")
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seed-base", type=int, default=1000, help="seeds used are seed_base, seed_base+1, ...")
    parser.add_argument("--diagnosis-duration", type=float, default=3.0)
    parser.add_argument("--include-agent", action="store_true", help="also run the real Gemini coding agent (slow + rate limited, uses API quota)")
    parser.add_argument("--agent-verbose", action="store_true", help="print the agent's tool calls as it goes")
    parser.add_argument("--output", default="results/benchmark.csv")
    parser.add_argument("--traces-dir", default="results/traces")
    parser.add_argument("--resume", action="store_true", help="skip (fault_kind, seed, budget) triples already present in --output")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traces_dir = Path(args.traces_dir) if args.include_agent else None

    all_rows: list[dict] = []
    done: set[tuple] = set()
    if args.resume and out_path.exists():
        existing = pd.read_csv(out_path)
        all_rows = existing.to_dict("records")
        done = {(r["fault_kind"], r["seed"], r["budget"]) for r in all_rows}

    seeds = [args.seed_base + i for i in range(args.n_seeds)]
    combos = [(fk, seed, budget) for fk in args.fault_types for seed in seeds for budget in args.budgets]

    print(f"running {len(combos)} configs ({len(args.fault_types)} fault types x {args.n_seeds} seeds x {len(args.budgets)} budgets)")
    if args.include_agent:
        print("--include-agent is on: this makes real Gemini API calls and will be slow.")

    for i, (fault_kind, seed, budget) in enumerate(combos, start=1):
        if (fault_kind, seed, budget) in done:
            print(f"[{i}/{len(combos)}] skip (resume) fault={fault_kind} seed={seed} budget={budget}")
            continue

        print(f"[{i}/{len(combos)}] fault={fault_kind} seed={seed} budget={budget}")
        t0 = time.time()
        try:
            rows = run_one(
                fault_kind, seed, budget,
                include_agent=args.include_agent,
                agent_verbose=args.agent_verbose,
                diagnosis_duration=args.diagnosis_duration,
                traces_dir=traces_dir,
            )
        except Exception as exc:  # noqa: BLE001, one failed config shouldn't kill the whole sweep
            print(f"  FAILED: {exc}")
            continue

        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
        print(f"  done in {time.time() - t0:.1f}s")

    print(f"\nwrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
