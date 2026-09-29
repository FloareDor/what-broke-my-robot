"""Turns a benchmark CSV into three charts: error vs budget, parameter error vs budget, and error by fault type."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # headless: this script only ever saves PNGs, never shows a window
import matplotlib.pyplot as plt
import pandas as pd

METHOD_COLORS = {
    "nominal": "tab:gray",
    "random_sysid": "tab:blue",
    "coding_agent": "tab:orange",
}
METHOD_ORDER = ["nominal", "random_sysid", "coding_agent"]
# internal method names -> plain-language legend labels
METHOD_LABELS = {
    "nominal": "doing nothing",
    "random_sysid": "random probing",
    "coding_agent": "coding agent",
}


def _methods_present(df: pd.DataFrame) -> list[str]:
    present = set(df["method"].unique())
    return [m for m in METHOD_ORDER if m in present]


def plot_error_vs_budget(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    methods = _methods_present(df)
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for method in methods:
        sub = df[df["method"] == method]
        grouped = sub.groupby("budget")[metric].agg(["mean", "sem"]).reset_index().sort_values("budget")
        ax.errorbar(
            grouped["budget"], grouped["mean"], yerr=grouped["sem"].fillna(0.0),
            label=METHOD_LABELS.get(method, method), marker="o", color=METHOD_COLORS.get(method), capsize=3,
        )

    ax.set_xlabel("experiment budget")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs experiment budget\n(mean +/- s.e. across fault types and seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_error_by_fault_type(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    methods = _methods_present(df)
    fault_kinds = sorted(df["fault_kind"].unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    x = range(len(fault_kinds))

    for i, method in enumerate(methods):
        means, sems = [], []
        for fk in fault_kinds:
            sub = df[(df["method"] == method) & (df["fault_kind"] == fk)][metric]
            means.append(sub.mean())
            sems.append(sub.sem() if len(sub) > 1 else 0.0)
        offsets = [xi + (i - (n_methods - 1) / 2) * bar_width for xi in x]
        ax.bar(offsets, means, width=bar_width, yerr=sems, capsize=3, label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method))

    ax.set_xticks(list(x))
    ax.set_xticklabels(fault_kinds)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by fault type\n(mean +/- s.e. across seeds and budgets)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/benchmark.csv")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"{in_path} doesn't exist yet, run scripts/run_benchmark.py first")

    df = pd.read_csv(in_path)
    if df.empty:
        raise SystemExit(f"{in_path} is empty, run scripts/run_benchmark.py first")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_error_vs_budget(df, "body_position_error", "held-out body position error", out_dir / "held_out_error_vs_budget.png")
    plot_error_vs_budget(df, "param_error", "parameter estimation error", out_dir / "param_error_vs_budget.png")
    plot_error_by_fault_type(df, "body_position_error", "held-out body position error", out_dir / "error_by_fault_type.png")

    print("\nsummary (mean body_position_error by method x budget):")
    print(df.pivot_table(index="budget", columns="method", values="body_position_error", aggfunc="mean").to_string())


if __name__ == "__main__":
    main()
