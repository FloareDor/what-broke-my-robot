"""Baseline A: no adaptation. The self-model is just the nominal model."""

from __future__ import annotations

from src.prediction import ParameterEstimate


def run_nominal_baseline() -> ParameterEstimate:
    return ParameterEstimate()  # every field None -> unchanged from nominal
