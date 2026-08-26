"""Numerical calibration metrics; no LLM or simulation behavior belongs here."""
from __future__ import annotations

import math
from collections.abc import Sequence


def _validated(measured: Sequence[float], simulated: Sequence[float]) -> tuple[list[float], list[float]]:
    if len(measured) != len(simulated):
        raise ValueError("Measured and simulated series must have equal lengths")
    if len(measured) < 2:
        raise ValueError("At least two aligned samples are required")
    m = [float(value) for value in measured]
    s = [float(value) for value in simulated]
    if not all(math.isfinite(value) for value in (*m, *s)):
        raise ValueError("Measured and simulated series cannot contain NaN or infinity")
    return m, s


def calculate_rmse(measured: Sequence[float], simulated: Sequence[float]) -> float:
    measured_values, simulated_values = _validated(measured, simulated)
    return math.sqrt(sum((s - m) ** 2 for m, s in zip(measured_values, simulated_values)) / len(measured_values))


def calculate_mae(measured: Sequence[float], simulated: Sequence[float]) -> float:
    measured_values, simulated_values = _validated(measured, simulated)
    return sum(abs(s - m) for m, s in zip(measured_values, simulated_values)) / len(measured_values)


def calculate_mape(measured: Sequence[float], simulated: Sequence[float]) -> tuple[float | None, int]:
    measured_values, simulated_values = _validated(measured, simulated)
    usable = [(m, s) for m, s in zip(measured_values, simulated_values) if m != 0]
    excluded = len(measured_values) - len(usable)
    if not usable:
        return None, excluded
    return 100.0 * sum(abs((s - m) / m) for m, s in usable) / len(usable), excluded


def calculate_nmbe(measured: Sequence[float], simulated: Sequence[float], fitted_parameters: int = 1) -> float:
    measured_values, simulated_values = _validated(measured, simulated)
    denominator_count = len(measured_values) - int(fitted_parameters)
    mean = sum(measured_values) / len(measured_values)
    if denominator_count <= 0:
        raise ValueError("Sample count must exceed fitted parameter count")
    if mean == 0:
        raise ValueError("Measured mean cannot be zero for NMBE")
    return 100.0 * sum(s - m for m, s in zip(measured_values, simulated_values)) / (denominator_count * mean)


def calculate_cvrmse(measured: Sequence[float], simulated: Sequence[float], fitted_parameters: int = 1) -> float:
    measured_values, simulated_values = _validated(measured, simulated)
    denominator_count = len(measured_values) - int(fitted_parameters)
    mean = sum(measured_values) / len(measured_values)
    if denominator_count <= 0:
        raise ValueError("Sample count must exceed fitted parameter count")
    if mean == 0:
        raise ValueError("Measured mean cannot be zero for CV(RMSE)")
    squared_error = sum((s - m) ** 2 for m, s in zip(measured_values, simulated_values))
    return 100.0 * math.sqrt(squared_error / denominator_count) / abs(mean)


def calculate_calibration_metrics(
    measured: Sequence[float], simulated: Sequence[float], *, unit: str, fitted_parameters: int = 1
) -> dict:
    """Return engineering metrics for already-aligned arrays.

    RMSE and MAE retain ``unit``. Percentage metrics are dimensionless. NMBE
    and CV(RMSE) use n-p degrees of freedom, where p is ``fitted_parameters``.
    """
    measured_values, simulated_values = _validated(measured, simulated)
    mape, zero_count = calculate_mape(measured_values, simulated_values)
    return {
        "rmse": calculate_rmse(measured_values, simulated_values),
        "mae": calculate_mae(measured_values, simulated_values),
        "mape_percent": mape,
        "mape_zero_measured_excluded": zero_count,
        "nmbe_percent": calculate_nmbe(measured_values, simulated_values, fitted_parameters),
        "cvrmse_percent": calculate_cvrmse(measured_values, simulated_values, fitted_parameters),
        "sample_count": len(measured_values),
        "fitted_parameters": fitted_parameters,
        "value_unit": unit,
    }
