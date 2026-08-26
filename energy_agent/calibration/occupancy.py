"""Single-parameter occupancy calibration using repeated EnergyPlus runs."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Awaitable, Callable

from energy_agent.analytics.metrics import calculate_calibration_metrics
from energy_agent.analytics.timeseries import align_series, load_measured_csv
from energy_agent.simulation.runner import EnergyPlusRunner, file_sha256

ProgressCallback = Callable[[dict], Awaitable[None]]


@dataclass(frozen=True)
class OccupancyCalibrationRequest:
    idf_path: str
    epw_path: str
    measured_data_path: str
    measured_value_column: str = "electricity"
    timestamp_column: str = "timestamp"
    measured_unit: str = "kWh"
    lower_bound: float = 0.5
    upper_bound: float = 1.5
    maximum_evaluations: int = 12
    objective: str = "rmse"
    alignment_policy: str = "month_day_time"
    stopping_tolerance: float = 0.001
    rmse_target: float | None = None
    wall_clock_timeout_seconds: int = 7200
    output_directory: str = "runtime/calibration"

    def validate(self) -> None:
        if not 0 < self.lower_bound < self.upper_bound:
            raise ValueError("Occupancy bounds must satisfy 0 < lower < upper")
        if not self.lower_bound <= 1.0 <= self.upper_bound:
            raise ValueError("Bounds must contain the baseline occupancy multiplier 1.0")
        if not 3 <= self.maximum_evaluations <= 100:
            raise ValueError("maximum_evaluations must be between 3 and 100")
        if self.objective not in {"rmse", "cvrmse_percent", "abs_nmbe_percent"}:
            raise ValueError("Unsupported calibration objective")
        if self.wall_clock_timeout_seconds < 60:
            raise ValueError("wall_clock_timeout_seconds must be at least 60")


def _objective(metrics: dict, name: str) -> float:
    return abs(metrics["nmbe_percent"]) if name == "abs_nmbe_percent" else float(metrics[name])


def _next_candidate(evaluations: list[dict], lower: float, upper: float, tolerance: float) -> float | None:
    successful = sorted((row for row in evaluations if row["status"] == "success"), key=lambda row: row["occupancy_multiplier"])
    if not successful:
        return (lower + upper) / 2
    best = min(successful, key=lambda row: row["objective"])
    values = [lower, *[row["occupancy_multiplier"] for row in successful], upper]
    values = sorted(set(round(value, 12) for value in values))
    index = values.index(round(best["occupancy_multiplier"], 12))
    candidates = []
    if index > 0: candidates.append((values[index - 1] + values[index]) / 2)
    if index < len(values) - 1: candidates.append((values[index] + values[index + 1]) / 2)
    seen = {round(row["occupancy_multiplier"], 10) for row in evaluations}
    candidates = [value for value in candidates if round(value, 10) not in seen and min(abs(value-x) for x in values) > tolerance]
    return max(candidates, key=lambda value: min(abs(value-x) for x in values), default=None)


async def calibrate_occupancy(request: OccupancyCalibrationRequest, runner: EnergyPlusRunner, progress: ProgressCallback | None = None) -> dict:
    request.validate()
    started = time.monotonic()
    measured = load_measured_csv(request.measured_data_path, value_column=request.measured_value_column,
                                 timestamp_column=request.timestamp_column, unit=request.measured_unit)
    idf = runner.resolve_file(request.idf_path, ".idf")
    epw = runner.resolve_file(request.epw_path, ".epw")
    output_root = Path(request.output_directory).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    inspection = await runner.inspect_people(idf)
    evaluations: list[dict] = []
    cache: dict[str, dict] = {}
    failed = 0

    async def report(payload: dict) -> None:
        if progress:
            await progress(payload)

    async def evaluate(multiplier: float) -> dict:
        nonlocal failed
        key = f"{multiplier:.10f}"
        if key in cache:
            return cache[key]
        number = len(evaluations) + 1
        run_dir = output_root / ("baseline" if multiplier == 1.0 and not evaluations else f"iteration_{number:03d}")
        raw_model = run_dir / f"occupancy_{multiplier:.6f}_raw.idf"
        model = run_dir / f"occupancy_{multiplier:.6f}.idf"
        await report({"stage": "evaluating", "evaluation": number, "maximum_evaluations": request.maximum_evaluations,
                      "occupancy_multiplier": multiplier})
        try:
            trace = await runner.apply_occupancy_multiplier(idf, raw_model, inspection, multiplier)
            await runner.configure_hourly_electricity(raw_model, model)
            evidence = await runner.simulate_electricity(model, epw, run_dir / "simulation", measured.timestamps[0].year)
            measured_values, simulated_values, alignment = align_series(measured, evidence.series, policy=request.alignment_policy)
            metrics = calculate_calibration_metrics(measured_values, simulated_values, unit=measured.unit, fitted_parameters=1)
            row = {"evaluation": number, "occupancy_multiplier": multiplier, "status": "success",
                   "objective": _objective(metrics, request.objective), "metrics": metrics,
                   "alignment": alignment, "model_path": str(model), "output_directory": evidence.output_directory,
                   "meter_csv": evidence.meter_csv, "modifications": trace}
        except Exception as exc:
            failed += 1
            row = {"evaluation": number, "occupancy_multiplier": multiplier, "status": "simulation_failed",
                   "objective": 1_000_000_000.0, "error": str(exc), "model_path": str(model),
                   "output_directory": str(run_dir / "simulation")}
        evaluations.append(row); cache[key] = row
        (output_root / "trace.json").write_text(json.dumps(evaluations, indent=2), encoding="utf-8")
        await report({"stage": "evaluation_completed", **{k: row.get(k) for k in ("evaluation", "occupancy_multiplier", "status", "objective")}})
        return row

    baseline = await evaluate(1.0)
    for initial in (request.lower_bound, request.upper_bound, (request.lower_bound + request.upper_bound) / 2):
        if len(evaluations) >= request.maximum_evaluations: break
        if round(initial, 10) not in {round(row["occupancy_multiplier"], 10) for row in evaluations}:
            await evaluate(initial)
    stopping_reason = "maximum_evaluations"
    while len(evaluations) < request.maximum_evaluations:
        if time.monotonic() - started >= request.wall_clock_timeout_seconds:
            stopping_reason = "wall_clock_timeout"; break
        successes = [row for row in evaluations if row["status"] == "success"]
        if successes and request.rmse_target is not None and min(row["metrics"]["rmse"] for row in successes) <= request.rmse_target:
            stopping_reason = "rmse_target"; break
        candidate = _next_candidate(evaluations, request.lower_bound, request.upper_bound, request.stopping_tolerance)
        if candidate is None:
            stopping_reason = "parameter_tolerance"; break
        await evaluate(candidate)
    successes = [row for row in evaluations if row["status"] == "success"]
    if not successes:
        raise RuntimeError("Every EnergyPlus calibration evaluation failed")
    best = min(successes, key=lambda row: row["objective"])
    improvement = 100.0 * (baseline["objective"] - best["objective"]) / baseline["objective"] if baseline["objective"] else 0.0
    result = {
        "status": "success", "mode": "energyplus-backed-real-calibration",
        "parameter": "occupancy_multiplier", "optimizer": "bounded-grid-refinement-v1",
        "objective_name": request.objective, "baseline": baseline, "calibrated": best,
        "improvement_percent": improvement, "energyplus_evaluations": len(evaluations),
        "failed_evaluations": failed, "converged": stopping_reason != "maximum_evaluations",
        "stopping_reason": stopping_reason, "execution_time_seconds": round(time.monotonic() - started, 3),
        "iterations": evaluations,
        "provenance": {"original_model": str(idf), "original_model_sha256": file_sha256(idf),
                       "weather_file": str(epw), "weather_sha256": file_sha256(epw),
                       "measured_data": measured.source, "measured_data_sha256": file_sha256(measured.source),
                       "meter": "Electricity:Facility", "request": asdict(request)},
    }
    (output_root / "calibration_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_root / "calibration_iterations.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["evaluation", "occupancy_multiplier", "status", "objective", "rmse", "nmbe_percent", "cvrmse_percent", "error"])
        writer.writeheader()
        for row in evaluations:
            metrics = row.get("metrics", {})
            writer.writerow({"evaluation": row["evaluation"], "occupancy_multiplier": row["occupancy_multiplier"],
                             "status": row["status"], "objective": row["objective"], "rmse": metrics.get("rmse"),
                             "nmbe_percent": metrics.get("nmbe_percent"), "cvrmse_percent": metrics.get("cvrmse_percent"),
                             "error": row.get("error", "")})
    await report({"stage": "completed", "result": result})
    return result
