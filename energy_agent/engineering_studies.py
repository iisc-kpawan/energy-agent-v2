"""Deterministic EnergyPlus-backed optimization and sensitivity studies."""
from __future__ import annotations

import asyncio
import json
import secrets
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from langchain_core.tools import tool

from energy_agent.simulation.runner import EnergyPlusRunner, file_sha256

Progress = Callable[[dict], Awaitable[None]]


@dataclass(frozen=True)
class EnergyPlusStudyRequest:
    idf_path: str
    epw_path: str
    parameter: str = "lighting_multiplier"
    lower_bound: float = 0.7
    upper_bound: float = 1.0
    maximum_evaluations: int = 5
    wall_clock_timeout_seconds: int = 3600
    output_directory: str = "runtime/engineering_studies"

    def validate(self, kind: str) -> None:
        if self.parameter not in {"lighting_multiplier", "occupancy_multiplier"}:
            raise ValueError("parameter must be lighting_multiplier or occupancy_multiplier")
        if kind == "optimization" and self.parameter != "lighting_multiplier":
            raise ValueError("V2 real optimization currently supports lighting_multiplier only")
        if not 0 < self.lower_bound < self.upper_bound:
            raise ValueError("Bounds must satisfy 0 < lower_bound < upper_bound")
        if not self.lower_bound <= 1.0 <= self.upper_bound:
            raise ValueError("Bounds must include baseline multiplier 1.0")
        if not 3 <= self.maximum_evaluations <= 25:
            raise ValueError("maximum_evaluations must be between 3 and 25")
        if self.wall_clock_timeout_seconds < 60:
            raise ValueError("wall_clock_timeout_seconds must be at least 60")


async def _evaluate(
    request: EnergyPlusStudyRequest, runner: EnergyPlusRunner, original: Path, weather: Path,
    inspection: dict | None, multiplier: float, number: int, root: Path,
) -> dict:
    run_dir = root / ("baseline" if multiplier == 1.0 else f"evaluation_{number:03d}")
    raw = run_dir / f"{request.parameter}_{multiplier:.6f}_raw.idf"
    model = run_dir / f"{request.parameter}_{multiplier:.6f}.idf"
    try:
        if request.parameter == "lighting_multiplier":
            changes = await runner.apply_lighting_multiplier(original, raw, multiplier)
        else:
            changes = await runner.apply_occupancy_multiplier(original, raw, inspection or {}, multiplier)
        await runner.configure_hourly_electricity(raw, model)
        evidence = await runner.simulate_electricity(model, weather, run_dir / "simulation", 2026)
        annual = sum(evidence.series.values)
        return {"evaluation": number, "parameter": request.parameter, "multiplier": multiplier,
                "status": "success", "annual_electricity_kwh": annual,
                "model_path": str(model), "output_directory": evidence.output_directory,
                "meter_csv": evidence.meter_csv, "modifications": changes}
    except Exception as exc:
        return {"evaluation": number, "parameter": request.parameter, "multiplier": multiplier,
                "status": "simulation_failed", "objective": 1_000_000_000.0,
                "error": str(exc), "model_path": str(model),
                "output_directory": str(run_dir / "simulation")}


async def run_energyplus_study(
    kind: str, request: EnergyPlusStudyRequest, runner: EnergyPlusRunner, progress: Progress | None = None,
) -> dict:
    """Run a bounded, traceable study in isolated directories using real EnergyPlus."""
    request.validate(kind)
    started = time.monotonic()
    original = runner.resolve_file(request.idf_path, ".idf")
    weather = runner.resolve_file(request.epw_path, ".epw")
    root = Path(request.output_directory).resolve(); root.mkdir(parents=True, exist_ok=True)
    inspection = await runner.inspect_people(original) if request.parameter == "occupancy_multiplier" else None
    if kind == "sensitivity":
        candidates = [1.0, request.lower_bound, request.upper_bound]
    else:
        count = request.maximum_evaluations
        grid = [request.lower_bound + i * (request.upper_bound-request.lower_bound)/(count-1)
                for i in range(count)]
        candidates = [1.0, *grid]
    candidates = list(dict.fromkeys(round(value, 10) for value in candidates))[:request.maximum_evaluations]
    evaluations: list[dict] = []
    for number, multiplier in enumerate(candidates, 1):
        if time.monotonic() - started >= request.wall_clock_timeout_seconds:
            break
        if progress:
            await progress({"stage": "evaluating", "evaluation": number,
                            "maximum_evaluations": len(candidates), "multiplier": multiplier})
        row = await _evaluate(request, runner, original, weather, inspection, multiplier, number, root)
        evaluations.append(row)
        (root / "trace.json").write_text(json.dumps(evaluations, indent=2), encoding="utf-8")
    successful = [row for row in evaluations if row["status"] == "success"]
    if not successful:
        raise RuntimeError("Every EnergyPlus study evaluation failed")
    baseline = next((row for row in successful if row["multiplier"] == 1.0), None)
    if baseline is None:
        raise RuntimeError("EnergyPlus baseline evaluation failed")
    if kind == "optimization":
        best = min(successful, key=lambda row: row["annual_electricity_kwh"])
        result_detail = {"best": best, "improvement_percent": 100 *
                         (baseline["annual_electricity_kwh"]-best["annual_electricity_kwh"])
                         / baseline["annual_electricity_kwh"]}
    else:
        sensitivities = []
        for row in successful:
            if row is baseline: continue
            output_change = (row["annual_electricity_kwh"] - baseline["annual_electricity_kwh"]) / baseline["annual_electricity_kwh"]
            parameter_change = row["multiplier"] - 1.0
            sensitivities.append({"multiplier": row["multiplier"], "normalized_sensitivity": output_change / parameter_change,
                                  "annual_electricity_kwh": row["annual_electricity_kwh"]})
        result_detail = {"sensitivity_method": "one-at-a-time-normalized-elasticity", "effects": sensitivities}
    result = {"status": "success", "mode": f"energyplus-backed-real-{kind}",
              "parameter": request.parameter, "objective": "annual_electricity_kwh",
              "baseline": baseline, **result_detail, "evaluations": evaluations,
              "energyplus_evaluations": len(evaluations),
              "failed_evaluations": len(evaluations)-len(successful),
              "execution_time_seconds": round(time.monotonic()-started, 3),
              "provenance": {"original_model": str(original), "original_model_sha256": file_sha256(original),
                             "weather_file": str(weather), "weather_sha256": file_sha256(weather),
                             "meter": "Electricity:Facility", "request": asdict(request)}}
    (root / f"{kind}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if progress: await progress({"stage": "completed"})
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EngineeringStudyManager:
    """Persistent background manager for long real optimization/sensitivity jobs."""
    def __init__(self, root: Path, runner: EnergyPlusRunner):
        self.root = root.resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.runner = runner; self.tasks: dict[str, asyncio.Task] = {}
        for path in self.root.glob("*/status.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                if state.get("state") in {"queued", "running"}:
                    state.update({"state": "interrupted", "error": "Application restarted", "updated_at": _now()})
                    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            except Exception: pass

    def _dir(self, job_id: str) -> Path:
        if not job_id.startswith("study_") or not job_id.replace("_", "").isalnum(): raise KeyError("Invalid study job ID")
        return self.root / job_id

    def _write(self, job_id: str, state: dict) -> None:
        directory = self._dir(job_id); directory.mkdir(parents=True, exist_ok=True)
        state.update({"job_id": job_id, "updated_at": _now()})
        (directory / "status.json").write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    def status(self, job_id: str) -> dict:
        path = self._dir(job_id) / "status.json"
        if not path.is_file(): raise KeyError(f"Study job not found: {job_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    async def start(self, kind: str, request: EnergyPlusStudyRequest) -> dict:
        request.validate(kind); job_id = f"study_{secrets.token_hex(8)}"; directory = self._dir(job_id)
        request = replace(request, output_directory=str(directory))
        self._write(job_id, {"state": "queued", "kind": kind, "created_at": _now(), "request": asdict(request), "progress": {"stage": "queued"}})
        self.tasks[job_id] = asyncio.create_task(self._execute(job_id, kind, request))
        return self.status(job_id)

    async def _execute(self, job_id: str, kind: str, request: EnergyPlusStudyRequest) -> None:
        state = self.status(job_id); state["state"] = "running"; self._write(job_id, state)
        async def progress(payload: dict) -> None:
            current = self.status(job_id); current["progress"] = payload; self._write(job_id, current)
        try:
            result = await run_energyplus_study(kind, request, self.runner, progress)
            state = self.status(job_id); state.update({"state": "completed", "result": result, "completed_at": _now()})
        except Exception as exc:
            state = self.status(job_id); state.update({"state": "failed", "error": str(exc), "completed_at": _now()})
        self._write(job_id, state)

    def files(self, job_id: str) -> list[dict]:
        directory = self._dir(job_id)
        if not directory.is_dir(): raise KeyError(f"Study job not found: {job_id}")
        return [{"name": str(path.relative_to(directory)).replace("\\", "/"), "size": path.stat().st_size}
                for path in sorted(directory.rglob("*")) if path.is_file()]

    def file(self, job_id: str, relative: str) -> Path:
        directory = self._dir(job_id).resolve(); target = (directory / relative).resolve()
        if directory not in target.parents or not target.is_file(): raise KeyError("Study artifact not found")
        return target


def engineering_study_tools(manager: EngineeringStudyManager) -> list:
    @tool("run_energyplus_optimization")
    async def start_optimization(idf_path: str, epw_path: str, lower_bound: float = 0.7,
                                 upper_bound: float = 1.0, maximum_evaluations: int = 5) -> str:
        """Start REAL EnergyPlus lighting-multiplier optimization minimizing annual facility electricity; returns a background job ID."""
        return json.dumps(await manager.start("optimization", EnergyPlusStudyRequest(
            idf_path=idf_path, epw_path=epw_path, parameter="lighting_multiplier", lower_bound=lower_bound,
            upper_bound=upper_bound, maximum_evaluations=maximum_evaluations)))

    @tool("run_energyplus_sensitivity")
    async def start_sensitivity(idf_path: str, epw_path: str, parameter: str = "lighting_multiplier",
                                lower_bound: float = 0.8, upper_bound: float = 1.2) -> str:
        """Start REAL EnergyPlus one-at-a-time sensitivity for lighting or occupancy multiplier; returns a background job ID."""
        return json.dumps(await manager.start("sensitivity", EnergyPlusStudyRequest(
            idf_path=idf_path, epw_path=epw_path, parameter=parameter, lower_bound=lower_bound,
            upper_bound=upper_bound, maximum_evaluations=3)))

    @tool("get_energyplus_study_job")
    def get_study(job_id: str) -> str:
        """Get progress or numerical results for a real EnergyPlus optimization/sensitivity job."""
        return json.dumps(manager.status(job_id))
    return [start_optimization, start_sensitivity, get_study]
