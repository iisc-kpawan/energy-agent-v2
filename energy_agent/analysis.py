"""Deterministic optimization and calibration services for Energy Agent V2.

The LLM may configure and explain these services, but it never computes the
metrics or decides that an optimization converged.  Engines are evaluator-
agnostic so the same orchestration can use a fast test surrogate, BESOS, or an
EnergyPlus worker.
"""
from __future__ import annotations

import csv
import html
import json
import math
import os
import random
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from langchain_core.tools import tool


@dataclass(frozen=True)
class Parameter:
    name: str
    lower: float
    upper: float

    def validate(self) -> None:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper) or self.lower >= self.upper:
            raise ValueError(f"Invalid bounds for {self.name}: [{self.lower}, {self.upper}]")


@dataclass(frozen=True)
class Objective:
    name: str
    minimize: bool = True


def calibration_metrics(measured: Sequence[float], simulated: Sequence[float]) -> dict:
    """Return standard deterministic goodness-of-fit metrics.

    NMBE uses (n-1) degrees of freedom, as is conventional for calibrated BEM.
    Values are percentages except RMSE and MAE, which retain the input unit.
    """
    if len(measured) != len(simulated) or len(measured) < 2:
        raise ValueError("Measured and simulated series must have the same length >= 2")
    pairs = [(float(m), float(s)) for m, s in zip(measured, simulated)]
    if not all(math.isfinite(x) for pair in pairs for x in pair):
        raise ValueError("Calibration series contain non-finite values")
    mean = sum(m for m, _ in pairs) / len(pairs)
    if mean == 0: raise ValueError("Measured mean cannot be zero")
    errors = [s - m for m, s in pairs]
    n = len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    nmbe = 100.0 * sum(errors) / ((n - 1) * mean)
    cvrmse = 100.0 * math.sqrt(sum(e * e for e in errors) / (n - 1)) / mean
    mae = sum(abs(e) for e in errors) / n
    mape_values = [abs(e / m) for e, (m, _) in zip(errors, pairs) if m != 0]
    return {"count": n, "measured_mean": mean, "rmse": rmse, "mae": mae,
            "mape_percent": 100.0 * sum(mape_values) / len(mape_values) if mape_values else None,
            "nmbe_percent": nmbe, "cvrmse_percent": cvrmse}


def guideline14_assessment(metrics: dict, interval: str = "monthly") -> dict:
    """Assess conventional ASHRAE Guideline 14 calibration limits.

    Thresholds are configurable policy in production; these defaults are the
    commonly used monthly (5/15) and hourly (10/30) NMBE/CV(RMSE) limits.
    """
    limits = {"monthly": {"abs_nmbe_percent": 5.0, "cvrmse_percent": 15.0},
              "hourly": {"abs_nmbe_percent": 10.0, "cvrmse_percent": 30.0}}
    if interval not in limits: raise ValueError("interval must be monthly or hourly")
    limit = limits[interval]
    passed = abs(metrics["nmbe_percent"]) <= limit["abs_nmbe_percent"] and metrics["cvrmse_percent"] <= limit["cvrmse_percent"]
    return {"interval": interval, "limits": limit, "passed": passed,
            "nmbe_passed": abs(metrics["nmbe_percent"]) <= limit["abs_nmbe_percent"],
            "cvrmse_passed": metrics["cvrmse_percent"] <= limit["cvrmse_percent"]}


def latin_hypercube(parameters: Sequence[Parameter], count: int, seed: int = 42) -> list[dict[str, float]]:
    if not 1 <= count <= 5000: raise ValueError("sample count must be between 1 and 5000")
    for p in parameters: p.validate()
    rng = random.Random(seed)
    columns = []
    for p in parameters:
        values = [p.lower + (p.upper - p.lower) * ((i + rng.random()) / count) for i in range(count)]
        rng.shuffle(values); columns.append(values)
    return [{p.name: columns[j][i] for j, p in enumerate(parameters)} for i in range(count)]


def _dominates(a: dict, b: dict, objectives: Sequence[Objective]) -> bool:
    av = [a["objectives"][o.name] * (1 if o.minimize else -1) for o in objectives]
    bv = [b["objectives"][o.name] * (1 if o.minimize else -1) for o in objectives]
    return all(x <= y for x, y in zip(av, bv)) and any(x < y for x, y in zip(av, bv))


def pareto_front(rows: Sequence[dict], objectives: Sequence[Objective]) -> list[dict]:
    return [row for i, row in enumerate(rows) if not any(i != j and _dominates(other, row, objectives) for j, other in enumerate(rows))]


def optimize(parameters: Sequence[Parameter], objectives: Sequence[Objective], evaluator: Callable[[dict[str, float]], dict[str, float]], evaluations: int = 40, seed: int = 42) -> dict:
    """Budgeted multi-objective search using LHS exploration plus Pareto ranking."""
    if not objectives: raise ValueError("At least one objective is required")
    if not 2 <= evaluations <= 500: raise ValueError("evaluations must be between 2 and 500")
    rows = []
    for index, candidate in enumerate(latin_hypercube(parameters, evaluations, seed)):
        values = evaluator(candidate)
        missing = [o.name for o in objectives if o.name not in values]
        if missing: raise ValueError(f"Evaluator omitted objectives: {missing}")
        rows.append({"evaluation": index + 1, "parameters": candidate,
                     "objectives": {o.name: float(values[o.name]) for o in objectives}})
    front = pareto_front(rows, objectives)
    return {"algorithm": "latin-hypercube-pareto-v1", "seed": seed,
            "evaluation_budget": evaluations, "evaluations": rows, "pareto_front": front,
            "parameters": [asdict(p) for p in parameters], "objectives": [asdict(o) for o in objectives]}


def sensitivity_screen(parameters: Sequence[Parameter], evaluator: Callable[[dict[str, float]], float],
                       trajectories: int = 8, step_fraction: float = .05, seed: int = 42) -> dict:
    """Bounded elementary-effects screening to rank influential parameters."""
    if not 2 <= trajectories <= 100: raise ValueError("trajectories must be between 2 and 100")
    if not 0 < step_fraction <= .25: raise ValueError("step_fraction must be in (0, .25]")
    bases = latin_hypercube(parameters, trajectories, seed); effects = {p.name: [] for p in parameters}
    for base in bases:
        baseline = float(evaluator(base))
        for p in parameters:
            delta = (p.upper - p.lower) * step_fraction
            changed = dict(base); changed[p.name] = min(p.upper, base[p.name] + delta)
            actual = changed[p.name] - base[p.name]
            if actual == 0:
                changed[p.name] = max(p.lower, base[p.name] - delta); actual = changed[p.name] - base[p.name]
            effects[p.name].append((float(evaluator(changed)) - baseline) / actual)
    ranking = []
    for p in parameters:
        values = effects[p.name]; mean = sum(values)/len(values)
        ranking.append({"parameter": p.name, "mu": mean, "mu_star": sum(abs(v) for v in values)/len(values),
                        "sigma": math.sqrt(sum((v-mean)**2 for v in values)/len(values))})
    ranking.sort(key=lambda row: row["mu_star"], reverse=True)
    return {"method": "bounded-elementary-effects", "trajectories": trajectories,
            "evaluation_count": trajectories * (len(parameters)+1), "ranking": ranking, "seed": seed}


def demo_optimization(evaluations: int = 30, seed: int = 42) -> dict:
    """Deterministic runnable example mirroring BESOS orientation/WWR optimization."""
    params = [Parameter("north_axis_deg", 0.0, 359.0), Parameter("window_to_wall_ratio", 0.1, 0.9),
              Parameter("insulation_r_m2k_w", 1.0, 8.0)]
    objectives = [Objective("annual_energy_kwh"), Objective("peak_cooling_kw")]
    def evaluator(x):
        radians = math.radians(x["north_axis_deg"])
        solar = 1.0 + .18 * math.cos(radians - math.pi)
        wwr, insulation = x["window_to_wall_ratio"], x["insulation_r_m2k_w"]
        # Annual energy has a daylighting/glazing optimum while peak cooling
        # continues to increase with glass area, producing a real trade-off.
        return {"annual_energy_kwh": 115000 - 36000 * solar + 90000 * (wwr - .38) ** 2 + 38000 / insulation,
                "peak_cooling_kw": 27 * solar + 34 * wwr + 8 / math.sqrt(insulation)}
    result = optimize(params, objectives, evaluator, evaluations, seed)
    result["mode"] = "demonstration-surrogate-not-energyplus"
    return result


def calibrate(parameters: Sequence[Parameter], measured: Sequence[float], simulator: Callable[[dict[str, float]], Sequence[float]], evaluations: int = 60, interval: str = "monthly", seed: int = 42) -> dict:
    """Budgeted calibration minimizing absolute NMBE and CV(RMSE)."""
    objectives = [Objective("abs_nmbe_percent"), Objective("cvrmse_percent")]
    cache: dict[str, dict] = {}
    def evaluator(candidate):
        simulated = list(simulator(candidate)); metrics = calibration_metrics(measured, simulated)
        cache[json.dumps(candidate, sort_keys=True)] = {"metrics": metrics, "simulated": simulated}
        return {"abs_nmbe_percent": abs(metrics["nmbe_percent"]), "cvrmse_percent": metrics["cvrmse_percent"]}
    result = optimize(parameters, objectives, evaluator, evaluations, seed)
    ranked = sorted(result["evaluations"], key=lambda r: r["objectives"]["abs_nmbe_percent"] + r["objectives"]["cvrmse_percent"])
    best = ranked[0]; detail = cache[json.dumps(best["parameters"], sort_keys=True)]
    result.update({"best": {**best, **detail}, "assessment": guideline14_assessment(detail["metrics"], interval),
                   "interval": interval})
    return result


def demo_calibration(evaluations: int = 40, seed: int = 42) -> dict:
    measured = [118, 112, 104, 92, 82, 76, 79, 85, 94, 103, 111, 121]
    params = [Parameter("load_multiplier", .7, 1.3), Parameter("seasonal_amplitude", .5, 1.5)]
    base = [110, 106, 99, 89, 80, 75, 77, 83, 91, 98, 105, 113]
    def simulator(x):
        return [x["load_multiplier"] * value + x["seasonal_amplitude"] * 5 * math.cos(2*math.pi*i/12) for i, value in enumerate(base)]
    result = calibrate(params, measured, simulator, evaluations, "monthly", seed)
    result["mode"] = "demonstration-surrogate-not-energyplus"
    return result


def demo_sensitivity(trajectories: int = 8, seed: int = 42) -> dict:
    parameters = [Parameter("infiltration_multiplier", .5, 2), Parameter("lighting_multiplier", .5, 1.5),
                  Parameter("equipment_multiplier", .5, 1.5)]
    result = sensitivity_screen(parameters, lambda x: 50000 + 18000*x["infiltration_multiplier"] + 32000*x["lighting_multiplier"] + 9000*x["equipment_multiplier"]**2, trajectories, seed=seed)
    result["mode"] = "demonstration-surrogate-not-energyplus"; return result


def export_result(result: dict, directory: Path, name: str) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{name}.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = result.get("evaluations", [])
    csv_path = directory / f"{name}.csv"
    if rows:
        keys = ["evaluation"] + [f"parameter:{k}" for k in rows[0]["parameters"]] + [f"objective:{k}" for k in rows[0]["objectives"]]
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer=csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
            for row in rows:
                writer.writerow({"evaluation": row["evaluation"], **{f"parameter:{k}":v for k,v in row["parameters"].items()}, **{f"objective:{k}":v for k,v in row["objectives"].items()}})
    elif result.get("ranking"):
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result["ranking"][0])); writer.writeheader(); writer.writerows(result["ranking"])
    html_path = directory / f"{name}.html"
    report_rows = rows or result.get("ranking", [])
    if rows:
        headings = ["evaluation"] + list(rows[0]["parameters"]) + list(rows[0]["objectives"])
        flat_rows = [[row["evaluation"], *row["parameters"].values(), *row["objectives"].values()] for row in rows]
    elif report_rows:
        headings = list(report_rows[0]); flat_rows = [[row.get(key, "") for key in headings] for row in report_rows]
    else:
        headings, flat_rows = ["Result"], [[json.dumps(result.get("best", result), default=str)]]
    table = "<table><thead><tr>" + "".join(f"<th>{html.escape(str(x))}</th>" for x in headings) + "</tr></thead><tbody>" + "".join("<tr>"+"".join(f"<td>{html.escape(str(x))}</td>" for x in row)+"</tr>" for row in flat_rows) + "</tbody></table>"
    html_path.write_text(f"<!doctype html><meta charset='utf-8'><title>{html.escape(name)}</title><style>body{{font-family:system-ui;margin:2rem;background:#0b1220;color:#e5e7eb}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.55rem;border:1px solid #334155;text-align:left}}th{{background:#172554}}</style><h1>{html.escape(name)}</h1>{table}", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def _export_tool_result(result: dict, prefix: str) -> dict:
    directory = Path(os.getenv("ENERGY_AGENT_DATA_DIR", Path.cwd() / "runtime")) / "analysis"
    paths = export_result(result, directory, f"{prefix}-{secrets.token_hex(5)}")
    return {kind: f"/api/v2/analysis/files/{Path(path).name}" for kind, path in paths.items()}


@tool("run_surrogate_optimization_demo")
def run_optimization_demo(evaluations: int = 30, seed: int = 42) -> str:
    """DEMO ONLY: run algebraic surrogate optimization. This does not modify or run EnergyPlus."""
    result = demo_optimization(evaluations, seed)
    compact = {k: result[k] for k in ("mode", "algorithm", "evaluation_budget", "pareto_front")}
    compact["artifacts"] = _export_tool_result(result, "optimization-chat")
    return json.dumps(compact)


@tool("run_surrogate_calibration_demo")
def run_calibration_demo(evaluations: int = 40, seed: int = 42) -> str:
    """DEMO ONLY: calibrate hard-coded surrogate arrays. This does not use measured files or EnergyPlus."""
    result = demo_calibration(evaluations, seed)
    compact = {k: result[k] for k in ("mode", "best", "assessment")}
    compact["artifacts"] = _export_tool_result(result, "calibration-chat")
    return json.dumps(compact)


@tool
def calculate_calibration_metrics(measured: list[float], simulated: list[float], interval: str = "monthly") -> str:
    """Calculate NMBE, CV(RMSE), RMSE, MAE and MAPE for aligned measured/simulated series."""
    metrics = calibration_metrics(measured, simulated)
    return json.dumps({"metrics": metrics, "assessment": guideline14_assessment(metrics, interval)})


@tool("run_surrogate_sensitivity_demo")
def run_sensitivity_demo(trajectories: int = 8, seed: int = 42) -> str:
    """DEMO ONLY: screen an algebraic surrogate. This does not modify or run EnergyPlus."""
    result = demo_sensitivity(trajectories, seed); result["artifacts"] = _export_tool_result(result, "sensitivity-chat")
    return json.dumps(result)


def analysis_tools() -> list:
    return [run_optimization_demo, run_calibration_demo, calculate_calibration_metrics, run_sensitivity_demo]
