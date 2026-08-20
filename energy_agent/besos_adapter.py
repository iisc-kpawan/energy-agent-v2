"""Optional production adapter for BESOS/EnergyPlus optimization.

Install requirements-optimization.txt inside an EnergyPlus-capable worker.  The
web application deliberately imports BESOS only when a real job is requested,
so chat and calibration metrics remain lightweight.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def run_besos_optimization(idf_path: str, weather_path: str, parameters: list[dict[str, Any]],
                           objectives: list[str], evaluations: int = 40,
                           population_size: int = 20, output_directory: str = "besos-output"):
    if evaluations < 2 or evaluations > 500:
        raise ValueError("evaluations must be between 2 and 500")
    if population_size < 4 or population_size > 200:
        raise ValueError("population_size must be between 4 and 200")
    for path in (idf_path, weather_path):
        if not Path(path).is_file(): raise FileNotFoundError(path)
    try:
        from besos import eppy_funcs as ef
        from besos.evaluator import EvaluatorEP
        from besos.optimizer import NSGAII
        from besos.parameters import FieldSelector, Parameter as BesosParameter, RangeParameter, wwr
        from besos.problem import EPProblem
    except ImportError as exc:
        raise RuntimeError("Install requirements-optimization.txt in the EnergyPlus worker") from exc

    building = ef.get_building(idf_path)
    inputs = []
    for spec in parameters:
        bounds = RangeParameter(float(spec["lower"]), float(spec["upper"]))
        if spec.get("kind") == "wwr": inputs.append(wwr(bounds)); continue
        selector = FieldSelector(class_name=spec["class_name"], object_name=spec["object_name"],
                                 field_name=spec["field_name"])
        inputs.append(BesosParameter(selector, value_descriptor=bounds, name=spec.get("name")))
    problem = EPProblem(inputs=inputs, outputs=objectives)
    out = Path(output_directory); out.mkdir(parents=True, exist_ok=True)
    evaluator = EvaluatorEP(problem, building, epw_file=weather_path, out_dir=str(out), err_dir=str(out))
    # Fail fast with a midpoint candidate before spending the optimization budget.
    midpoint = [(float(p["lower"]) + float(p["upper"])) / 2 for p in parameters]
    evaluator(midpoint)
    return NSGAII(evaluator, evaluations=evaluations, population_size=population_size)
