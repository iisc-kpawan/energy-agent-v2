import math

from energy_agent.analysis import (Parameter, Objective, calibration_metrics,
    guideline14_assessment, latin_hypercube, optimize, demo_optimization, demo_calibration, demo_sensitivity)
from energy_agent.orchestrator import MultiAgentOrchestrator


def test_calibration_metrics_are_exact_and_assessed():
    measured = [100, 120, 80]
    simulated = [100, 120, 80]
    metrics = calibration_metrics(measured, simulated)
    assert metrics["nmbe_percent"] == 0
    assert metrics["cvrmse_percent"] == 0
    assert guideline14_assessment(metrics, "monthly")["passed"] is True


def test_latin_hypercube_is_bounded_and_reproducible():
    parameters = [Parameter("x", 0, 1), Parameter("y", -2, 2)]
    first = latin_hypercube(parameters, 12, seed=7)
    assert first == latin_hypercube(parameters, 12, seed=7)
    assert all(0 <= row["x"] <= 1 and -2 <= row["y"] <= 2 for row in first)


def test_multiobjective_engine_returns_only_nondominated_rows():
    result = optimize([Parameter("x", 0, 1)], [Objective("low"), Objective("high")],
                      lambda p: {"low": p["x"], "high": 1-p["x"]}, evaluations=10)
    assert len(result["evaluations"]) == 10
    assert len(result["pareto_front"]) == 10


def test_demo_workflows_run_end_to_end():
    optimization = demo_optimization(12, 3)
    calibration = demo_calibration(15, 3)
    assert optimization["pareto_front"]
    assert math.isfinite(calibration["best"]["metrics"]["cvrmse_percent"])
    assert "passed" in calibration["assessment"]
    sensitivity = demo_sensitivity(6, 3)
    assert sensitivity["ranking"][0]["parameter"] == "lighting_multiplier"


def test_analysis_requests_route_to_new_specialists_and_qa():
    optimization = MultiAgentOrchestrator.route("optimize energy and peak cooling with a Pareto search")
    calibration = MultiAgentOrchestrator.route("calibrate against measured utility bills using NMBE and CVRMSE")
    assert "optimization_expert" in optimization and "sensitivity_analyst" in optimization and "qa_reviewer" in optimization
    assert "calibration_expert" in calibration and "sensitivity_analyst" in calibration and "qa_reviewer" in calibration
