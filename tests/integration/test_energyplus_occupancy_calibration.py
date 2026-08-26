"""Heavy real-EnergyPlus calibration proof; opt in with RUN_ENERGYPLUS_INTEGRATION=1."""
import asyncio
import csv
import os
from pathlib import Path

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

from energy_agent.calibration.occupancy import OccupancyCalibrationRequest, calibrate_occupancy
from energy_agent.engineering_studies import EnergyPlusStudyRequest, run_energyplus_study
from energy_agent.simulation.runner import EnergyPlusRunner


pytestmark = pytest.mark.skipif(os.getenv("RUN_ENERGYPLUS_INTEGRATION") != "1", reason="heavy EnergyPlus integration test")


def test_calibration_recovers_hidden_real_energyplus_occupancy(tmp_path):
    asyncio.run(_run(tmp_path))


def test_real_energyplus_lighting_optimization_uses_simulated_energy(tmp_path):
    asyncio.run(_run_lighting_optimization(tmp_path))


async def _runner(tmp_path: Path):
    root = Path(os.environ["ENERGY_AGENT_INTEGRATION_ROOT"]).resolve()
    mcp_root = root / "EnergyPlus-MCP" / "energyplus-mcp-server"
    config = {"url": os.environ["ENERGYPLUS_MCP_URL"], "transport": "streamable_http",
              "headers": {"Authorization": f"Bearer {os.environ['ENERGYPLUS_MCP_TOKEN']}"}}
    tools = await MultiServerMCPClient({"energyplus": config}).get_tools()
    return EnergyPlusRunner(tools, [root, mcp_root, tmp_path]), mcp_root


async def _run_lighting_optimization(tmp_path: Path):
    runner, mcp_root = await _runner(tmp_path)
    request = EnergyPlusStudyRequest(
        idf_path=str(mcp_root / "sample_files" / "5ZoneAirCooled_with_outputs.idf"),
        epw_path=str(mcp_root / "sample_files" / "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"),
        lower_bound=0.7, upper_bound=1.0, maximum_evaluations=3,
        output_directory=str(tmp_path / "optimization"), wall_clock_timeout_seconds=1800,
    )
    result = await run_energyplus_study("optimization", request, runner)
    assert result["mode"] == "energyplus-backed-real-optimization"
    assert result["best"]["multiplier"] == pytest.approx(0.7)
    assert result["best"]["annual_electricity_kwh"] < result["baseline"]["annual_electricity_kwh"]
    assert result["energyplus_evaluations"] == 3
    assert result["failed_evaluations"] == 0


async def _run(tmp_path: Path):
    runner, mcp_root = await _runner(tmp_path)
    model = mcp_root / "sample_files" / "5ZoneAirCooled_with_outputs.idf"
    weather = mcp_root / "sample_files" / "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"
    inspection = await runner.inspect_people(model)
    hidden_raw = tmp_path / "reference" / "hidden_075_raw.idf"
    hidden_model = tmp_path / "reference" / "hidden_075.idf"
    await runner.apply_occupancy_multiplier(model, hidden_raw, inspection, 0.75)
    await runner.configure_hourly_electricity(hidden_raw, hidden_model)
    reference = await runner.simulate_electricity(hidden_model, weather, tmp_path / "reference" / "simulation", 2026)
    measured_csv = tmp_path / "measured_reference_test_only.csv"
    with measured_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["timestamp", "electricity"])
        writer.writerows((timestamp.strftime("%Y-%m-%d %H:%M:%S"), value)
                         for timestamp, value in zip(reference.series.timestamps, reference.series.values))

    request = OccupancyCalibrationRequest(
        idf_path=str(model), epw_path=str(weather), measured_data_path=str(measured_csv),
        lower_bound=0.5, upper_bound=1.5, maximum_evaluations=5,
        output_directory=str(tmp_path / "calibration"), wall_clock_timeout_seconds=3600,
    )
    result = await calibrate_occupancy(request, runner)
    assert result["calibrated"]["objective"] < result["baseline"]["objective"]
    assert result["calibrated"]["occupancy_multiplier"] == pytest.approx(0.75, abs=0.01)
    assert result["energyplus_evaluations"] == 5
    assert result["failed_evaluations"] == 0
